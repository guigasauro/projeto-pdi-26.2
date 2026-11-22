"""Segmentação de uma transmissão em tomadas e classificação de cada tomada (câmera principal ao vivo
ou não: close-up, torcida, replay, gráfico). Só as tomadas da câmera principal entram no heatmap.

Por quadro (medidas baratas, sem YOLO):
- histograma HSV 3D → distância de Bhattacharyya ao quadro anterior (salto = corte de câmera);
- fração de gramado (limiarização HSV da máscara do campo, sem morfologia);
- placar na tela: máximo da correlação normalizada com um recorte do placar (casamento de padrões).
"""
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.field_mask import green_mask

log = logging.getLogger(__name__)


def mmss(t: str) -> float:
    """'7:31' → 451.0 s."""
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def hsv_hist(img_small_bgr: np.ndarray, bins: list[int]) -> np.ndarray:
    """Histograma HSV 3D normalizado (soma 1): descreve a distribuição de cores do quadro inteiro."""
    hsv = cv2.cvtColor(img_small_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    return h / h.sum()


def scoreboard_score(img_bgr: np.ndarray, template_bgr: np.ndarray, roi: list[int]) -> float:
    """Casamento de padrões: máximo da correlação normalizada (TM_CCOEFF_NORMED) do recorte do placar
    dentro da região `roi` = [x0, y0, x1, y1]. ~1 = placar na tela."""
    x0, y0, x1, y1 = roi
    return float(cv2.matchTemplate(img_bgr[y0:y1, x0:x1], template_bgr, cv2.TM_CCOEFF_NORMED).max())


def scan(video: str, start_s: float, end_s: float, cfg: dict, template_bgr: np.ndarray, roi: list[int],
         cache_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Percorre todos os quadros de [start_s, end_s) e devolve, por quadro: `frame` (índice no vídeo),
    `dist` (Bhattacharyya ao anterior), `grass` (fração de gramado) e `board` (placar)."""
    if cache_path and Path(cache_path).exists():
        return dict(np.load(cache_path))
    sc, mc = cfg["shots"], cfg["field_mask"]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0, f1 = int(round(start_s * fps)), int(round(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out = {k: np.zeros(f1 - f0) for k in ("dist", "grass", "board")}
    prev = None
    for i in tqdm(range(f1 - f0), desc=f"tomadas {Path(video).name}"):
        ok, img = cap.read()
        if not ok:
            out = {k: v[:i] for k, v in out.items()}
            break
        small = cv2.resize(img, tuple(sc["small_size"]), interpolation=cv2.INTER_AREA)
        h = hsv_hist(small, sc["hist_bins"])
        out["dist"][i] = 0.0 if prev is None else cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)
        prev = h
        out["grass"][i] = green_mask(small, mc["h_range"], mc["s_min"], mc["v_min"]).mean() / 255
        out["board"][i] = scoreboard_score(img, template_bgr, roi)
    out["frame"] = np.arange(f0, f0 + len(out["dist"]))
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
    return out


def cuts(dist: np.ndarray, thresh: float, min_len: int) -> list[tuple[int, int]]:
    """Tomadas [a, b) entre cortes: quadros com distância ao anterior > `thresh`. Cortes a menos de
    `min_len` quadros do anterior são ignorados (flash, transição de vários quadros)."""
    starts = [0]
    for i in np.flatnonzero(dist > thresh):
        if i - starts[-1] >= min_len:
            starts.append(int(i))
    return list(zip(starts, starts[1:] + [len(dist)]))


def segments(dist: np.ndarray, board: np.ndarray, cfg_shots: dict) -> list[tuple[int, int]]:
    """Tomadas: cortes de câmera (histograma) + trocas de presença do placar. As transições graduais
    (fusões) dos replays não dão salto no histograma, mas o placar some no replay e volta depois.
    A presença do placar passa por um filtro de mediana 1D para ignorar piscadas."""
    on = _median1d(board > cfg_shots["board_thresh"], cfg_shots["board_median_frames"] | 1)
    dist = dist.copy()
    dist[1:][on[1:] != on[:-1]] = np.inf          # troca do placar = corte
    return cuts(dist, cfg_shots["cut_thresh"], cfg_shots["cut_min_frames"])


def _median1d(x: np.ndarray, k: int) -> np.ndarray:
    """Filtro de mediana 1D de um sinal binário (= votação da maioria numa janela de k amostras)."""
    c = np.convolve(x.astype(float), np.ones(k), mode="same")
    return c > k / 2


def probe(cap: cv2.VideoCapture, frame: int, model, cfg: dict, device: str) -> tuple[int, float]:
    """YOLO + filtro do gramado no quadro `frame`: (nº de pessoas, altura mediana em px)."""
    from src import detection
    from src.field_mask import field_mask
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    if not ok:
        return 0, 0.0
    d = detection.filter_detections(detection.detect_frame(model, img, cfg["detection"], device),
                                    field_mask(img, cfg["field_mask"]), cfg["detection"])
    return len(d), float(np.median(d[:, 3] - d[:, 1])) if len(d) else 0.0


def is_main(grass: float, board: float, people: int, height: float, cfg_shots: dict) -> bool:
    """Regra da câmera principal ao vivo (limiares justificados no YAML)."""
    return (grass >= cfg_shots["main_grass_min"] and board >= cfg_shots["main_board_min"]
            and people >= cfg_shots["main_people_min"] and 0 < height <= cfg_shots["main_height_max"])


def timeline_view(table: list[dict], n_frames: int, fps: float, width: int = 1800) -> np.ndarray:
    """Linha do tempo: verde = câmera principal ao vivo, cinza = outras; marcas a cada 5 min."""
    img = np.full((60, width, 3), 60, np.uint8)
    for s in table:
        x0, x1 = int(s["a"] / n_frames * width), max(int(s["a"] / n_frames * width) + 1, int(s["b"] / n_frames * width))
        img[10:50, x0:x1] = (60, 180, 60) if s["main"] else (150, 150, 150)
    for m in range(0, int(n_frames / fps / 60) + 1, 5):
        x = int(m * 60 * fps / n_frames * width)
        cv2.line(img, (x, 50), (x, 59), (255, 255, 255), 1)
        cv2.putText(img, f"{m}'", (x + 2, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return img
