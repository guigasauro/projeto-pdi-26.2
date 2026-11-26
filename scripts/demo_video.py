"""Vídeo demo de uma tomada da partida: quadro original | posições no campo (top-view) | heatmap acumulado.

  python scripts/demo_video.py data/annotations/barcelona-x-real_madrid.json --segundos 60 \
      --nomes "Barcelona,Real Madrid" -o outputs/videos/demo_final.mp4

Sem --tempo/--quadro, usa a tomada marcada de duração mais próxima de --segundos; com eles, a tomada que
contém o quadro, a partir dele. Tudo vem do cache de outputs/<partida>/ (rode antes scripts/run_match.py;
o pipeline da partida roda de novo a partir do cache, ~1,5 min, e reescreve as mesmas saídas).
Só os quadros processados (1 a cada `match.frame_step`) têm detecção e câmera: o vídeo sai nessa taxa
(10 quadros/s numa transmissão de 30 fps), em tempo real.
No painel original, o modelo do campo é projetado com a homografia inversa (campo → imagem) do quadro.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import heatmap, homography, match, pitch, team_classifier  # noqa: E402

TEAM_BGR = {0: (170, 60, 20), 1: (245, 245, 245), -1: (140, 140, 140)}   # time 0 azul, time 1 branco
LINE_BGR = (0, 230, 255)


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Faixa escura com o título do painel (cv2.putText não desenha acentos)."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (25, 25, 25), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit_h(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (round(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def draw_original(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, H: np.ndarray | None,
                  L: float, W: float) -> np.ndarray:
    out = img.copy()
    if H is not None:
        Hinv = np.linalg.inv(H)                                  # campo → imagem
        for pl in pitch.lines(L, W):
            q = np.clip(homography.project(Hinv, np.asarray(pl, float)), -1e5, 1e5)
            cv2.polylines(out, [q.round().astype(np.int32)], False, LINE_BGR, 1, cv2.LINE_AA)
    for b, lab in zip(boxes, labels):
        x1, y1, x2, y2 = np.round(b[:4]).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), TEAM_BGR[int(lab)], 2)
    return out


def draw_topview(pos: np.ndarray, cfg: dict) -> np.ndarray:
    pc = cfg["pitch"]
    img = pitch.draw_pitch(pc["length_m"], pc["width_m"], pc["px_per_m"], pc["margin_m"])
    r = max(4, round(pc["px_per_m"] * 0.8))
    for _, _, team, X, Y in pos:
        x, y = np.round(pitch.to_px([X, Y], pc["px_per_m"], pc["margin_m"])).astype(int)
        cv2.circle(img, (x, y), r, TEAM_BGR[int(team)], -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def pick_shot(shots: list[dict], tempo: str | None, quadro: int | None, n_target: int) -> tuple[dict, int]:
    """Tomada e índice inicial (relativo a frame_a, múltiplo do passo de amostragem já aplicado em frame_a)."""
    if tempo is not None and quadro is not None:
        s = next((s for s in shots if s["tempo"] == tempo and s["frame_a"] <= quadro < s["frame_b"]), None)
        if s is None:
            sys.exit(f"nenhuma tomada processada contém o quadro {quadro} do tempo {tempo}")
        return s, quadro - s["frame_a"]
    marked = [s for s in shots if not s.get("auto")]
    return min(marked, key=lambda s: abs(s["frame_b"] - s["frame_a"] - n_target)), 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--segundos", type=float, default=60.0)
    ap.add_argument("--tempo", default=None, help="tempo da partida (1 ou 2); usar com --quadro")
    ap.add_argument("--quadro", type=int, default=None, help="quadro do vídeo do tempo onde o demo começa")
    ap.add_argument("--nomes", default="time 0,time 1", help="nomes dos times 0 e 1, separados por vírgula")
    ap.add_argument("--altura", type=int, default=432, help="altura dos painéis (px)")
    ap.add_argument("-o", "--out", default="outputs/videos/demo_final.mp4")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    base = yaml.safe_load(open(args.config))
    M_info = json.load(open(args.match))
    cfg = match.match_cfg(base, M_info)
    pc, hc, tc, step = cfg["pitch"], cfg["heatmap"], cfg["team"], cfg["video"]["frame_step"]
    L, W = pc["length_m"], pc["width_m"]
    names = args.nomes.split(",")

    r = match.run(args.match, base)
    fps = r["shots"][0]["fps"]
    s, i0 = pick_shot(r["shots"], args.tempo, args.quadro, round(args.segundos * fps))
    i0 -= i0 % step
    n = min(s["frame_b"] - s["frame_a"], i0 + round(args.segundos * fps))
    labels = team_classifier.vote(s["tracks"], team_classifier.classify(r["kmeans"], s["descs"]),
                                  round(tc["vote_window_s"] * fps))
    print(f"tomada {s['tempo']}:{s['frame_a']} ({'sem marcação' if s.get('auto') else 'marcada'}), "
          f"quadros {s['frame_a'] + i0}–{s['frame_a'] + n} ({(n - i0) / fps:.1f} s)")

    cap = cv2.VideoCapture(M_info["tempos"][s["tempo"]]["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame_a"] + i0)
    grids = {t: np.zeros(heatmap.grid_shape(L, W, hc["cell_m"])) for t in (0, 1)}
    sigma = hc["sigma_m"] / hc["cell_m"]
    hh = args.altura // 2
    writer, written, no_cam = None, 0, 0
    for i in range(i0, n):
        if (i - i0) % step:
            cap.grab()
            continue
        ok, img = cap.read()
        if not ok:
            break
        H = s["Hs"][i] if np.isfinite(s["Hs"][i]).all() else None
        pos = s["pos"][s["pos"][:, 0] == i]
        for t in (0, 1):
            if len(pos[pos[:, 2] == t]):
                grids[t] += heatmap.accumulate(pos[pos[:, 2] == t, 3:], L, W, hc["cell_m"])
        maps = {t: heatmap.smooth_opencv(g, sigma) for t, g in grids.items()}
        vmax = max(m.max() for m in maps.values()) or None

        orig = label(fit_h(draw_original(img, s["tracks"][i], labels[i], H, L, W), args.altura),
                     "Transmissao: deteccao (YOLO), times por cor e linhas do campo projetadas")
        top = draw_topview(pos, cfg)
        top = label(fit_h(top, args.altura), "Posicoes no campo (homografia)" if H is not None else "sem camera neste quadro")
        heat = np.vstack([label(fit_h(heatmap.colorize(maps[t], cfg, vmax=vmax), hh), f"Mapa de calor: {names[t]}")
                          for t in (0, 1)])
        heat = cv2.resize(heat, (heat.shape[1], args.altura)) if heat.shape[0] != args.altura else heat
        frame = np.hstack([orig, top, heat])
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        if writer is None:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / step,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        written += 1
        no_cam += H is None
    if writer is None:
        sys.exit("nenhum quadro escrito")
    writer.release()
    print(f"{args.out}: {written} quadros a {fps / step:.0f} fps ({written * step / fps:.1f} s), "
          f"{frame.shape[1]}x{frame.shape[0]}; {no_cam} quadros sem câmera")


if __name__ == "__main__":
    main()
