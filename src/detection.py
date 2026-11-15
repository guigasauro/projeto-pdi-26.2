"""Detecção de pessoas com YOLO (única etapa de deep learning) + filtros clássicos.

Saída por frame: `np.ndarray` (N, 6) com colunas `x1, y1, x2, y2, conf, cls` em pixels.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.field_mask import field_mask

log = logging.getLogger(__name__)


def load_model(cfg: dict):
    """Carrega o YOLO de `models/`. `cfg` é a seção `detection` do YAML."""
    from ultralytics import YOLO  # import tardio: torch é pesado e só esta etapa precisa dele

    return YOLO(cfg["model"])


def resolve_device(device: str) -> str:
    """`auto` → `cuda` se houver GPU, senão `cpu`; qualquer outro valor passa direto."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def foot_points(dets: np.ndarray) -> np.ndarray:
    """Ponto do pé = centro da base da bbox `((x1+x2)/2, y2)`. Shape (N, 2)."""
    return np.stack([(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]], axis=1)


def filter_detections(dets: np.ndarray, mask_field: np.ndarray, cfg: dict) -> np.ndarray:
    """Remove detecções de baixa confiança, pequenas demais e cujo pé não está no gramado
    (consulta pontual na máscara binária: tira torcida, banco e placas). A máscara é dilatada
    em `foot_margin` px (morfologia) para tolerar a borda serrilhada do campo junto às placas."""
    if len(dets) == 0:
        return dets.reshape(0, 6)
    m = cfg["foot_margin"]
    if m > 0:
        mask_field = cv2.dilate(mask_field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * m + 1, 2 * m + 1)))
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    h, w = mask_field.shape
    feet = foot_points(dets)
    fx = np.clip(feet[:, 0].astype(int), 0, w - 1)
    fy = np.clip(feet[:, 1].astype(int), 0, h - 1)
    keep = (dets[:, 4] >= cfg["conf"]) & (area >= cfg["min_area"]) & (mask_field[fy, fx] > 0)
    return dets[keep]


def detect_frame(model, img_bgr: np.ndarray, cfg: dict, device: str) -> np.ndarray:
    """YOLO bruto (sem filtro do gramado) → (N, 6)."""
    res = model.predict(img_bgr, conf=cfg["conf"], classes=cfg["classes"], imgsz=cfg["imgsz"],
                        device=device, verbose=False)[0]
    b = res.boxes
    return np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]])


def detect_video(video_path: str, cfg: dict, cache_path: str | Path | None = None) -> list[np.ndarray]:
    """Detecções filtradas para cada frame do vídeo (índice da lista = índice do frame;
    frames pulados por `frame_step` ficam vazios). Se `cache_path` existir, lê dele em vez de
    rodar o YOLO; senão roda e salva (.npz). `cfg` é o YAML inteiro."""
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        log.info("detecções lidas do cache %s", cache_path)
        return [data[k] for k in sorted(data.files, key=int)]

    dcfg, step = cfg["detection"], cfg["video"]["frame_step"]
    model, device = load_model(dcfg), resolve_device(dcfg["device"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    out = []
    for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc="detecção"):
        ok, img_bgr = cap.read()
        if not ok:
            break
        if i % step:
            out.append(np.empty((0, 6)))
            continue
        dets = detect_frame(model, img_bgr, dcfg, device)
        out.append(filter_detections(dets, field_mask(img_bgr, cfg["field_mask"]), dcfg))
    cap.release()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{str(i): d for i, d in enumerate(out)})
    return out


def debug_view(img_bgr: np.ndarray, dets_raw: np.ndarray, mask_field: np.ndarray, cfg: dict) -> np.ndarray:
    """Detecções mantidas (verde) e descartadas pelo filtro (vermelho), com o ponto do pé;
    fora do gramado escurecido."""
    kept = filter_detections(dets_raw, mask_field, cfg)
    kept_set = {tuple(d) for d in kept}
    view = img_bgr.copy()
    view[mask_field == 0] //= 2
    for d in dets_raw:
        color = (0, 255, 0) if tuple(d) in kept_set else (0, 0, 255)
        x1, y1, x2, y2 = d[:4].astype(int)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        cv2.circle(view, ((x1 + x2) // 2, y2), 4, color, -1)
    return view
