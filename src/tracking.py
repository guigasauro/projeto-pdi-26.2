"""Rastreamento: ByteTrack (via supervision) sobre as detecções em cache → IDs persistentes.

Saída por frame: `np.ndarray` (N, 7) com colunas `x1, y1, x2, y2, conf, cls, id`.
"""
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from src.detection import foot_points

log = logging.getLogger(__name__)


def track(dets_per_frame: list[np.ndarray], cfg: dict, fps: float) -> list[np.ndarray]:
    """Associa as detecções entre frames (IoU + filtro de Kalman do ByteTrack). Frames pulados
    por `frame_step` não são passados ao tracker e ficam vazios. `cfg` é o YAML inteiro."""
    tcfg, step = cfg["tracking"], cfg["video"]["frame_step"]
    rate = fps / step
    with warnings.catch_warnings():
        # shortcut: sv.ByteTrack está deprecated (sai no supervision 0.31); versão fixada no requirements.txt
        warnings.simplefilter("ignore", FutureWarning)
        tracker = sv.ByteTrack(
            track_activation_threshold=tcfg["activation_conf"],
            lost_track_buffer=round(tcfg["lost_buffer_s"] * rate),
            minimum_matching_threshold=tcfg["match_thresh"],
            frame_rate=rate,
            minimum_consecutive_frames=tcfg["min_consecutive"],
        )
    out = []
    for i, dets in enumerate(dets_per_frame):
        if i % step:
            out.append(np.empty((0, 7)))
            continue
        d = sv.Detections(xyxy=dets[:, :4].reshape(-1, 4), confidence=dets[:, 4], class_id=dets[:, 5].astype(int))
        t = tracker.update_with_detections(d)
        out.append(np.hstack([t.xyxy, t.confidence[:, None], t.class_id[:, None], t.tracker_id[:, None]]).reshape(-1, 7))
    return out


def track_cached(dets_per_frame: list[np.ndarray], cfg: dict, fps: float, cache_path: str | Path) -> list[np.ndarray]:
    """`track` com cache .npz (mesmo formato do cache de detecções)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        log.info("tracks lidos do cache %s", cache_path)
        return [data[k] for k in sorted(data.files, key=int)]
    out = track(dets_per_frame, cfg, fps)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{str(i): t for i, t in enumerate(out)})
    return out


def history(tracks: list[np.ndarray]) -> dict[int, list[tuple[int, float, float]]]:
    """Histórico por ID: `{id: [(frame, x_pe, y_pe), ...]}` com o ponto do pé em pixels."""
    hist: dict[int, list[tuple[int, float, float]]] = {}
    for i, t in enumerate(tracks):
        for tid, (x, y) in zip(t[:, 6].astype(int), foot_points(t)):
            hist.setdefault(int(tid), []).append((i, float(x), float(y)))
    return hist


def id_color(tid: int) -> tuple[int, int, int]:
    """Cor BGR fixa por ID (matiz espalhado pela razão áurea, convertido de HSV)."""
    hue = int((tid * 0.618033988749895 % 1) * 180)
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def debug_view(img_bgr: np.ndarray, tracks: list[np.ndarray], frame: int, trail: int) -> np.ndarray:
    """Caixas e IDs do `frame`, com o rastro dos pés nos últimos `trail` frames."""
    view = img_bgr.copy()
    hist = history(tracks[max(0, frame - trail): frame + 1])
    for tid, pts in hist.items():
        p = np.array([(x, y) for _, x, y in pts], np.int32)
        cv2.polylines(view, [p], False, id_color(tid), 2)
    for x1, y1, x2, y2, _, _, tid in tracks[frame]:
        c = id_color(int(tid))
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(view, str(int(tid)), (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    return view
