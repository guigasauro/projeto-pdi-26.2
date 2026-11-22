"""Classificação de time pela cor do uniforme: recorte do tronco → remoção de grama → descritor de cor
→ k-means (calibração nos primeiros frames) → votação temporal por ID.

Rótulo por detecção: 0..k-1 = cluster, -1 = sem descritor (recorte só de grama) ou outlier.
"""
import logging

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.field_mask import green_mask

log = logging.getLogger(__name__)


def torso_crop(img_bgr: np.ndarray, box: np.ndarray, torso: list[float]) -> np.ndarray:
    """Recorte do tronco: frações `[y0, y1, x0, x1]` da bbox (evita cabeça, shorts e a grama dos lados)."""
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    ya, yb = int(y1 + torso[0] * h), int(y1 + torso[1] * h)
    xa, xb = int(x1 + torso[2] * w), int(x1 + torso[3] * w)
    H, W = img_bgr.shape[:2]
    return img_bgr[max(0, ya):min(H, yb), max(0, xa):min(W, xb)]


def descriptor(crop_bgr: np.ndarray, cfg_team: dict, cfg_mask: dict | None) -> np.ndarray | None:
    """Descritor de cor dos pixels do recorte que não são grama (máscara HSV do gramado, se `cfg_mask`).
    `rgb_mean`: média BGR (referência ingênua).
    `lab_mean`: média L*a*b* (sistema de cores com luminância separada da cromaticidade).
    `ab_mean`: só a*b* da média Lab (descarta L: sol/sombra no mesmo time variam ±50 em L).
    `hs_hist`: histograma 2D H-S normalizado (processamento de histograma; ignora V = brilho)."""
    if crop_bgr.size == 0:
        return None
    keep = np.full(crop_bgr.shape[:2], 255, np.uint8)
    if cfg_mask is not None:
        keep = cv2.bitwise_not(green_mask(crop_bgr, cfg_mask["h_range"], cfg_mask["s_min"], cfg_mask["v_min"]))
    if cv2.countNonZero(keep) < cfg_team["min_pixels"]:
        return None
    if cfg_team["descriptor"] == "rgb_mean":
        return np.array(cv2.mean(crop_bgr, keep)[:3])
    if cfg_team["descriptor"] in ("lab_mean", "ab_mean"):
        lab = np.array(cv2.mean(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB), keep)[:3])
        return lab if cfg_team["descriptor"] == "lab_mean" else lab[1:]
    hb, sb = cfg_team["hs_bins"]
    hist = cv2.calcHist([cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)], [0, 1], keep, [hb, sb], [0, 180, 0, 256])
    return hist.ravel() / hist.sum()


def extract(video_path: str, tracks: list[np.ndarray], cfg: dict) -> list[np.ndarray]:
    """Descritores de todas as detecções rastreadas: lista por frame de (N, D), NaN onde não há descritor."""
    tc = cfg["team"]
    cfg_mask = cfg["field_mask"] if tc["remove_grass"] else None
    cap = cv2.VideoCapture(video_path)
    out, dim = [], None
    for t in tqdm(tracks, desc="descritores de cor"):
        ok, img_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"vídeo terminou antes dos tracks ({len(out)} frames)")
        ds = [descriptor(torso_crop(img_bgr, b, tc["torso"]), tc, cfg_mask) for b in t]
        dim = dim or next((len(d) for d in ds if d is not None), None)
        out.append(np.array([d if d is not None else np.full(dim or 1, np.nan) for d in ds]).reshape(len(t), -1))
    cap.release()
    return [d if d.shape[1] == dim else np.full((len(d), dim), np.nan) for d in out]


def calibrate(descs: list[np.ndarray], cfg_team: dict) -> KMeans:
    """k-means com os descritores dos primeiros `calib_frames` frames (k = número de grupos de cor)."""
    X = np.vstack(descs[:cfg_team["calib_frames"]])
    X = X[~np.isnan(X).any(1)]
    log.info("calibração: %d descritores", len(X))
    return KMeans(cfg_team["k"], n_init=10, random_state=cfg_team["seed"]).fit(X)


def classify(km: KMeans, descs: list[np.ndarray]) -> list[np.ndarray]:
    """Cluster do centróide mais próximo, por detecção (-1 sem descritor)."""
    out = []
    for d in descs:
        lab = np.full(len(d), -1)
        ok = ~np.isnan(d).any(1)
        if ok.any():
            lab[ok] = km.predict(d[ok])
        out.append(lab)
    return out


def vote(tracks: list[np.ndarray], labels: list[np.ndarray], window: int | None = None) -> list[np.ndarray]:
    """Votação temporal por ID: moda dos rótulos do mesmo ID a até `window` frames de distância
    (filtro de moda 1D ao longo do track). `None` = moda do track inteiro."""
    seq: dict[int, list[tuple[int, int, int]]] = {}       # id → [(frame, índice na linha, rótulo)]
    for f, (t, lab) in enumerate(zip(tracks, labels)):
        for j, (tid, l) in enumerate(zip(t[:, 6].astype(int), lab)):
            seq.setdefault(int(tid), []).append((f, j, int(l)))
    out = [np.full(len(t), -1, int) for t in tracks]
    w = np.inf if window is None else window
    for s in seq.values():
        fr = np.array([f for f, _, _ in s]); lab = np.array([l for _, _, l in s])
        for f, j, _ in s:
            near = lab[(lab >= 0) & (np.abs(fr - f) <= w)]
            out[f][j] = np.bincount(near).argmax() if len(near) else -1
    return out


def debug_view(crops: dict[int, list[np.ndarray]], n: int = 12, size: tuple[int, int] = (40, 64)) -> np.ndarray:
    """Mosaico: uma linha por cluster com `n` recortes de tronco de exemplo."""
    rows = []
    for c in sorted(crops):
        cs = [cv2.resize(x, size, interpolation=cv2.INTER_NEAREST) for x in crops[c][:n]]
        cs += [np.zeros((size[1], size[0], 3), np.uint8)] * (n - len(cs))
        rows.append(np.hstack(cs))
    return np.vstack(rows)
