"""Orquestra o pipeline: detecção → tracking → times → homografia → posições no campo → heatmaps.
Resultados intermediários ficam em `<out>/cache/` (YOLO e descritores não são recalculados)."""
import logging
from pathlib import Path

import cv2
import numpy as np

from src import detection, heatmap, homography, team_classifier, tracking
from src.detection import foot_points

log = logging.getLogger(__name__)


def _load_or(cache: Path, fn) -> list[np.ndarray]:
    """Lê uma lista de arrays por frame do .npz, ou calcula com `fn()` e salva."""
    if cache.exists():
        data = np.load(cache)
        return [data[k] for k in sorted(data.files, key=int)]
    out = fn()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{str(i): a for i, a in enumerate(out)})
    return out


def team_labels(video: str, tracks: list[np.ndarray], cfg: dict, fps: float) -> list[np.ndarray]:
    """Rótulo de time por detecção rastreada, já com a votação em janela."""
    tc = cfg["team"]
    descs = team_classifier.extract(video, tracks, cfg)
    labels = team_classifier.classify(team_classifier.calibrate(descs, tc), descs)
    win = None if tc["vote_window_s"] is None else round(tc["vote_window_s"] * fps)
    return team_classifier.vote(tracks, labels, win)


def positions(tracks: list[np.ndarray], labels: list[np.ndarray], Hs: np.ndarray, point: str = "foot") -> np.ndarray:
    """Posições no campo (N, 5): frame, id, time, X, Y, com a homografia do próprio quadro `Hs[f]`.
    Quadros sem homografia válida (NaN) ficam de fora.
    `point` = `foot` (centro da base da bbox) ou `center` (centro da bbox, para a ablação)."""
    rows = []
    for f, (t, lab) in enumerate(zip(tracks, labels)):
        if not len(t) or f >= len(Hs) or not np.isfinite(Hs[f]).all():
            continue
        p = foot_points(t) if point == "foot" else np.c_[(t[:, 0] + t[:, 2]) / 2, (t[:, 1] + t[:, 3]) / 2]
        rows.append(np.c_[np.full(len(t), f), t[:, 6], lab, homography.project(Hs[f], p)])
    return np.vstack(rows) if rows else np.empty((0, 5))


def team_heatmaps(pos: np.ndarray, n_frames: int, cfg: dict) -> dict[int, np.ndarray]:
    """Heatmap suavizado por time, normalizado pelo número de frames (jogadores·tempo por célula)."""
    pc, hc = cfg["pitch"], cfg["heatmap"]
    out = {}
    for team in sorted({int(t) for t in pos[:, 2] if t >= 0}):
        grid = heatmap.accumulate(pos[pos[:, 2] == team, 3:], pc["length_m"], pc["width_m"], hc["cell_m"])
        out[team] = heatmap.smooth(grid / n_frames, hc["sigma_m"] / hc["cell_m"], hc["smoothing"])
    return out


def player_heatmaps(pos: np.ndarray, min_frames: int, cfg: dict) -> dict[int, np.ndarray]:
    """Heatmap por `tracker_id` com pelo menos `min_frames` posições, normalizado para soma 1
    (distribuição do tempo daquele ID no campo). IDs fragmentados viram mapas separados."""
    pc, hc = cfg["pitch"], cfg["heatmap"]
    out = {}
    for tid in np.unique(pos[:, 1]).astype(int):
        p = pos[pos[:, 1] == tid]
        if len(p) >= min_frames:
            g = heatmap.accumulate(p[:, 3:], pc["length_m"], pc["width_m"], hc["cell_m"])
            out[tid] = heatmap.smooth(g / max(g.sum(), 1), hc["sigma_m"] / hc["cell_m"], hc["smoothing"])
    return out


def homographies(video: str, H0: np.ndarray, frame0: int, dets: list[np.ndarray], cfg: dict,
                 cache: Path) -> np.ndarray:
    """Homografia imagem→campo de cada quadro (N, 3, 3), NaN onde não há estimativa válida.
    `manual` (v1): H0 fixa até `valid_until_frame`; `propagate` (v2): propagação a partir de H0."""
    hc, n = cfg["homography"], len(dets)
    Hs = np.full((n, 3, 3), np.nan)
    if hc["mode"] == "manual":
        Hs[frame0:min(hc["valid_until_frame"], n - 1) + 1] = H0
        return Hs
    if cache.exists():
        return np.load(cache)
    Hs[frame0:], _ = homography.propagate(video, H0, frame0, n, dets, cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, Hs)
    return Hs


def run(video: str, cfg: dict, out_dir: str | Path, annotation: str | Path) -> dict:
    """Pipeline completo; salva os heatmaps em `<out>/` e devolve os intermediários."""
    out_dir = Path(out_dir)
    cache = out_dir / "cache"
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
    dets = detection.detect_video(video, cfg, cache / "detections.npz")
    tracks = tracking.track_cached(dets, cfg, fps, cache / "tracks.npz")
    labels = _load_or(cache / "teams.npz", lambda: team_labels(video, tracks, cfg, fps))
    H, ann = homography.load(annotation)
    Hs = homographies(video, H, ann["frame"], dets, cfg, cache / "homographies.npy")
    valid = np.isfinite(Hs).all((1, 2))
    last = int(np.flatnonzero(valid).max())
    pos = positions(tracks, labels, Hs)
    n_frames = int(valid[::cfg["video"]["frame_step"]].sum())
    maps = team_heatmaps(pos, n_frames, cfg)
    log.info("%d posições em %d frames com homografia (0..%d); times: %s", len(pos), n_frames, last, list(maps))
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = max(m.max() for m in maps.values())
    for team, m in maps.items():
        cv2.imwrite(str(out_dir / f"heatmap_time{team}.png"), heatmap.colorize(m, cfg, vmax=vmax))
    if len(maps) == 2:
        cv2.imwrite(str(out_dir / "heatmap_diferenca.png"),
                    heatmap.colorize(maps[0] - maps[1], cfg, cmap=cfg["heatmap"]["diff_colormap"], signed=True))
    players = player_heatmaps(pos, round(cfg["heatmap"]["min_player_s"] * fps / cfg["video"]["frame_step"]), cfg)
    tiles = []
    for tid, m in players.items():
        team = int(np.bincount(pos[pos[:, 1] == tid, 2].astype(int) + 1).argmax()) - 1   # +1: rótulo -1 vira 0
        t = heatmap.colorize(m, cfg)
        cv2.putText(t, f"ID {tid} | time {team}", (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(t)
    if tiles:
        cols = 4
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(out_dir / "heatmap_jogadores.png"), cv2.resize(grid, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    log.info("heatmaps: %d times, %d jogadores (IDs com >= %.0f s)", len(maps), len(players), cfg["heatmap"]["min_player_s"])
    return {"fps": fps, "tracks": tracks, "labels": labels, "Hs": Hs, "positions": pos, "maps": maps,
            "n_frames": n_frames, "last_frame": last}
