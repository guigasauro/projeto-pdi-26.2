import cv2
import numpy as np
import yaml

from src import camera, lines, pitch

W, H = 1280, 720
C = np.array([52.5, 80.0, -14.0])          # câmera acima do gramado (Z para dentro do gramado)


def _look_at(target: list[float], f: float) -> np.ndarray:
    """Câmera em C olhando para `target`, sem roll, eixo x da imagem paralelo ao gramado."""
    z = np.asarray(target, float) - C; z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    return np.r_[cv2.Rodrigues(np.stack([x, np.cross(z, x), z]))[0].ravel(), np.log(f)]


def test_relocalize_finds_lost_camera():
    lc = yaml.safe_load(open("configs/default.yaml"))["lines"]
    true = _look_at([85.0, 30.0, 0.0], 2200)
    mask = np.zeros((H, W), np.uint8)
    for pl in pitch.lines(105, 68):
        q = camera.project(true, C, W, H, np.asarray(pl, float))
        cv2.polylines(mask, [q.round().astype(np.int32)], False, 255, 3)
    dt, region = lines.distance_map(mask, lc["max_px"]), np.full((H, W), 255, np.uint8)
    grid = lines.pose_grid([_look_at([30.0, 40.0, 0.0], 1500), _look_at([95.0, 20.0, 0.0], 2800)], lc)
    p, score = lines.relocalize(grid, C, dt, region, lines.model_samples(105, 68, lc["sample_m"]), lc)
    grid_pts = np.mgrid[0:106:5, 0:69:4].reshape(2, -1).T.astype(float)
    q = camera.project(true, C, W, H, grid_pts)
    inside = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)   # só o que a câmera vê
    err = np.linalg.norm(camera.project(p, C, W, H, grid_pts[inside]) - q[inside], axis=1)
    assert score > 0.9 and inside.sum() > 20 and np.median(err) < 5 and err.max() < 10   # ~1% de zoom de folga
