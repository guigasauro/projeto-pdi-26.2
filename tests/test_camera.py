import numpy as np

from src import camera

W, H = 1024, 576
C = np.array([52.0, 106.0, -18.0])


def _look_at(target: np.ndarray, f: float) -> np.ndarray:
    """Parâmetros de uma câmera em C olhando para `target` (sem roll)."""
    import cv2
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross([0, 0, -1.0], z); x /= np.linalg.norm(x)   # eixo x da imagem paralelo ao gramado
    R = np.stack([x, np.cross(z, x), z])                    # linhas: eixos da câmera no mundo
    return np.r_[cv2.Rodrigues(R)[0].ravel(), np.log(f)]


def test_decompose_recovers_center_and_params():
    p = _look_at(np.array([70.0, 30.0, 0.0]), 2000)
    Hm = camera.to_homography(p, C, W, H)
    q, Cq, resid = camera.decompose(Hm, W, H)
    assert resid < 1e-8 and np.allclose(Cq, C, atol=1e-4) and np.allclose(q, p, atol=1e-5)


def test_fit_points_from_collinear_clicks():
    p = _look_at(np.array([52.5, 34.0, 0.0]), 1800)
    pitch_pts = np.array([[52.5, 0.0], [52.5, 24.85], [52.5, 34.0], [52.5, 43.15]])   # só a linha do meio
    img = camera.project(p, C, W, H, pitch_pts)
    q = camera.fit_points(img, pitch_pts, C, W, H, _look_at(np.array([45.0, 40.0, 0.0]), 1500))
    assert np.abs(camera.project(q, C, W, H, [[30.0, 20.0], [70.0, 50.0]])
                  - camera.project(p, C, W, H, [[30.0, 20.0], [70.0, 50.0]])).max() < 1.0


def test_line_refinement_recovers_camera():
    import cv2
    from src import lines, pitch
    p = _look_at(np.array([60.0, 34.0, 0.0]), 1800)
    mask = np.zeros((H, W), np.uint8)
    for pl in pitch.lines(105, 68):
        cv2.polylines(mask, [np.round(camera.project(p, C, W, H, pl)).astype(np.int32)], False, 255, 3)
    cfg = {"stages": [[60, 10.0], [30, 5.0], [10, 2.0]], "prior_weight": 0.05, "max_iter": 50,
           "inlier_px": 2.0, "min_visible": 40}
    p0 = p + np.r_[0.01, -0.008, 0.005, 0.05]               # 8-33 px de erro nos pontos de teste
    q, score, n = lines.refine(p0, C, lines.distance_map(mask, 60), np.full((H, W), 255, np.uint8),
                               lines.model_samples(105, 68, 0.5), cfg)
    from src.homography import project
    probe = project(camera.to_homography(p, C, W, H), [[200, 300], [800, 400], [512, 550]])   # pontos visíveis
    assert score > 0.9 and np.abs(camera.project(q, C, W, H, probe) - camera.project(p, C, W, H, probe)).max() < 4.0   # linha de 3 px + viés do termo de regularização


def test_refine_center_from_noisy_clicks():
    rng = np.random.default_rng(0)
    pts = np.array([[x, y] for x in (0, 16.5, 36, 52.5, 69, 88.5, 105) for y in (0, 13.85, 34, 54.15, 68)], float)
    marks, inits = [], []
    for tx, f in [(15, 2200), (35, 1500), (52, 1200), (75, 1600), (92, 2400)]:
        p = _look_at(np.array([tx, 34.0, 0.0]), f)
        img = camera.project(p, C, W, H, pts)
        ok = (img[:, 0] > 0) & (img[:, 0] < W) & (img[:, 1] > 0) & (img[:, 1] < H)
        marks.append((img[ok] + rng.normal(0, 1.5, (ok.sum(), 2)), pts[ok]))
    C0 = C + [3.0, -5.0, 1.5]                                # chute do centro errado, como o da mediana
    inits = [camera.fit_points(pi, pp, C0, W, H, _look_at(np.array([52.5, 34.0, 0.0]), 1500)) for pi, pp in marks]
    Cq, _ = camera.refine_center(C0, marks, inits, W, H)
    assert np.linalg.norm(Cq - C) < 1.5
