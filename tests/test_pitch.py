import numpy as np

from src.pitch import draw_pitch, from_soccernet, keypoints, lines, to_px

L, W = 105.0, 68.0


def test_keypoints_symmetric_and_inside():
    kp = keypoints(L, W)
    for name, (x, y) in kp.items():
        assert 0 <= x <= L and 0 <= y <= W, name
        if name.endswith("_esq"):
            xd, yd = kp[name[:-3] + "dir"]
            assert np.isclose(x + xd, L) and np.isclose(y, yd), name   # espelhado no meio-campo
    assert np.allclose(kp["ga_frente_sup_esq"], (16.5, 34 - 20.16))
    assert np.allclose(kp["penalti_dir"], (94, 34))
    # meia-lua: a 9,15 m da marca do pênalti e sobre a linha da grande área
    assert np.isclose(np.hypot(*np.subtract(kp["meialua_sup_esq"], kp["penalti_esq"])), 9.15)


def test_lines_inside_pitch():
    for pl in lines(L, W):
        assert pl.min() >= -1e-9 and (pl[:, 0] <= L + 1e-9).all() and (pl[:, 1] <= W + 1e-9).all()


def test_draw_and_scale():
    img = draw_pitch(L, W, px_per_m=10, margin_m=3)
    assert img.shape == (740, 1110, 3)
    assert to_px([0, 0], 10, 3).tolist() == [30, 30]
    assert (img[30, 30:1080].min(axis=1) > 200).all()   # linha lateral superior inteira clara (anti-aliasing)
    assert (img[100, 500] == (40, 110, 40)).all()       # gramado


def test_nearest_keypoint():
    from src.pitch import nearest_keypoint
    kp = keypoints(L, W)
    assert nearest_keypoint(kp, (94.5, 33.0), 2.0) == "penalti_dir"
    assert nearest_keypoint(kp, (70.0, 20.0), 2.0) is None           # clique longe de qualquer ponto


def test_from_soccernet():
    assert from_soccernet([[-52.5, -34], [0, 0]], L, W).tolist() == [[0, 0], [52.5, 34]]
