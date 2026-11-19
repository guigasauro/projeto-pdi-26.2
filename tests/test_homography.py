import numpy as np
import pytest

from src.homography import compute_homography, dlt_homography, load, project, reprojection_error, save

# Homografia "verdadeira" sintética: câmera em perspectiva olhando o campo
H_TRUE = np.array([[0.05, 0.01, -10.0], [0.002, 0.12, -40.0], [0.00001, 0.0015, 1.0]])
PTS_IMG = np.array([[100, 300], [1800, 320], [1700, 1000], [200, 980], [960, 600], [500, 700], [1400, 450]], float)
PTS_PITCH = project(H_TRUE, PTS_IMG)


def test_dlt_recovers_synthetic_homography():
    H = dlt_homography(PTS_IMG[:4], PTS_PITCH[:4])                       # 4 pontos: solução exata
    assert np.allclose(H, H_TRUE / H_TRUE[2, 2], rtol=1e-6, atol=1e-9)
    assert reprojection_error(H, PTS_IMG, PTS_PITCH).max() < 1e-6        # vale para os outros 3 também


def test_dlt_matches_opencv_and_ransac_rejects_outlier():
    noisy = PTS_PITCH + np.random.default_rng(0).normal(0, 0.05, PTS_PITCH.shape)
    bad = noisy.copy(); bad[4] += (8, -5)                                 # um clique errado
    H_cv, inl = compute_homography(PTS_IMG, bad, ransac_thresh_m=1.0)
    assert not inl[4] and inl.sum() == 6
    H_dlt = dlt_homography(PTS_IMG[inl], bad[inl])
    assert np.abs(project(H_dlt, PTS_IMG) - project(H_cv, PTS_IMG)).max() < 0.05


def test_dlt_needs_four_points():
    with pytest.raises(ValueError):
        dlt_homography(PTS_IMG[:3], PTS_PITCH[:3])


def test_save_load_roundtrip(tmp_path):
    save(tmp_path / "c.json", H_TRUE, PTS_IMG[:2], ["a", "b"], frame=0)
    H, data = load(tmp_path / "c.json")
    assert np.allclose(H, H_TRUE) and [p["name"] for p in data["points"]] == ["a", "b"]
