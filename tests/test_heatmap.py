import cv2
import numpy as np

from src.heatmap import accumulate, gaussian_kernel, grid_shape, smooth_frequency, smooth_opencv, smooth_spatial


def test_accumulate_counts_and_drops_outside():
    g = accumulate([[0.1, 0.1], [0.4, 0.2], [52.6, 34.1], [-1, 5], [105, 10], [104.99, 67.99]], 105, 68, 0.5)
    assert g.shape == grid_shape(105, 68, 0.5) == (136, 210)
    assert g[0, 0] == 2 and g[68, 105] == 1 and g[135, 209] == 1 and g.sum() == 4   # 2 pontos fora do campo


def test_kernel_normalized_and_symmetric():
    k = gaussian_kernel(2.0)
    assert k.shape == (13, 13) and np.isclose(k.sum(), 1) and np.allclose(k, k.T) and k.argmax() == 6 * 13 + 6


def test_manual_convolution_matches_opencv_and_frequency():
    rng = np.random.default_rng(0)
    g = np.zeros((60, 80)); g[rng.integers(10, 50, 50), rng.integers(10, 70, 50)] = 1   # longe da borda (raio 3σ = 9)
    ref = smooth_spatial(g, 3.0)
    assert np.allclose(ref, smooth_opencv(g, 3.0), atol=1e-12)
    assert np.abs(ref - smooth_frequency(g, 3.0)).max() < 0.01 * ref.max()   # truncamento do kernel em 3σ
    assert np.isclose(ref.sum(), g.sum(), rtol=1e-3)                        # suavizar conserva a massa


def test_rotate_half_is_180_degrees():
    from src.match import rotate_half
    pos = np.array([[0, 1, 0, 10.0, 5.0], [3, 2, 1, 52.5, 34.0]])
    out = rotate_half(pos, 105, 68)
    assert out[:, 3:].tolist() == [[95.0, 63.0], [52.5, 34.0]] and (out[:, :3] == pos[:, :3]).all()
