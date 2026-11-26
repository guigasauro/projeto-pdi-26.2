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


def test_last_defender_votes():
    from src.match import last_defender_votes
    cfg = {"gk_zone_m": 6.0, "deep_zone_m": 30.0, "min_players": 4}
    # quadro 0: time 1 defende a esquerda (mais recuado fora da pequena área), goleiro do time 0 ignorado
    pos = np.array([[0, 1, 0, 3.0, 34], [0, 2, 1, 12.0, 30], [0, 3, 0, 20.0, 40], [0, 4, 1, 25.0, 20]])
    v = last_defender_votes(pos, cfg, 105.0)
    assert v.tolist() == [[0, 1], [0, 0]]


def test_outside_frames_and_goalkeeper():
    from src.match import goalkeeper_tracks, outside_frames
    cfg = {"max_outside_m": 3.0, "max_outside_frac": 0.25, "gk_zone_m": 6.0, "gk_min_frac": 0.5, "gk_min_rows": 3}
    # quadro 0: 1 de 4 fora (técnico na lateral) fica; quadro 1: 3 de 4 fora (pose falsa) sai
    pos = np.array([[0, 1, 0, 50, 30], [0, 2, 1, 60, 30], [0, 3, 0, 70, 30], [0, 4, 1, 50, 75],
                    [1, 1, 0, 50, 80], [1, 2, 1, 60, 80], [1, 3, 0, 70, 80], [1, 4, 1, 50, 30]], float)
    assert outside_frames(pos, 105, 68, cfg).tolist() == [1]
    # ID 7 é o mais recuado junto ao gol esquerdo nos dois quadros; ID 8 (atacante) fica à frente dele
    pos = np.array([[0, 7, 0, 1.0, 34], [0, 8, 1, 4.0, 30], [1, 7, 0, 2.0, 35], [1, 8, 1, 5.0, 36]], float)
    assert goalkeeper_tracks(pos, 105, 68, cfg) == {7: 0}
    # bola parada: defensor 9 na linha à frente do goleiro 7 em 2 de 3 quadros, depois sai; 7 fica os 3 quadros
    pos = np.array([[0, 9, 1, 0.5, 30], [0, 7, 0, 2.0, 34], [1, 9, 1, 0.5, 31], [1, 7, 0, 1.5, 34],
                    [2, 7, 0, 1.0, 35], [2, 9, 1, 8.0, 30]], float)
    assert goalkeeper_tracks(pos, 105, 68, cfg)[7] == 0


def test_pitch_for_match_uses_match_dimensions():
    from src import pitch
    cfg = {"pitch": {"length_m": 105, "width_m": 68, "px_per_m": 10}}
    c = pitch.for_match(cfg, {"campo": [101, 68]})
    assert (c["pitch"]["length_m"], c["pitch"]["width_m"]) == (101, 68) and cfg["pitch"]["length_m"] == 105
    assert pitch.for_match(cfg, {})["pitch"]["length_m"] == 105
