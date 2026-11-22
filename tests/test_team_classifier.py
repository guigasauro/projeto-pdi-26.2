import numpy as np

from src.team_classifier import calibrate, classify, descriptor, torso_crop, vote

CFG = {"descriptor": "hs_hist", "hs_bins": [16, 8], "min_pixels": 20, "calib_frames": 10, "k": 2, "seed": 0}
MASK = {"h_range": [33, 85], "s_min": 40, "v_min": 25}


def shirt(bgr: tuple[int, int, int], grass_frac: float = 0.5) -> np.ndarray:
    """Recorte 20x20: camisa de cor `bgr` com uma faixa de grama à esquerda."""
    img = np.full((20, 20, 3), bgr, np.uint8)
    img[:, : int(20 * grass_frac)] = (40, 140, 40)
    return img


def test_torso_crop():
    img = np.zeros((100, 100, 3), np.uint8)
    assert torso_crop(img, np.array([10, 0, 60, 100]), [0.2, 0.6, 0.2, 0.8]).shape == (40, 30, 3)


def test_grass_removed_from_descriptor():
    d_grass = descriptor(shirt((0, 0, 200)), CFG, MASK)
    d_clean = descriptor(shirt((0, 0, 200), 0.0), CFG, MASK)
    assert np.allclose(d_grass, d_clean)                         # a grama não entra no histograma
    assert descriptor(shirt((0, 0, 200), 1.0), CFG, MASK) is None  # recorte só de grama: sem rótulo


def test_kmeans_separates_red_and_blue():
    descs = [np.array([descriptor(shirt((0, 0, 200)), CFG, MASK), descriptor(shirt((200, 0, 0)), CFG, MASK)])] * 10
    lab = classify(calibrate(descs, CFG), descs)
    assert lab[0][0] != lab[0][1] and all((l == lab[0]).all() for l in lab)


def test_vote_window_fixes_isolated_error_but_keeps_id_switch():
    tracks = [np.array([[0, 0, 1, 1, 1, 0, 7]], float)] * 40
    labels = [np.array([0])] * 20 + [np.array([1])] * 20     # ID 7 troca de jogador (e de time) no frame 20
    labels[5] = np.array([1])                                # erro isolado
    w = vote(tracks, labels, window=3)
    assert w[5][0] == 0 and w[30][0] == 1                    # corrige o erro e respeita a troca
    assert vote(tracks, labels)[30][0] == vote(tracks, labels)[5][0]   # track inteiro: um rótulo só
