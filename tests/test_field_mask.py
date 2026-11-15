from pathlib import Path

import cv2
import numpy as np
import yaml

from src.field_mask import field_mask, hue_histogram

CFG = yaml.safe_load(open(Path(__file__).parents[1] / "configs/default.yaml"))["field_mask"]


def synthetic_frame() -> np.ndarray:
    """Arquibancada cinza no topo, gramado verde embaixo, jogador vermelho e linha branca no campo,
    e uma placa verde pequena na arquibancada (ruído que deve sumir)."""
    img_bgr = np.full((200, 300, 3), 128, np.uint8)
    img_bgr[60:] = (40, 140, 40)
    cv2.rectangle(img_bgr, (100, 100), (115, 140), (0, 0, 200), -1)
    cv2.line(img_bgr, (0, 160), (299, 160), (255, 255, 255), 2)
    img_bgr[10:13, 10:13] = (40, 140, 40)
    return img_bgr


def test_field_mask_synthetic():
    mask = field_mask(synthetic_frame(), CFG)
    assert mask[:55].max() == 0              # arquibancada e placa fora
    assert mask[70:].min() == 255            # gramado inteiro, jogador e linha preenchidos


def test_field_mask_player_cut_at_bottom_border():
    img_bgr = synthetic_frame()
    img_bgr[170:, 200:230] = (0, 0, 200)     # jogador cortado pela borda inferior
    mask = field_mask(img_bgr, CFG)
    assert mask[170:, 200:230].min() == 255


def test_field_mask_empty():
    assert field_mask(np.full((50, 50, 3), 128, np.uint8), CFG).max() == 0


def test_hue_histogram_peak_in_range():
    hist = hue_histogram([synthetic_frame()], CFG["s_min"], CFG["v_min"])
    assert CFG["h_range"][0] <= hist.argmax() <= CFG["h_range"][1]
