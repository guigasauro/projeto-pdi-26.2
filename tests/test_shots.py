import numpy as np

from src.shots import cuts, hsv_hist, is_main, mmss, segments

SC = {"cut_thresh": 0.3, "cut_min_frames": 5, "board_thresh": 0.8, "board_median_frames": 5,
      "main_grass_min": 0.5, "main_board_min": 0.5, "main_people_min": 6, "main_height_max": 0.208}


def test_mmss():
    assert mmss("7:31") == 451 and mmss("0:05.5") == 5.5


def test_cuts_ignore_close_peaks():
    d = np.zeros(100); d[[30, 32, 70]] = 0.9          # 32 está a menos de 5 quadros do corte em 30
    assert cuts(d, 0.3, 5) == [(0, 30), (30, 70), (70, 100)]


def test_scoreboard_change_splits_but_blink_does_not():
    board = np.ones(100); board[40:60] = 0.2; board[80] = 0.2     # replay sem placar em 40..59; piscada em 80
    assert segments(np.zeros(100), board, SC) == [(0, 40), (40, 60), (60, 100)]


def test_identical_frames_have_zero_distance():
    import cv2
    img = np.random.default_rng(0).integers(0, 255, (36, 64, 3), np.uint8)
    h = hsv_hist(img, [16, 4, 4])
    assert np.isclose(h.sum(), 1) and cv2.compareHist(h, h, cv2.HISTCMP_BHATTACHARYYA) < 1e-6


def test_main_rule():
    assert is_main(0.8, 1.0, 18, 55 / 576, SC)
    assert not is_main(0.8, 1.0, 2, 400 / 576, SC)  # close-up
    assert not is_main(0.8, 0.0, 18, 55 / 576, SC)  # replay da câmera principal (sem placar)
