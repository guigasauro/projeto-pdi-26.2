import numpy as np

from src.detection import filter_detections, foot_points

CFG = {"conf": 0.3, "min_area": 150, "foot_margin": 10}


def test_foot_points():
    dets = np.array([[10, 20, 30, 80, 0.9, 0]], float)
    assert foot_points(dets).tolist() == [[20, 80]]


def test_filter_detections():
    mask = np.zeros((200, 200), np.uint8)
    mask[100:] = 255                                   # gramado só na metade de baixo
    dets = np.array([
        [10, 100, 30, 150, 0.9, 0],                    # ok
        [10, 20, 30, 80, 0.9, 0],                      # pé na arquibancada (y=80, 20 px fora)
        [50, 45, 70, 95, 0.9, 0],                      # pé 5 px fora: tolerado pela margem
        [10, 100, 30, 150, 0.1, 0],                    # confiança baixa
        [10, 140, 15, 150, 0.9, 0],                    # área 50 < 150
        [150, 160, 190, 200, 0.9, 0],                  # pé na borda inferior (y2 = altura)
    ], float)
    kept = filter_detections(dets, mask, CFG)
    assert kept.tolist() == dets[[0, 2, 5]].tolist()


def test_filter_detections_empty():
    assert filter_detections(np.empty((0, 6)), np.zeros((10, 10), np.uint8), CFG).shape == (0, 6)
