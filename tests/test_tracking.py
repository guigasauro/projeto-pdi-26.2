import numpy as np

from src.tracking import history, track

CFG = {"video": {"frame_step": 2},
       "tracking": {"activation_conf": 0.5, "lost_buffer_s": 3.0, "match_thresh": 0.8, "min_consecutive": 1}}


def two_players(n: int) -> list[np.ndarray]:
    """Dois jogadores andando em sentidos opostos, sem se cruzar."""
    return [np.array([[100 + 2 * i, 100, 130 + 2 * i, 180, 0.9, 0],
                      [500 - 2 * i, 300, 530 - 2 * i, 380, 0.9, 0]], float) for i in range(n)]


def test_track_keeps_two_ids_and_skips_frames():
    tracks = track(two_players(40), CFG, fps=25.0)
    assert all(len(t) == 0 for t in tracks[1::2])            # frames pulados ficam vazios
    ids = {tuple(sorted(t[:, 6].astype(int))) for t in tracks[2::2]}
    assert len(ids) == 1 and len(next(iter(ids))) == 2        # mesmos 2 IDs do começo ao fim


def test_history_foot_points():
    tracks = track(two_players(40), CFG, fps=25.0)
    hist = history(tracks)
    assert len(hist) == 2
    frame, x, y = hist[min(hist)][-1]
    assert frame % 2 == 0 and y in (180.0, 380.0)            # pé = base da bbox
