"""Leitura de vídeo e amostragem de frames."""
import cv2
import numpy as np


def sample_frames(video_path: str, n: int) -> list[np.ndarray]:
    """Amostragem temporal uniforme: retorna `n` frames BGR igualmente espaçados no vídeo."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if ok:
            frames.append(frame_bgr)
    cap.release()
    return frames
