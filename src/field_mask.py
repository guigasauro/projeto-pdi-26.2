"""Segmentação do gramado: limiarização em HSV + morfologia matemática."""
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def green_mask(img_bgr: np.ndarray, h_range: list[int], s_min: int, v_min: int) -> np.ndarray:
    """Sistemas de cores + operação pontual: converte BGR→HSV e limiariza H na faixa do verde,
    exigindo S e V mínimos (descarta pixels acinzentados/escuros cujo H é instável)."""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_range[0], s_min, v_min], np.uint8)
    upper = np.array([h_range[1], 255, 255], np.uint8)
    return cv2.inRange(img_hsv, lower, upper)


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morfologia: abertura (remove ruído verde fora do campo), fechamento (une falhas pequenas),
    depois mantém só o maior componente conexo com os buracos preenchidos (jogadores e linhas
    brancas ficam dentro da máscara do campo)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    field = np.zeros_like(mask)
    if not contours:
        log.warning("nenhum pixel de gramado encontrado")
        return field
    cv2.drawContours(field, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return field


def field_mask(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Máscara binária (0/255) do campo. `cfg` é a seção `field_mask` do YAML."""
    raw = green_mask(img_bgr, cfg["h_range"], cfg["s_min"], cfg["v_min"])
    return clean_mask(raw, cfg["morph_kernel"])


def hue_histogram(frames_bgr: list[np.ndarray], s_min: int, v_min: int) -> np.ndarray:
    """Histograma do canal H (180 bins) dos pixels com S e V acima do mínimo, somado nos frames.
    O pico verde justifica `h_range`."""
    hist = np.zeros(180)
    for img_bgr in frames_bgr:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        valid = cv2.inRange(img_hsv, np.array([0, s_min, v_min], np.uint8), np.array([179, 255, 255], np.uint8))
        hist += cv2.calcHist([img_hsv], [0], valid, [180], [0, 180]).ravel()
    return hist


def debug_view(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Painel 2x2: original | limiarização HSV crua | após morfologia | máscara sobreposta (verde)."""
    raw = green_mask(img_bgr, cfg["h_range"], cfg["s_min"], cfg["v_min"])
    field = clean_mask(raw, cfg["morph_kernel"])
    overlay = img_bgr.copy()
    overlay[field > 0] = (0.5 * overlay[field > 0] + (0, 127, 0)).astype(np.uint8)
    overlay[field == 0] //= 3
    to_bgr = lambda m: cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return np.vstack([np.hstack([img_bgr, to_bgr(raw)]), np.hstack([to_bgr(field), overlay])])
