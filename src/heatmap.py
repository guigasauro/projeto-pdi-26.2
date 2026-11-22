"""Mapa de calor: acumulação dos pés numa grade do campo → suavização gaussiana (espacial ou na
frequência) → pseudocoloração sobre o desenho do campo.

Grade: célula `cell_m` metros; linha = Y, coluna = X (mesma orientação do top-view de `pitch.py`).
"""
import cv2
import matplotlib
import numpy as np

from src import pitch


def grid_shape(length: float, width: float, cell_m: float) -> tuple[int, int]:
    """Amostragem do campo: (linhas, colunas) da grade discreta."""
    return int(round(width / cell_m)), int(round(length / cell_m))


def accumulate(pts_pitch: np.ndarray, length: float, width: float, cell_m: float) -> np.ndarray:
    """Quantização das posições (X, Y) em metros: conta quantas caem em cada célula.
    Pontos fora do campo são descartados."""
    rows, cols = grid_shape(length, width, cell_m)
    grid = np.zeros((rows, cols))
    p = np.asarray(pts_pitch, float).reshape(-1, 2)
    p = p[(p[:, 0] >= 0) & (p[:, 0] < length) & (p[:, 1] >= 0) & (p[:, 1] < width)]
    c = np.minimum((p[:, 0] / cell_m).astype(int), cols - 1)
    r = np.minimum((p[:, 1] / cell_m).astype(int), rows - 1)
    np.add.at(grid, (r, c), 1)
    return grid


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Kernel gaussiano 2D separável, raio 3σ, soma 1 (σ em células)."""
    r = int(np.ceil(3 * sigma))
    g = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Versão didática da convolução 2D com borda zero: soma das cópias deslocadas da imagem
    ponderadas pelo kernel (um laço por elemento do kernel, não por pixel).
    Para kernel simétrico, convolução = correlação."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)))
    out = np.zeros_like(img, dtype=float)
    for i in range(kh):
        for j in range(kw):
            out += kernel[kh - 1 - i, kw - 1 - j] * pad[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def smooth_spatial(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Suavização espacial com a convolução manual."""
    return convolve(grid, gaussian_kernel(sigma))


def smooth_opencv(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Mesma suavização com cv2.GaussianBlur (referência), mesmo tamanho de kernel e borda zero."""
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(grid, (k, k), sigma, borderType=cv2.BORDER_CONSTANT)


def smooth_frequency(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Suavização no domínio da frequência: DFT → filtro passa-baixa gaussiano
    H(u, v) = exp(-2π²σ²(u² + v²)) (transformada de uma gaussiana de desvio σ) → DFT inversa.
    A grade é preenchida com zeros (3σ de cada lado) para a convolução circular da DFT não
    misturar bordas opostas do campo."""
    r = int(np.ceil(3 * sigma))
    pad = np.pad(grid, r)
    u = np.fft.fftfreq(pad.shape[0])[:, None]
    v = np.fft.fftfreq(pad.shape[1])[None, :]
    H = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (u ** 2 + v ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(pad) * H))[r:-r, r:-r]


def smooth(grid: np.ndarray, sigma: float, method: str) -> np.ndarray:
    """`spatial` (convolução manual) | `frequency` (DFT)."""
    return smooth_spatial(grid, sigma) if method == "spatial" else smooth_frequency(grid, sigma)


def colorize(heat: np.ndarray, cfg: dict, cmap: str | None = None, vmax: float | None = None,
             signed: bool = False) -> np.ndarray:
    """Pseudocoloração: normaliza o mapa para [0, 1] (ou [-1, 1] se `signed`), aplica a correção gama,
    o colormap
    (tabela de cores) e mistura sobre o desenho do campo com opacidade proporcional à intensidade.
    `cfg` é o YAML inteiro."""
    pc, hc = cfg["pitch"], cfg["heatmap"]
    L, W, ppm, mg = pc["length_m"], pc["width_m"], pc["px_per_m"], pc["margin_m"]
    base = pitch.draw_pitch(L, W, ppm, mg)
    vmax = vmax or (np.abs(heat).max() or 1.0)
    norm = np.clip(heat / vmax, -1, 1)
    size = (int(round(L * ppm)), int(round(W * ppm)))
    norm = cv2.resize(norm, size, interpolation=cv2.INTER_LINEAR)
    norm = np.sign(norm) * np.abs(norm) ** hc["gamma"]     # operação pontual: realça os valores baixos
    lut_in = (norm + 1) / 2 if signed else norm
    rgb = matplotlib.colormaps[cmap or hc["colormap"]](lut_in)[..., :3]
    color = (rgb[..., ::-1] * 255).astype(np.float32)
    alpha = (np.abs(norm) * hc["alpha"])[..., None]
    o = int(round(mg * ppm))
    roi = base[o:o + size[1], o:o + size[0]].astype(np.float32)
    base[o:o + size[1], o:o + size[0]] = (roi * (1 - alpha) + color * alpha).astype(np.uint8)
    # linhas por cima do mapa, para o campo continuar legível
    lines = pitch.draw_pitch(L, W, ppm, mg)
    white = (lines == 255).all(2)
    base[white] = 255
    return base
