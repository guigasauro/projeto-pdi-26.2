"""Linhas do campo: máscara das linhas brancas e refinamento da homografia por alinhamento às linhas.

A propagação (v2) acumula deriva em tomadas longas. Aqui a homografia de cada quadro é corrigida
para que o desenho do modelo do campo, projetado na imagem, caia sobre as linhas brancas reais:
1. top-hat morfológico na luminância (L do Lab): realça estruturas claras mais finas que o elemento
   estruturante (linhas de ~10 px), removendo o gradiente de iluminação do gramado;
2. limiarização + máscara do gramado erodida − caixas dos jogadores (uniformes brancos também passam no top-hat);
3. transformada de distância: cada pixel guarda a distância até a linha branca mais próxima;
4. mínimos quadrados nos 4 parâmetros da câmera PTZ (`camera`): minimiza a distância dos pontos do
   modelo projetados até a linha mais próxima, com um termo que penaliza mudar a câmera sem evidência.
"""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from src import camera, pitch


def line_mask(img_bgr: np.ndarray, mask_field: np.ndarray, dets: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Máscara binária das linhas brancas (top-hat em L ∩ gramado erodido − jogadores) e a região
    onde uma linha poderia ser vista (gramado erodido − jogadores)."""
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    k = cfg["tophat_px"]
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    e = cfg["mask_erode"]
    m = cv2.erode(mask_field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    p = cfg["player_margin"]
    for x1, y1, x2, y2 in (np.asarray(dets, float)[:, :4] if len(dets) else []):
        m[max(0, int(y1) - p):int(y2) + p, max(0, int(x1) - p):int(x2) + p] = 0
    return ((th > cfg["tophat_thresh"]) & (m > 0)).astype(np.uint8) * 255, m


def distance_map(mask_lines: np.ndarray, max_px: float) -> np.ndarray:
    """Transformada de distância euclidiana até a linha mais próxima, saturada em `max_px` (uint8, ≤ 255)."""
    return np.minimum(cv2.distanceTransform(255 - mask_lines, cv2.DIST_L2, 3), max_px).round().astype(np.uint8)


def model_samples(length: float, width: float, step_m: float) -> np.ndarray:
    """Pontos (X, Y) a cada `step_m` metros ao longo de todas as marcações do modelo do campo."""
    pts = []
    for pl in pitch.lines(length, width):
        for p, q in zip(pl[:-1], pl[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(q - p) / step_m)))
            pts.append(p + (q - p) * np.arange(n)[:, None] / n)
    return np.vstack(pts)


def refine(params: np.ndarray, C: np.ndarray, dt: np.ndarray, region: np.ndarray, samples: np.ndarray,
           cfg: dict) -> tuple[np.ndarray, float, int]:
    """Ajusta os parâmetros PTZ (`camera`) para que os pontos do modelo projetados caiam sobre as linhas
    (transformada de distância `dt`). Devolve (parâmetros, fração dos pontos visíveis a até `inlier_px`
    de uma linha, nº de pontos visíveis). Só contam pontos do modelo que caem em `region` (onde uma
    linha seria visível) na estimativa inicial. O termo de regularização mede, em pixels, quanto a
    imagem se deslocaria (f·Δângulo, (w/2)·Δlog f)."""
    h, w = dt.shape
    p = samples[_visible(params, C, region, samples)]
    if len(p) < cfg["min_visible"]:
        return params, np.nan, len(p)
    dtf = dt.astype(np.float32)
    scale = np.r_[np.full(3, np.exp(params[3])), w / 2]

    def dist(x: np.ndarray, sat: float) -> np.ndarray:
        q = camera.project(x, C, w, h, p)
        return np.minimum(map_coordinates(dtf, [q[:, 1], q[:, 0]], order=1, mode="constant", cval=sat), sat)

    # grosso → fino: com a distância saturada alto, erros de dezenas de px ainda puxam o modelo;
    # depois, saturação baixa para que só as linhas próximas contem
    x = np.asarray(params, float)
    for sat, fs in cfg["stages"]:
        x = least_squares(lambda x: np.r_[dist(x, sat), cfg["prior_weight"] * scale * (x - params)], x,
                          method="trf", loss="soft_l1", f_scale=fs, max_nfev=cfg["max_iter"], x_scale=1 / scale).x
    return x, float((dist(x, np.inf) <= cfg["inlier_px"]).mean()), len(p)


def _visible(params: np.ndarray, C: np.ndarray, region: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Pontos do modelo à frente da câmera que caem em `region`."""
    h, w = region.shape
    q = np.c_[samples, np.ones(len(samples))] @ np.linalg.inv(camera.to_homography(params, C, w, h)).T
    ok = q[:, 2] > 0
    xy = np.round(q[:, :2] / np.where(ok, q[:, 2], 1)[:, None]).astype(int)
    ok &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ok[ok] &= region[xy[ok, 1], xy[ok, 0]] > 0
    return ok


def pose_grid(candidates: list[np.ndarray], cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Grade de câmeras sem roll (rotações (N, 3, 3), log f (N,)) cobrindo a faixa de pan, tilt e zoom
    das câmeras marcadas, com margem. Pan e tilt saem do eixo óptico no referencial do campo (Z para
    dentro do gramado); a câmera da transmissão quase não gira em torno do eixo óptico."""
    d = np.array([cv2.Rodrigues(np.asarray(c[:3], float))[0][2] for c in candidates])
    pan, tilt, lf = np.arctan2(d[:, 1], d[:, 0]), np.arcsin(d[:, 2]), np.array([c[3] for c in candidates])
    m, st = np.radians(cfg["grid_margin_deg"]), np.radians(cfg["grid_step_deg"])
    P, T, F = (g.ravel() for g in np.meshgrid(np.arange(pan.min() - m, pan.max() + m, st),
                                              np.arange(tilt.min() - m, tilt.max() + m, st),
                                              np.arange(lf.min() - cfg["grid_logf_margin"], lf.max() + cfg["grid_logf_margin"],
                                                        cfg["grid_logf_step"]), indexing="ij"))
    z = np.c_[np.cos(T) * np.cos(P), np.cos(T) * np.sin(P), np.sin(T)]
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return np.stack([x, np.cross(z, x), z], axis=1), F


def relocalize(grid: tuple[np.ndarray, np.ndarray], C: np.ndarray, dt: np.ndarray, region: np.ndarray,
               samples: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, float]:
    """Câmera perdida: busca exaustiva na grade de pan/tilt/zoom (`pose_grid`). Cada pose é avaliada
    pela fração dos pontos do modelo (amostrados a cada `reloc_sample_m`) que caem a até `coarse_px`
    de uma linha na transformada de distância; as `reloc_top` melhores passam pelo `refine`.
    Devolve (parâmetros, fração de pontos sobre as linhas) da melhor, ou (None, nan)."""
    h, w = dt.shape
    Rs, lf = grid
    pts = samples[::max(1, round(cfg["reloc_sample_m"] / cfg["sample_m"]))]
    X = np.c_[pts, np.zeros(len(pts))] - C
    scores = np.zeros(len(lf))
    for s in range(0, len(lf), 2000):                        # em blocos, para caber na memória
        q = np.einsum("nij,pj->npi", Rs[s:s + 2000], X)
        f = np.exp(lf[s:s + 2000])[:, None]
        zc = np.where(q[..., 2] > 0, q[..., 2], np.nan)
        u, v = f * q[..., 0] / zc + w / 2, f * q[..., 1] / zc + h / 2
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui, vi = np.where(ok, u, 0).astype(int), np.where(ok, v, 0).astype(int)
        ok &= region[vi, ui] > 0
        n = ok.sum(1)
        scores[s:s + 2000] = np.where(n >= cfg["reloc_min_visible"],
                                      (ok & (dt[vi, ui] <= cfg["coarse_px"])).sum(1) / np.maximum(n, 1), 0)
    best, best_s = None, np.nan
    for j in np.argsort(-scores)[:cfg["reloc_top"]]:
        p0 = np.r_[cv2.Rodrigues(Rs[j])[0].ravel(), lf[j]]
        p, sc, _ = refine(p0, C, dt, region, samples, cfg)
        if np.isfinite(sc) and not sc <= best_s:
            best, best_s = p, sc
    return best, best_s
