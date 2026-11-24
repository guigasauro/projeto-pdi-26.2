"""Modelo de câmera PTZ da transmissão: posição fixa no estádio, só gira (pan/tilt/roll) e muda o zoom.

Para um ponto do gramado (X, Y, 0): x ~ K R (X − C), ou seja, P = K R [e1 e2 −C] (campo → imagem),
com K = [[f, 0, w/2], [0, f, h/2], [0, 0, 1]] (pixels quadrados, centro óptico no centro da imagem).
Uma homografia geral tem 8 graus de liberdade; com C fixo (estimado das marcações manuais) cada quadro
tem só 4 (rotação de Rodrigues + f), o que impede o ajuste às linhas de "entortar" o campo quando
poucas linhas aparecem.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def intrinsics(f: float, w: int, h: int) -> np.ndarray:
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def to_homography(params: np.ndarray, C: np.ndarray, w: int, h: int) -> np.ndarray:
    """params = (rvec (3), log f) → H imagem→campo."""
    R = cv2.Rodrigues(np.asarray(params[:3], float))[0]
    P = intrinsics(np.exp(params[3]), w, h) @ R @ np.c_[np.eye(3)[:, :2], -np.asarray(C, float)]
    H = np.linalg.inv(P)
    return H / H[2, 2]


def decompose(H: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Homografia imagem→campo → (params, centro da câmera C, resíduo da ortogonalidade).
    f sai das restrições r1 ⟂ r2 e |r1| = |r2| (colunas de K⁻¹P) com o centro óptico conhecido."""
    P = np.linalg.inv(H)

    def cols(logf: float) -> np.ndarray:
        return np.linalg.inv(intrinsics(np.exp(logf), w, h)) @ P

    def cost(logf: float) -> float:
        B = cols(logf); r1, r2 = B[:, 0], B[:, 1]
        return (r1 @ r2) ** 2 / (r1 @ r1 * (r2 @ r2)) + ((r1 @ r1 - r2 @ r2) / (r1 @ r1 + r2 @ r2)) ** 2

    res = minimize_scalar(cost, bounds=(np.log(100), np.log(50000)), method="bounded")
    B = cols(res.x)
    B /= np.sqrt(np.linalg.norm(B[:, 0]) * np.linalg.norm(B[:, 1]))
    if B[2, 2] < 0:                       # escolhe o sinal com o campo à frente da câmera
        B = -B
    r1, r2 = B[:, 0] / np.linalg.norm(B[:, 0]), B[:, 1] / np.linalg.norm(B[:, 1])
    U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    R = U @ Vt                            # rotação mais próxima (as colunas medidas não são exatamente ortonormais)
    C = -R.T @ B[:, 2]
    return np.r_[cv2.Rodrigues(R)[0].ravel(), res.x], C, float(res.fun)


def estimate_center(Hs: list[np.ndarray], w: int, h: int, max_resid: float) -> np.ndarray:
    """Centro da câmera: mediana dos centros das marcações bem condicionadas (resíduo ≤ max_resid)."""
    Cs = [C for p, C, r in (decompose(H, w, h) for H in Hs) if r <= max_resid]
    if not Cs:
        raise ValueError("nenhuma homografia marcada serve para estimar o centro da câmera")
    return np.median(Cs, axis=0)


def refine_center(C0: np.ndarray, marks: list[tuple[np.ndarray, np.ndarray]], inits: list[np.ndarray],
                  w: int, h: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Ajuste conjunto (bundle adjustment): um C para todas as marcações e (rotação, f) de cada uma,
    minimizando o erro de reprojeção dos cliques em pixels. A mediana dos C de cada homografia isolada
    é ruidosa (C se confunde com f ao longo da direção de visada); com todos os cliques juntos, não.
    `marks` = [(pts_img, pts_pitch)], `inits` = parâmetros iniciais de cada marcação."""
    n = len(marks)

    def resid(x: np.ndarray) -> np.ndarray:
        P = x[3:].reshape(n, 4)
        return np.concatenate([(project(P[i], x[:3], w, h, pp) - pi).ravel() for i, (pi, pp) in enumerate(marks)])

    x = least_squares(resid, np.r_[C0, np.ravel(inits)], loss="soft_l1", f_scale=3.0).x
    return x[:3], list(x[3:].reshape(n, 4))


def from_marks(marks: list[dict], kp: dict, w: int, h: int, max_resid: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """Centro da câmera e parâmetros de cada marcação (JSON `<partida>_homografias.json`, só as salvas).
    C inicial: mediana das homografias gerais bem condicionadas (`estimate_center`); depois, ajuste
    conjunto com os cliques de todas as marcações, inclusive as feitas já com a câmera fixa."""
    general = [np.array(s["H_img_to_pitch"]) for s in marks if s.get("modelo", "homografia") == "homografia"]
    C0 = estimate_center(general, w, h, max_resid)
    good = [fit(H, C0, w, h) for H in general if decompose(H, w, h)[2] <= max_resid]
    data, inits = [], []
    for s in marks:
        pi, pp = np.array([q["img"] for q in s["points"]]), np.array([kp[q["name"]] for q in s["points"]])
        H = np.array(s["H_img_to_pitch"]) if s.get("modelo", "homografia") == "homografia" else None
        p, _ = anchor(pi, pp, H, C0, w, h, good)
        if p is not None:
            data.append((pi, pp)); inits.append(p)
    return refine_center(C0, data, inits, w, h)


def image_grid(w: int, h: int, n: int = 8) -> np.ndarray:
    """Grade de pontos da imagem (metade de baixo e meio: onde fica o gramado)."""
    xs, ys = np.meshgrid(np.linspace(0.05 * w, 0.95 * w, n), np.linspace(0.3 * h, 0.95 * h, n))
    return np.c_[xs.ravel(), ys.ravel()]


def fit(H: np.ndarray, C: np.ndarray, w: int, h: int, init: np.ndarray | None = None) -> np.ndarray:
    """Parâmetros PTZ (com C fixo) mais próximos de uma homografia qualquer: pontos da imagem vão ao
    campo por H e voltam pela câmera; minimiza a diferença em pixels."""
    g = image_grid(w, h)
    ph = np.c_[g, np.ones(len(g))] @ H.T
    ok = ph[:, 2] > 0
    pts = ph[ok, :2] / ph[ok, 2:3]
    g = g[ok]

    def resid(p: np.ndarray) -> np.ndarray:
        q = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(to_homography(p, C, w, h)).T
        return (q[:, :2] / q[:, 2:3] - g).ravel()

    p0 = decompose(H, w, h)[0] if init is None else init
    if len(g) < 4:                       # H degenerada (horizonte na imagem): fica com o chute inicial
        return p0
    return least_squares(resid, p0, method="lm").x


def project(params: np.ndarray, C: np.ndarray, w: int, h: int, pts_pitch: np.ndarray) -> np.ndarray:
    """Pontos do campo (N, 2) → imagem (N, 2) pela câmera."""
    q = np.c_[np.asarray(pts_pitch, float), np.ones(len(pts_pitch))] @ np.linalg.inv(to_homography(params, C, w, h)).T
    return q[:, :2] / q[:, 2:3]


def fit_points(pts_img: np.ndarray, pts_pitch: np.ndarray, C: np.ndarray, w: int, h: int,
               init: np.ndarray) -> np.ndarray:
    """Parâmetros PTZ a partir de cliques: minimiza o erro de reprojeção em pixels (custo robusto).
    Com só 4 incógnitas, bastam 2 pontos e pontos colineares (linha do meio) não degeneram."""
    return least_squares(lambda p: (project(p, C, w, h, pts_pitch) - pts_img).ravel(), init,
                         loss="soft_l1", f_scale=3.0).x


def anchor(pts_img: np.ndarray, pts_pitch: np.ndarray, H: np.ndarray | None, C: np.ndarray, w: int, h: int,
           candidates: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Câmera de um quadro marcado: `fit_points` partindo da homografia marcada e de cada câmera em
    `candidates` (outras marcações; resolve marcações quase colineares, em que a H de 8 parâmetros
    degenera). Devolve (parâmetros, erro mediano de reprojeção em px)."""
    best, err = None, np.inf
    for init in ([fit(H, C, w, h)] if H is not None else []) + list(candidates):
        p = fit_points(pts_img, pts_pitch, C, w, h, init)
        e = float(np.median(np.linalg.norm(project(p, C, w, h, pts_pitch) - pts_img, axis=1)))
        if e < err and np.exp(p[3]) > 0.1 * w:
            best, err = p, e
    return best, err
