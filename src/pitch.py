"""Modelo geométrico do campo (regras FIFA) e desenho top-view.

Coordenadas em metros: origem no canto superior esquerdo, X ao longo do comprimento (0..L),
Y ao longo da largura (0..W), Y crescendo para o lado da câmera (lado de baixo no top-view).
SoccerNet usa origem no centro: X = x + L/2, Y = y + W/2.
"""
import cv2
import numpy as np

# Medidas fixadas pela regra (Lei 1 da IFAB), não são parâmetros de ajuste.
PENALTY_DEPTH, PENALTY_WIDTH = 16.5, 40.32
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5, 18.32
PENALTY_SPOT, CIRCLE_R, GOAL_WIDTH = 11.0, 9.15, 7.32


def keypoints(length: float, width: float) -> dict[str, tuple[float, float]]:
    """Pontos notáveis (X, Y) em metros, usados como correspondências na homografia.
    Lado `esq` = X pequeno, `dir` = X grande; `sup` = Y pequeno (lado oposto à câmera), `inf` = Y grande."""
    cy = width / 2
    kp = {"meio_sup": (length / 2, 0.0), "meio_inf": (length / 2, width), "centro": (length / 2, cy),
          "circulo_sup": (length / 2, cy - CIRCLE_R), "circulo_inf": (length / 2, cy + CIRCLE_R)}
    arc_dy = np.sqrt(CIRCLE_R ** 2 - (PENALTY_DEPTH - PENALTY_SPOT) ** 2)  # meia-lua cruza a linha da grande área
    for side, x0, s in (("esq", 0.0, 1), ("dir", length, -1)):
        kp |= {
            f"canto_sup_{side}": (x0, 0.0), f"canto_inf_{side}": (x0, width),
            f"ga_fundo_sup_{side}": (x0, cy - PENALTY_WIDTH / 2), f"ga_fundo_inf_{side}": (x0, cy + PENALTY_WIDTH / 2),
            f"ga_frente_sup_{side}": (x0 + s * PENALTY_DEPTH, cy - PENALTY_WIDTH / 2),
            f"ga_frente_inf_{side}": (x0 + s * PENALTY_DEPTH, cy + PENALTY_WIDTH / 2),
            f"pa_fundo_sup_{side}": (x0, cy - GOAL_AREA_WIDTH / 2), f"pa_fundo_inf_{side}": (x0, cy + GOAL_AREA_WIDTH / 2),
            f"pa_frente_sup_{side}": (x0 + s * GOAL_AREA_DEPTH, cy - GOAL_AREA_WIDTH / 2),
            f"pa_frente_inf_{side}": (x0 + s * GOAL_AREA_DEPTH, cy + GOAL_AREA_WIDTH / 2),
            f"penalti_{side}": (x0 + s * PENALTY_SPOT, cy),
            f"meialua_sup_{side}": (x0 + s * PENALTY_DEPTH, cy - arc_dy),
            f"meialua_inf_{side}": (x0 + s * PENALTY_DEPTH, cy + arc_dy),
            f"trave_sup_{side}": (x0, cy - GOAL_WIDTH / 2), f"trave_inf_{side}": (x0, cy + GOAL_WIDTH / 2),
        }
    return kp


def nearest_keypoint(kp: dict[str, tuple[float, float]], XY: tuple[float, float], max_dist_m: float) -> str | None:
    """Nome do ponto notável mais próximo de (X, Y), ou None se nenhum estiver a menos de `max_dist_m`."""
    name = min(kp, key=lambda k: np.hypot(kp[k][0] - XY[0], kp[k][1] - XY[1]))
    return name if np.hypot(kp[name][0] - XY[0], kp[name][1] - XY[1]) <= max_dist_m else None


def _arc(center: tuple[float, float], r: float, a0: float, a1: float, n: int = 64) -> np.ndarray:
    t = np.linspace(a0, a1, n)
    return np.c_[center[0] + r * np.cos(t), center[1] + r * np.sin(t)]


def lines(length: float, width: float) -> list[np.ndarray]:
    """Marcações do campo como polilinhas (N, 2) em metros. Curvas amostradas em segmentos para
    poderem ser projetadas ponto a ponto pela homografia (overlay de validação)."""
    kp = keypoints(length, width)
    rect = lambda a, b, c, d: np.array([kp[a], kp[b], kp[c], kp[d]] + [kp[a]])
    out = [rect("canto_sup_esq", "canto_sup_dir", "canto_inf_dir", "canto_inf_esq"),
           np.array([kp["meio_sup"], kp["meio_inf"]]),
           _arc(kp["centro"], CIRCLE_R, 0, 2 * np.pi)]
    half = np.arcsin(np.sqrt(CIRCLE_R ** 2 - (PENALTY_DEPTH - PENALTY_SPOT) ** 2) / CIRCLE_R)
    for side, ang in (("esq", 0.0), ("dir", np.pi)):
        for p in ("ga", "pa"):
            out.append(np.array([kp[f"{p}_fundo_sup_{side}"], kp[f"{p}_frente_sup_{side}"],
                                 kp[f"{p}_frente_inf_{side}"], kp[f"{p}_fundo_inf_{side}"]]))
        out.append(_arc(kp[f"penalti_{side}"], CIRCLE_R, ang - half, ang + half, 32))
    return out


def to_px(pts_pitch: np.ndarray, px_per_m: float, margin_m: float) -> np.ndarray:
    """Escala (operação geométrica) metros → pixels do top-view, com margem em volta do campo."""
    return (np.asarray(pts_pitch, float) + margin_m) * px_per_m


def draw_pitch(length: float, width: float, px_per_m: float, margin_m: float) -> np.ndarray:
    """Campo top-view (BGR): gramado verde, linhas brancas e marcas de pênalti/centro."""
    w = int(round((length + 2 * margin_m) * px_per_m)); h = int(round((width + 2 * margin_m) * px_per_m))
    img_bgr = np.full((h, w, 3), (40, 110, 40), np.uint8)
    thick = max(1, round(px_per_m * 0.12))  # linha de 12 cm (regra)
    for pl in lines(length, width):
        cv2.polylines(img_bgr, [np.round(to_px(pl, px_per_m, margin_m)).astype(np.int32)], False, (255, 255, 255), thick, cv2.LINE_AA)
    kp = keypoints(length, width)
    for k in ("centro", "penalti_esq", "penalti_dir"):
        x, y = np.round(to_px(kp[k], px_per_m, margin_m)).astype(int)
        cv2.circle(img_bgr, (x, y), max(2, thick), (255, 255, 255), -1, cv2.LINE_AA)
    return img_bgr


def from_soccernet(xy: np.ndarray, length: float, width: float) -> np.ndarray:
    """Coordenadas do SoccerNet (origem no centro) → coordenadas do projeto (origem no canto)."""
    return np.asarray(xy, float) + (length / 2, width / 2)


def debug_view(cfg: dict) -> np.ndarray:
    """Top-view com os pontos notáveis rotulados (referência para a anotação manual)."""
    L, W, ppm, mg = cfg["length_m"], cfg["width_m"], cfg["px_per_m"], cfg["margin_m"]
    img_bgr = draw_pitch(L, W, ppm, mg)
    for name, p in keypoints(L, W).items():
        x, y = np.round(to_px(p, ppm, mg)).astype(int)
        cv2.circle(img_bgr, (x, y), 4, (0, 0, 255), -1)
        tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0][0]
        tx = x + 4 if p[0] <= L / 2 else x - 4 - tw          # rótulos do lado direito à esquerda do ponto
        ty = y - 4 if "sup" in name or "fundo" in name else y + 12
        cv2.putText(img_bgr, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)
    return img_bgr
