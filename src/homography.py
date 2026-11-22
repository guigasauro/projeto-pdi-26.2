"""Homografia imagem → campo (operação geométrica projetiva).

`H` leva pontos da imagem (x, y) em pixels para o campo (X, Y) em metros: [X, Y, 1]ᵀ ~ H [x, y, 1]ᵀ.
Vale para pontos no plano do gramado (pés), não para pontos no ar (bola alta, cabeça).
"""
import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalização de Hartley: centróide na origem e distância média √2 (melhora o condicionamento do DLT)."""
    c = pts.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (pts - c) * s, T


def dlt_homography(pts_src: np.ndarray, pts_dst: np.ndarray) -> np.ndarray:
    """Versão didática: DLT normalizado. Cada correspondência dá 2 equações lineares em h (9 incógnitas);
    a solução é o vetor singular de menor valor singular de A (mínimos quadrados com ‖h‖ = 1)."""
    pts_src, pts_dst = np.asarray(pts_src, float), np.asarray(pts_dst, float)
    if len(pts_src) < 4:
        raise ValueError("homografia precisa de pelo menos 4 correspondências")
    a, Ta = _normalize(pts_src)
    b, Tb = _normalize(pts_dst)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta          # desfaz a normalização
    return H / H[2, 2]


def compute_homography(pts_img: np.ndarray, pts_pitch: np.ndarray, ransac_thresh_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Homografia imagem→campo com RANSAC do OpenCV (robusta a um clique errado).
    Retorna `(H, inliers)`; `ransac_thresh_m` é o erro máximo em metros para um ponto contar como inlier."""
    H, inl = cv2.findHomography(np.asarray(pts_img, np.float64), np.asarray(pts_pitch, np.float64), cv2.RANSAC, ransac_thresh_m)
    if H is None:
        raise ValueError("findHomography falhou (pontos colineares ou insuficientes?)")
    return H, inl.ravel().astype(bool)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica a homografia a pontos (N, 2) (divisão pela coordenada homogênea incluída)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if not len(pts):
        return np.empty((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def reprojection_error(H: np.ndarray, pts_img: np.ndarray, pts_pitch: np.ndarray) -> np.ndarray:
    """Erro em metros de cada correspondência: ‖H·p_img − p_campo‖."""
    return np.linalg.norm(project(H, pts_img) - np.asarray(pts_pitch, float), axis=1)


def save(path: str | Path, H: np.ndarray, pts_img: np.ndarray, names: list[str], frame: int) -> None:
    """Salva os cliques e a homografia (JSON legível em `data/annotations/`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frame": frame, "H_img_to_pitch": np.asarray(H).tolist(),
                                "points": [{"name": n, "img": [float(x), float(y)]} for n, (x, y) in zip(names, pts_img)]},
                               indent=2, ensure_ascii=False))


def load(path: str | Path) -> tuple[np.ndarray, dict]:
    """Lê `H` (imagem→campo) e o JSON completo."""
    data = json.loads(Path(path).read_text())
    return np.array(data["H_img_to_pitch"]), data


def debug_view(img_bgr: np.ndarray, H: np.ndarray, polylines_pitch: list[np.ndarray],
               pts_img: np.ndarray | None = None, pts_pitch: np.ndarray | None = None) -> np.ndarray:
    """Validação: projeta as linhas do modelo do campo de volta na imagem com H⁻¹ (magenta).
    Se houver cliques, mostra o ponto clicado (verde) e onde H⁻¹ coloca o ponto do modelo (vermelho)."""
    view = img_bgr.copy()
    Hinv = np.linalg.inv(H)
    h, w = view.shape[:2]
    # Pontos do campo atrás da câmera têm coordenada homogênea w com sinal oposto ao de um ponto visível
    # (o centro da imagem) e seriam projetados "espelhados"; são descartados.
    c_pitch = H @ np.array([w / 2, h / 2, 1.0])
    w_sign = np.sign((Hinv @ c_pitch)[2])
    for pl in polylines_pitch:
        ph = np.c_[pl, np.ones(len(pl))] @ Hinv.T
        p = ph[:, :2] / ph[:, 2:3]
        ok = (np.sign(ph[:, 2]) == w_sign) & (np.abs(p) < 4 * max(h, w)).all(1)
        for seg in np.split(np.arange(len(pl)), np.where(~ok)[0]):   # desenha só trechos contínuos válidos
            seg = seg[ok[seg]]
            if len(seg) >= 2:
                cv2.polylines(view, [np.round(p[seg]).astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
    if pts_img is not None and pts_pitch is not None:
        for (x, y), (u, v) in zip(np.asarray(pts_img), project(Hinv, pts_pitch)):
            cv2.circle(view, (int(x), int(y)), 7, (0, 255, 0), 2)
            cv2.circle(view, (int(u), int(v)), 3, (0, 0, 255), -1)
    return view


# ── v2: propagação entre quadros ─────────────────────────────────────────────

def plane_mask(mask_field: np.ndarray, dets: np.ndarray, cfg: dict) -> np.ndarray:
    """Onde procurar pontos que pertencem ao plano do campo: máscara do gramado erodida (a borda com
    as placas não está no plano) menos as caixas dos jogadores, dilatadas de `player_margin` px
    (jogadores se movem; só o campo é rígido)."""
    e = cfg["mask_erode"]
    m = cv2.erode(mask_field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    p = cfg["player_margin"]
    for x1, y1, x2, y2 in np.asarray(dets, float).reshape(-1, 6)[:, :4] if len(dets) else []:
        m[max(0, int(y1) - p):int(y2) + p, max(0, int(x1) - p):int(x2) + p] = 0
    return m


def inter_frame(kp_prev, des_prev, kp_cur, des_cur, cfg: dict) -> tuple[np.ndarray | None, int]:
    """Homografia G que leva pontos do quadro atual para o de referência: casamento de descritores
    (Hamming para ORB, L2 para SIFT), teste da razão de Lowe e RANSAC. Devolve (G, nº de inliers); G = None se houver poucos inliers."""
    if des_prev is None or des_cur is None or len(des_prev) < 4 or len(des_cur) < 4:
        return None, 0
    norm = cv2.NORM_HAMMING if des_cur.dtype == np.uint8 else cv2.NORM_L2   # ORB binário | SIFT float
    pairs = cv2.BFMatcher(norm).knnMatch(des_cur, des_prev, k=2)
    good = [m for m, *n in pairs if n and m.distance < cfg["ratio"] * n[0].distance]
    if len(good) < cfg["min_inliers"]:
        return None, len(good)
    src = np.float32([kp_cur[m.queryIdx].pt for m in good])
    dst = np.float32([kp_prev[m.trainIdx].pt for m in good])
    G, inl = cv2.findHomography(src, dst, cv2.RANSAC, cfg["ransac_px"])
    n = int(inl.sum()) if inl is not None else 0
    return (G, n) if G is not None and n >= cfg["min_inliers"] else (None, n)


def propagate(video: str, H0: np.ndarray, frame0: int, frame1: int, dets: list[np.ndarray] | None,
              cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """v2: a partir de H0 (imagem→campo no quadro `frame0`), estima H_t até `frame1` (exclusivo).
    Cada quadro é casado com o último quadro-chave k (não com o anterior): H_t = H_k · G_{t→k}.
    Um novo quadro-chave só é criado quando os inliers caem abaixo de `keyframe_inliers` (pouca
    sobreposição), o que reduz o número de composições e, com ele, a deriva acumulada.
    `dets[t]` são as detecções do quadro t (caixas excluídas da busca de pontos).
    Devolve (H por quadro (N, 3, 3), inliers por quadro); se a estimativa falhar, H fica NaN
    daí em diante (precisa de reancoragem). `cfg` é o YAML inteiro."""
    from src.field_mask import field_mask
    hc = cfg["homography"]
    # SIFT do OpenCV escolhe os N melhores pontos da imagem inteira antes de aplicar a máscara (a torcida
    # ficava com todos): detecta sem limite e mantém os N de maior resposta dentro da máscara
    feat = (cv2.SIFT_create(0, contrastThreshold=hc["sift_contrast"]) if hc["features"] == "sift"
            else cv2.ORB_create(hc["n_features"]))
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame0)
    Hs = np.full((frame1 - frame0, 3, 3), np.nan)
    n_inl = np.zeros(frame1 - frame0, int)
    key = None                                   # (keypoints, descritores, H do quadro-chave)
    for i, t in enumerate(range(frame0, frame1)):
        ok, img = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = plane_mask(field_mask(img, cfg["field_mask"]), dets[t] if dets is not None else [], hc)
        kp = sorted(feat.detect(gray, m), key=lambda k: -k.response)[:hc["n_features"]]
        kp, des = feat.compute(gray, kp)
        if key is None:
            H = np.asarray(H0, float)
        else:
            G, n_inl[i] = inter_frame(key[0], key[1], kp, des, hc)
            if G is None:
                log.warning("propagação perdida no quadro %d (%d inliers)", t, n_inl[i])
                break
            H = key[2] @ G
            H /= H[2, 2]
        Hs[i] = H
        if key is None or n_inl[i] < hc["keyframe_inliers"]:
            key = (kp, des, H)
    return Hs, n_inl


def anchor(M: np.ndarray, k: int, Hk: np.ndarray) -> np.ndarray:
    """Ancora uma tomada propagada com H0 = I (M_t: imagem t → imagem do 1º quadro) numa homografia
    marcada no quadro k (índice dentro da tomada): H_t = H_k · M_k⁻¹ · M_t. Assim a marcação pode ser
    feita em qualquer quadro da tomada com uma única passada da propagação para a frente."""
    H = np.asarray(Hk, float) @ np.linalg.inv(M[k]) @ M
    return H / H[:, 2:3, 2:3]
