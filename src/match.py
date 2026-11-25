"""Pipeline de uma partida completa: só as tomadas da câmera principal com homografia marcada.

Cada tomada é processada isoladamente (detecção, tracking, descritores de cor e propagação SIFT com
H0 = I) e guardada em `outputs/<partida>/<tempo>/tomadas/<frame_a>/`. Depois, a câmera de cada quadro
é ajustada às linhas do campo (modelo PTZ com o centro da câmera estimado de todas as marcações,
`camera` + `lines`), partindo do quadro marcado; remarcar uma tomada refaz só esse ajuste.
Times: um único k-means para a partida inteira (os uniformes não mudam entre os tempos).
2º tempo: as equipes trocam de lado no intervalo. Em vez de supor isso (a transmissão poderia mostrar
um tempo por outra câmera), o lado de cada time sai dos dados (`last_defender_votes`): se o time que
defende a esquerda mudou, as posições do 2º tempo giram 180° (X → L − X, Y → W − Y) e cada time fica
sempre atacando no sentido do 1º tempo.
"""
import hashlib
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src import camera, detection, heatmap, homography, pitch, team_classifier, tracking
from src.pipeline import _load_or, positions, team_heatmaps

log = logging.getLogger(__name__)


def match_cfg(cfg: dict, M_info: dict) -> dict:
    """Configuração com o `frame_step` e a medida do campo da partida (os módulos leem `video.frame_step`
    e `pitch`)."""
    c = pitch.for_match(cfg, M_info)
    c["video"]["frame_step"] = cfg["match"]["frame_step"]
    return c


def process_shot(video: str, a: int, b: int, cfg: dict, cache: Path) -> dict:
    """Detecções, tracks, descritores e M_t (imagem t → imagem do quadro a) da tomada [a, b)."""
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
    dets = detection.detect_video(video, cfg, cache / "detections.npz", start=a, end=b)
    tracks = tracking.track_cached(dets, cfg, fps, cache / "tracks.npz")
    descs = _load_or(cache / "descs.npz", lambda: team_classifier.extract(video, tracks, cfg, start=a))
    if (cache / "M.npy").exists():
        M = np.load(cache / "M.npy")
    else:
        M, _ = homography.propagate(video, np.eye(3), a, b, {a + i: d for i, d in enumerate(dets)}, cfg)
        np.save(cache / "M.npy", M)
    return {"dets": dets, "tracks": tracks, "descs": descs, "M": M, "fps": fps}


def rotate_half(pos: np.ndarray, length: float, width: float) -> np.ndarray:
    """Giro de 180° do campo nas colunas X, Y de `positions` (troca de lado no intervalo)."""
    out = pos.copy()
    out[:, 3], out[:, 4] = length - pos[:, 3], width - pos[:, 4]
    return out


def last_defender_votes(pos: np.ndarray, cfg_match: dict, length: float) -> np.ndarray:
    """Quem defende cada gol: em cada quadro com o jogo perto de um gol, o jogador de linha mais recuado
    (fora da pequena área, onde fica o goleiro) costuma ser da defesa (regra do impedimento).
    `pos` = `positions` de uma tomada. Devolve votos (2 gols: esq, dir) × (2 times)."""
    votes = np.zeros((2, 2))
    gk, deep = cfg_match["gk_zone_m"], cfg_match["deep_zone_m"]
    p_all = pos[pos[:, 2] >= 0]
    for f in np.unique(p_all[:, 0]):
        p = p_all[p_all[:, 0] == f]
        if len(p) < cfg_match["min_players"]:
            continue
        for side, X in enumerate((p[:, 3], length - p[:, 3])):          # distância ao gol esq / dir
            out = X > gk
            if out.sum() and X[out].min() < deep:
                votes[side, int(p[out][X[out].argmin(), 2])] += 1
    return votes


def run(match_json: str | Path, cfg: dict, out_dir: str | Path | None = None) -> dict:
    """Processa as tomadas marcadas (as já processadas vêm do cache) e soma os heatmaps por time."""
    M_info = json.loads(Path(match_json).read_text())
    cfg = match_cfg(cfg, M_info)
    name = Path(match_json).stem
    out_dir = Path(out_dir or f"outputs/{name}")
    ann = json.loads(Path(f"data/annotations/{name}_homografias.json").read_text())
    step, tc, pc, lc = cfg["video"]["frame_step"], cfg["team"], cfg["pitch"], cfg["lines"]
    marked = sorted((s for s in ann.values() if "H_img_to_pitch" in s), key=lambda s: (s["tempo"], s["frame_a"]))
    w, h = (int(cv2.VideoCapture(M_info["tempos"]["1"]["video"]).get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    kp = pitch.keypoints(pc["length_m"], pc["width_m"])
    C, good = camera.from_marks(marked, kp, w, h, lc["center_max_resid"])
    log.info("centro da câmera (m): %s, de %d marcações", np.round(C, 1), len(good))
    shots = []
    for s in marked:
        a, b, k = s["frame_a"], s["frame_b"], s["frame"] - s["frame_a"]
        if k % step:
            raise ValueError(f"tomada {s['tempo']}:{a}: quadro marcado {s['frame']} fora da amostragem (step {step})")
        pts_img = np.array([q["img"] for q in s["points"]])
        p_k, err = camera.anchor(pts_img, np.array([kp[q["name"]] for q in s["points"]]),
                                 None, C, w, h, good)
        if err > lc["anchor_max_px"]:
            log.warning("tomada %s:%d descartada: cliques não batem com a câmera (erro mediano %.0f px); remarque",
                        s["tempo"], a, err)
            continue
        video = M_info["tempos"][s["tempo"]]["video"]
        cache = out_dir / s["tempo"] / "tomadas" / str(a)
        log.info("tomada %s:%d (%d quadros)", s["tempo"], a, b - a)
        r = process_shot(video, a, b, cfg, cache)
        tag = hashlib.md5(json.dumps([p_k.round(5).tolist(), k, C.round(2).tolist(), lc, pc]).encode()).hexdigest()[:8]
        f = cache / f"camera_{tag}.npz"
        if f.exists():
            Hs, score = (np.load(f)[x] for x in ("Hs", "score"))
        else:
            Hs, score = homography.refine_lines(video, a, b, k, p_k, C, r["M"], r["dets"], cfg)
            np.savez(f, Hs=Hs, score=score)
        Hs[~(score >= lc["min_score"])] = np.nan          # quadro sem linhas suficientes ou desalinhado
        r["Hs"], r["score"] = Hs, score
        shots.append(s | r)
    if not shots:
        raise RuntimeError("nenhuma tomada marcada e válida")

    # um k-means para a partida: primeiros `calib_frames` quadros processados de cada tomada
    sample = [d for r in shots for d in r["descs"][::step][:tc["calib_frames"]]]
    km = team_classifier.calibrate(sample, tc | {"calib_frames": len(sample)})
    pos_all, n_frames, crops, votes = [], 0, {}, {}
    for r in shots:
        labels = team_classifier.vote(r["tracks"], team_classifier.classify(km, r["descs"]),
                                      round(tc["vote_window_s"] * r["fps"]))
        r["pos"] = positions(r["tracks"], labels, r["Hs"])
        votes[r["tempo"]] = votes.get(r["tempo"], 0) + last_defender_votes(r["pos"], cfg["match"], pc["length_m"])
        n_frames += int(np.isfinite(r["Hs"]).all((1, 2)).sum())
        if len(crops) < 2 or min(len(c) for c in crops.values()) < 12:
            _sample_crops(M_info["tempos"][r["tempo"]]["video"], r, labels, tc, crops)
    left = {}
    for half, v in sorted(votes.items()):
        left[half] = int(np.argmax(v[0] + v[1, ::-1]))     # quem defende a esq (votos do gol dir contam ao contrário)
        log.info("tempo %s: último defensor no gol esq %s, no gol dir %s (votos por time) → time %d defende a esquerda",
                 half, v[0].astype(int), v[1].astype(int), left[half])
    ref = left[min(left)]
    for r in shots:
        flip = left[r["tempo"]] != ref
        pos_all.append(rotate_half(r["pos"], pc["length_m"], pc["width_m"]) if flip else r["pos"])
    pos = np.vstack(pos_all)
    maps = team_heatmaps(pos, n_frames, cfg)
    total = sum(len(range(0, r["frame_b"] - r["frame_a"], step)) for r in shots)
    log.info("%d tomadas, %.1f min com homografia (%.0f%% dos quadros das tomadas); %d posições",
             len(shots), n_frames * step / shots[0]["fps"] / 60, 100 * n_frames / total, len(pos))
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = max(m.max() for m in maps.values())
    for team, m in maps.items():
        cv2.imwrite(str(out_dir / f"heatmap_time{team}.png"), heatmap.colorize(m, cfg, vmax=vmax))
    if len(maps) == 2:
        cv2.imwrite(str(out_dir / "heatmap_diferenca.png"),
                    heatmap.colorize(maps[0] - maps[1], cfg, cmap=cfg["heatmap"]["diff_colormap"], signed=True))
    cv2.imwrite(str(out_dir / "times_recortes.png"), team_classifier.debug_view(crops))
    return {"shots": shots, "positions": pos, "maps": maps, "n_frames": n_frames, "kmeans": km, "left": left}


def _sample_crops(video: str, r: dict, labels: list[np.ndarray], tc: dict, crops: dict) -> None:
    """Recortes de tronco do quadro marcado, por cluster (para saber qual cluster é qual time)."""
    k = r["frame"] - r["frame_a"]
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
    ok, img = cap.read()
    if ok:
        for box, lab in zip(r["tracks"][k], labels[k]):
            if lab >= 0:
                crops.setdefault(int(lab), []).append(team_classifier.torso_crop(img, box, tc["torso"]))
