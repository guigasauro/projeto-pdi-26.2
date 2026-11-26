"""Métricas contra o gabarito do SoccerNet.

  python scripts/evaluate.py homography data/annotations/clip01.json --gt data/raw/SoccerNetGS/valid/SNGS-056/Labels-GameState.json

`homography`: projeta os pés dos jogadores do GT (imagem) com a H anotada e compara com a posição
real no campo (GT, metros), frame a frame. Mostra quanto uma H fixa se degrada quando a câmera mexe.
`--gt-reference` calcula a H a partir dos próprios pés do GT no frame anotado (referência ideal).

  python scripts/evaluate.py team data/clips/clip01.mp4 --gt .../Labels-GameState.json

`team`: casa os tracks em cache com o GT (IoU >= 0,5) e mede a acurácia do time (left/right) dos jogadores
de linha, por descritor e com/sem remoção de grama e votação temporal (ablações 1 e 2).

  python scripts/evaluate.py shots data/annotations/liverpool-x-manchester_city.json

`shots`: compara o filtro da câmera principal (outputs/<partida>/<tempo>/shots.json, de segment_match.py)
com o Labels-cameras.json do SoccerNet (chave "soccernet_cameras" do JSON da partida), quadro a quadro:
gabarito = "Main camera center/left/right" ao vivo (sem replay).

  python scripts/evaluate.py camera data/annotations/liverpool-x-manchester_city.json

`camera`: relocalização automática nos quadros marcados (erro contra os cliques) e quadros com câmera
aceita (com e sem marcação) contra o Labels-cameras.json.

  python scripts/evaluate.py goleiro [--gt-dir data/raw/SoccerNetGS/valid]

`goleiro`: regra do goleiro pela posição aplicada às posições e IDs do gabarito de cada sequência do GSR:
o ID escolhido é o goleiro? o time atribuído é o certo?
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import homography, pitch, team_classifier  # noqa: E402


def gt_feet(gt_path: str, length: float, width: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{frame: (pés na imagem (N,2) px, pés no campo (N,2) m)} para jogadores, goleiros e árbitros."""
    out: dict[int, list] = {}
    for a in json.load(open(gt_path))["annotations"]:
        if a["category_id"] in (1, 2, 3) and a.get("bbox_pitch"):
            b, p = a["bbox_image"], a["bbox_pitch"]
            f = out.setdefault(int(a["image_id"][-6:]) - 1, [[], []])
            f[0].append((b["x_center"], b["y"] + b["h"])); f[1].append((p["x_bottom_middle"], p["y_bottom_middle"]))
    return {i: (np.array(v[0], float), pitch.from_soccernet(np.array(v[1]), length, width)) for i, v in out.items()}


def eval_homography(args: argparse.Namespace, cfg: dict) -> None:
    pc = cfg["pitch"]
    feet = gt_feet(args.gt, pc["length_m"], pc["width_m"])
    H, data = homography.load(args.annotation)
    f0 = data["frame"]
    label = "manual"
    if args.gt_reference:
        H, _ = homography.compute_homography(*feet[f0], cfg["homography"]["ransac_thresh_m"])
        label = "referência GT"
    frames = sorted(feet)
    curves = {f"H fixa ({label})": [H] * len(frames)}
    if args.propagate:
        stem = Path(args.annotation).stem
        dets = np.load(f"outputs/{stem}/cache/detections.npz")
        dets = [dets[k] for k in sorted(dets.files, key=int)]
        video = args.video or f"data/clips/{stem}.mp4"
        Hs, n_inl = homography.propagate(video, H, f0, frames[-1] + 1, dets, cfg)
        print(f"propagação: inliers por quadro mediana {np.median(n_inl[1:]):.0f}, mínimo {n_inl[1:].min()}")
        curves[f"v2 propagada ({label})"] = [Hs[i - f0] for i in frames]
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, Hl in curves.items():
        med = np.array([np.median(homography.reprojection_error(h, *feet[i])) if np.isfinite(h).all() else np.nan
                        for h, i in zip(Hl, frames)])
        print(f"{name}: erro no frame {f0} = {med[frames.index(f0)]:.2f} m (mediana) | ao longo do clipe: "
              f"mediana {np.nanmedian(med):.2f} m, pior {np.nanmax(med):.2f} m (frame {frames[int(np.nanargmax(med))]})")
        ax.plot(frames, med, label=name)
    ax.set_xlabel("frame"); ax.set_ylabel("erro mediano (m)"); ax.set_title("Erro dos pés do GT projetados no campo")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    out = Path("outputs/figures") / (f"05_erro_homografia_{Path(args.annotation).stem}"
                                     f"{'_gtref' if args.gt_reference else ''}{'_v2' if args.propagate else ''}.png")
    fig.savefig(out, dpi=120); print("figura:", out)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre todas as caixas (N, 4) x (M, 4) no formato x1, y1, x2, y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter)


def gt_roles(gt_path: str, length: float = 105, width: float = 68) -> dict[int, tuple[np.ndarray, list[str], np.ndarray]]:
    """{frame: (caixas GT (N, 4), rótulo, pés no campo (N, 2) m)} com rótulo = left | right (jogador),
    gk_<time> ou referee."""
    out: dict[int, tuple[list, list, list]] = {}
    for a in json.load(open(gt_path))["annotations"]:
        if a["category_id"] in (1, 2, 3):
            b, at = a["bbox_image"], a["attributes"]
            lab = at["team"] if a["category_id"] == 1 else f"gk_{at['team']}" if a["category_id"] == 2 else "referee"
            p = a.get("bbox_pitch") or {"x_bottom_middle": np.nan, "y_bottom_middle": np.nan}
            f = out.setdefault(int(a["image_id"][-6:]) - 1, ([], [], []))
            f[0].append((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])); f[1].append(lab)
            f[2].append((p["x_bottom_middle"], p["y_bottom_middle"]))
    return {i: (np.array(v[0], float), v[1], pitch.from_soccernet(np.array(v[2], float), length, width))
            for i, v in out.items()}


def match_gt(tracks: list[np.ndarray], gt: dict) -> list[tuple[int, int, str]]:
    """Pares (frame, índice no track, rótulo GT) casados por IoU >= 0,5 (atribuição húngara)."""
    pairs = []
    for i, t in enumerate(tracks):
        if i not in gt or not len(t):
            continue
        m = iou(t[:, :4], gt[i][0]); r, c = linear_sum_assignment(-m)
        pairs += [(i, int(ti), gt[i][1][gi]) for ti, gi in zip(r, c) if m[ti, gi] >= 0.5]
    return pairs


def eval_team(args: argparse.Namespace, cfg: dict) -> None:
    from collections import Counter
    tracks = np.load(args.tracks); tracks = [tracks[k] for k in sorted(tracks.files, key=int)]
    pairs = match_gt(tracks, gt_roles(args.gt))
    print(f"{len(pairs)} detecções casadas com o GT: {dict(Counter(p[2] for p in pairs))}")
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    win = round(cfg["team"]["vote_window_s"] * fps) if cfg["team"]["vote_window_s"] is not None else None

    def acc(labels: list[np.ndarray]) -> float:
        pl = [(labels[i][j], g) for i, j, g in pairs if g in ("left", "right")]
        # o número do cluster é arbitrário: usa o mapeamento cluster→time que mais acerta
        return max(np.mean([l == (g == side) for l, g in pl]) for side in ("left", "right"))

    print(f"{'descritor':9s} {'grama':9s} {'por frame':>9s} {'janela':>7s} {'track':>7s}")
    for desc in ("rgb_mean", "lab_mean", "ab_mean", "hs_hist"):
        for grass in (False, True):
            c = {**cfg, "team": {**cfg["team"], "descriptor": desc, "remove_grass": grass}}
            descs = team_classifier.extract(args.video, tracks, c)
            labels = team_classifier.classify(team_classifier.calibrate(descs, c["team"]), descs)
            print(f"{desc:9s} {'removida' if grass else 'mantida':9s} {acc(labels):9.3f} "
                  f"{acc(team_classifier.vote(tracks, labels, win)):7.3f} {acc(team_classifier.vote(tracks, labels)):7.3f}")
            if (desc, grass) == (cfg["team"]["descriptor"], cfg["team"]["remove_grass"]):
                best = labels
    team_fig(args.video, tracks, team_classifier.vote(tracks, best, win), cfg)


def team_fig(video_path: str, tracks: list[np.ndarray], labels: list[np.ndarray], cfg: dict) -> None:
    """Figuras: recortes do tronco por cluster e dispersão a*b* (Lab) colorida pelo cluster."""
    tc = cfg["team"]
    cap = cv2.VideoCapture(video_path)
    crops: dict[int, list[np.ndarray]] = {}
    ab, cl = [], []
    for f, (t, lab) in enumerate(zip(tracks, labels)):
        ok, img_bgr = cap.read()
        if not ok:
            break
        for box, l in zip(t, lab):
            crop = team_classifier.torso_crop(img_bgr, box, tc["torso"])
            if f % 25 == 0 and l >= 0:
                crops.setdefault(int(l), []).append(crop)
            d = team_classifier.descriptor(crop, {**tc, "descriptor": "ab_mean"}, cfg["field_mask"])
            if d is not None and l >= 0:
                ab.append(d); cl.append(int(l))
    cv2.imwrite("outputs/figures/06_times_recortes.png", team_classifier.debug_view(crops, n=16))
    ab, cl = np.array(ab), np.array(cl)
    fig, ax = plt.subplots(figsize=(5, 4))
    for c in np.unique(cl):
        ax.scatter(ab[cl == c, 1], ab[cl == c, 0], s=2, alpha=.3, label=f"cluster {c}")
    ax.set_xlabel("b* (azul ← → amarelo)"); ax.set_ylabel("a* (verde ← → vermelho)")
    ax.set_title(f"Cor do tronco (rótulo: {tc['descriptor']} + votação)"); ax.legend(markerscale=5); fig.tight_layout()
    fig.savefig("outputs/figures/06_times_ab.png", dpi=120)
    print("figuras: outputs/figures/06_times_recortes.png, outputs/figures/06_times_ab.png")


def eval_heatmap(args: argparse.Namespace, cfg: dict) -> None:
    """Heatmaps vs. GT e ablações: suavização (manual × OpenCV × frequência), resolução da grade e
    ponto do pé × centro da bbox."""
    import time
    from collections import Counter
    from src import heatmap, pipeline
    pc, hc = cfg["pitch"], cfg["heatmap"]
    L, W = pc["length_m"], pc["width_m"]
    stem = Path(args.video).stem
    r = pipeline.run(args.video, cfg, f"outputs/{stem}", f"data/annotations/{stem}.json")
    gt, last = gt_roles(args.gt, L, W), r["last_frame"]
    tracks, labels = r["tracks"][:last + 1], r["labels"]
    pairs = match_gt(tracks, {i: g for i, g in gt.items() if i <= last})

    # cluster → time do GT (maioria entre os jogadores de linha casados)
    votes = Counter((int(labels[i][j]), g) for i, j, g in pairs if g in ("left", "right"))
    team_of = {c: max(("left", "right"), key=lambda g: votes[(c, g)]) for c in r["maps"]}
    print("cluster → time GT:", team_of)

    # 1. heatmap do GT (mesmos frames, células e suavização) × o nosso
    n = r["n_frames"]; sig = hc["sigma_m"] / hc["cell_m"]
    gt_maps = {}
    for side in ("left", "right"):
        P = np.vstack([gt[i][2][[l in (side, f"gk_{side}") for l in gt[i][1]]] for i in range(0, last + 1) if i in gt])
        gt_maps[side] = heatmap.smooth(heatmap.accumulate(P[~np.isnan(P).any(1)], L, W, hc["cell_m"]) / n, sig, hc["smoothing"])
    crop = lambda im: im[120:620, 600:]          # grande área direita (onde o clip01 acontece)
    rows = []
    for c, side in team_of.items():
        ours, ref = r["maps"][c], gt_maps[side]
        corr = np.corrcoef(ours.ravel(), ref.ravel())[0, 1]
        print(f"time {c} ({side}): correlação com o GT {corr:.3f}, massa {ours.sum():.2f} vs. GT {ref.sum():.2f} jogadores")
        vmax = max(ours.max(), ref.max())
        rows.append(np.hstack([crop(heatmap.colorize(ref, cfg, vmax=vmax)), crop(heatmap.colorize(ours, cfg, vmax=vmax))]))
    cv2.imwrite("outputs/figures/07_heatmap_vs_gt.png", np.vstack(rows))
    vmax = max(m.max() for m in r["maps"].values())
    cv2.imwrite("outputs/figures/07_heatmap_times.png", np.hstack(
        [crop(heatmap.colorize(r["maps"][c], cfg, vmax=vmax)) for c in sorted(r["maps"])] +
        [crop(heatmap.colorize(r["maps"][0] - r["maps"][1], cfg, cmap=hc["diff_colormap"], signed=True))]))

    # 2. espacial manual × cv2.GaussianBlur × frequência
    grid = heatmap.accumulate(r["positions"][:, 3:], L, W, hc["cell_m"]) / n
    res, times = {}, {}
    for name, fn in (("manual", heatmap.smooth_spatial), ("opencv", heatmap.smooth_opencv), ("frequência", heatmap.smooth_frequency)):
        t0 = time.perf_counter()
        for _ in range(20):
            res[name] = fn(grid, sig)
        times[name] = (time.perf_counter() - t0) / 20 * 1000
    rel = lambda a, b: np.abs(a - b).max() / res["manual"].max()
    print(f"suavização σ={sig:.0f} células, grade {grid.shape}: " + ", ".join(f"{k} {v:.2f} ms" for k, v in times.items()))
    print(f"diferença máx. relativa: manual×opencv {rel(res['manual'], res['opencv']):.1e}, "
          f"manual×frequência {rel(res['manual'], res['frequência']):.1e}")
    diff = heatmap.colorize(res["manual"] - res["frequência"], cfg, cmap=hc["diff_colormap"], signed=True)
    cv2.imwrite("outputs/figures/07_espacial_vs_frequencia.png",
                np.hstack([crop(heatmap.colorize(res["manual"], cfg)), crop(heatmap.colorize(res["frequência"], cfg)), crop(diff)]))

    # 3. resolução da grade (σ fixo em metros): bruto e suavizado
    tiles = []
    for cell in (0.25, 0.5, 1.0, 2.0):
        g = heatmap.accumulate(r["positions"][:, 3:], L, W, cell) / n
        c2 = {**cfg, "heatmap": {**hc, "cell_m": cell}}
        t = np.vstack([crop(heatmap.colorize(g, c2)), crop(heatmap.colorize(heatmap.smooth(g, hc["sigma_m"] / cell, "spatial"), c2))])
        cv2.putText(t, f"{cell} m", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        tiles.append(t)
    cv2.imwrite("outputs/figures/07_resolucao.png", np.hstack(tiles))

    # 4. ponto do pé × centro da bbox: erro da posição projetada vs. pé no campo do GT
    box = np.array([tracks[i][j, :4] for i, j, _ in pairs])
    ref = np.array([gt[i][2][int(np.argmax(iou(tracks[i][j:j + 1, :4], gt[i][0])[0]))] for i, j, _ in pairs])
    for name, p in (("pé", np.c_[(box[:, 0] + box[:, 2]) / 2, box[:, 3]]),
                    ("centro", np.c_[(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2])):
        proj = np.array([homography.project(r["Hs"][i], q[None])[0] for (i, _, _), q in zip(pairs, p)])
        err = np.linalg.norm(proj - ref, axis=1)
        print(f"ponto {name}: erro mediano {np.nanmedian(err):.2f} m, p90 {np.nanpercentile(err, 90):.2f} m ({len(err)} detecções)")


GT_MAIN = ("Main camera center", "Main camera left", "Main camera right")


def gt_main_frames(cameras_json: str, half: str, n_frames: int, fps: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rótulo do gabarito por quadro do vídeo do tempo `half`: (é câmera principal ao vivo, índice do rótulo,
    nomes dos rótulos; índice + len(nomes) = replay).
    Cada anotação marca uma troca de câmera e o rótulo descreve a tomada que termina nela (conferido
    nos quadros do Liverpool × City: aos 31 s, rotulado "Main camera center" na troca de 29,9 s, já é close-up;
    de 63 a 87 s, rotulado "Close-up" na troca de 63,5 s, é câmera principal). Depois da última troca não há gabarito."""
    ann = sorted((a for a in json.load(open(cameras_json))["annotations"] if a["gameTime"].startswith(half + " ")),
                 key=lambda a: int(a["position"]))
    main, lab = np.zeros(n_frames, bool), np.full(n_frames, -1)
    labels = sorted({a["label"] for a in json.load(open(cameras_json))["annotations"]})
    for prev, a in zip([None] + ann[:-1], ann):
        f0 = 0 if prev is None else round(int(prev["position"]) / 1000 * fps)
        f1 = round(int(a["position"]) / 1000 * fps)
        main[f0:f1] = a["label"] in GT_MAIN and a["replay"] == "real-time"
        lab[f0:f1] = labels.index(a["label"]) + (len(labels) if a["replay"] != "real-time" else 0)
    return main, lab, labels


def eval_shots(args: argparse.Namespace, cfg: dict) -> None:
    from src import shots as shots_mod
    M = json.load(open(args.match)); name = Path(args.match).stem
    tot = np.zeros(4)                                    # vp, fp, fn, vn (quadros)
    fp_by, fn_by, rows = {}, {}, []
    for half, v in M["tempos"].items():
        cap = cv2.VideoCapture(v["video"]); fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gt, lab, labels = gt_main_frames(M["soccernet_cameras"], half, n, fps)
        ours = np.zeros(n, bool)
        for s in json.load(open(f"outputs/{name}/{half}/shots.json")):
            ours[s["frame_a"]:s["frame_b"]] = s["main"]
        f0, f1 = round(shots_mod.mmss(v["inicio"]) * fps), round(shots_mod.mmss(v["fim"]) * fps)
        g, o, l = gt[f0:f1], ours[f0:f1], lab[f0:f1]
        g, o, l = g[l >= 0], o[l >= 0], l[l >= 0]          # antes da 1ª anotação do tempo não há gabarito
        tot += [(g & o).sum(), (~g & o).sum(), (g & ~o).sum(), (~g & ~o).sum()]
        for d, m in ((fp_by, ~g & o), (fn_by, g & ~o)):
            for i, c in zip(*np.unique(l[m], return_counts=True)):
                k = "?" if i < 0 else labels[i % len(labels)] + (" (replay)" if i >= len(labels) else "")
                d[k] = d.get(k, 0) + c / fps
        rows.append((half, gt[f0:f1], ours[f0:f1], fps))   # figura: linha do tempo inteira
    vp, fp, fn, vn = tot
    print(f"quadros: VP {vp:.0f} FP {fp:.0f} FN {fn:.0f} VN {vn:.0f}")
    print(f"precisão {vp / (vp + fp):.3f}  revocação {vp / (vp + fn):.3f}  F1 {2 * vp / (2 * vp + fp + fn):.3f}  "
          f"acurácia {(vp + vn) / tot.sum():.3f}")
    print("falsos positivos (s) por rótulo do gabarito:", {k: round(v) for k, v in sorted(fp_by.items(), key=lambda kv: -kv[1])})
    print("falsos negativos (s) por rótulo do gabarito:", {k: round(v) for k, v in sorted(fn_by.items(), key=lambda kv: -kv[1])})
    fig, axs = plt.subplots(len(rows), 1, figsize=(12, 1.4 * len(rows)), squeeze=False)
    for ax, (half, g, o, fps) in zip(axs[:, 0], rows):
        t = np.arange(len(g)) / fps / 60
        ax.fill_between(t, 1, 2, where=g, color="tab:green", step="post", label="gabarito (principal ao vivo)")
        ax.fill_between(t, 0, 1, where=o, color="tab:blue", step="post", label="filtro")
        ax.set_yticks([0.5, 1.5], ["filtro", "gabarito"]); ax.set_xlim(0, t[-1]); ax.set_title(f"{half}º tempo", fontsize=9)
    axs[-1, 0].set_xlabel("min do vídeo")
    fig.tight_layout(); fig.savefig(f"outputs/figures/08_tomadas_vs_gt_{name}.png", dpi=150)


def eval_camera(args: argparse.Namespace, cfg: dict) -> None:
    """(a) Relocalização sem marcação, avaliada nos quadros marcados: para cada marcação, a busca em grade
    (montada com as OUTRAS marcações) acha a câmera só pelas linhas, e os cliques dão o erro em px.
    (b) Quadros com câmera aceita (outputs/<partida>/cameras.json, de run_match) contra o Labels-cameras.json
    do SoccerNet: quantos são de fato da câmera principal ao vivo."""
    from src import camera, detection, lines, match
    from src.field_mask import field_mask
    M = json.load(open(args.match)); name = Path(args.match).stem
    cfg = match.match_cfg(cfg, M)
    lc, pc, step = cfg["lines"], cfg["pitch"], cfg["video"]["frame_step"]
    ann = json.load(open(f"data/annotations/{name}_homografias.json"))
    marked = sorted((s for s in ann.values() if "H_img_to_pitch" in s), key=lambda s: (s["tempo"], s["frame_a"]))
    kp = pitch.keypoints(pc["length_m"], pc["width_m"])
    cap = cv2.VideoCapture(M["tempos"]["1"]["video"]); w, h = int(cap.get(3)), int(cap.get(4))
    C, good = camera.from_marks(marked, kp, w, h, lc["center_max_resid"])
    assert len(good) == len(marked)
    samples = lines.model_samples(pc["length_m"], pc["width_m"], lc["sample_m"])
    print("(a) relocalização nos quadros marcados (grade sem a própria marcação):")
    errs, oks = [], []
    for j, s in enumerate(marked):
        video = M["tempos"][s["tempo"]]["video"]
        a, k = s["frame_a"], s["frame"] - s["frame_a"]
        a0 = a + k % step
        dets = detection.detect_video(video, cfg, Path(f"outputs/{name}/{s['tempo']}/tomadas/{a0}/detections.npz"),
                                      start=a0, end=s["frame_b"])
        cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"]); _, img = cap.read()
        lm, region = lines.line_mask(img, field_mask(img, cfg["field_mask"]), dets[s["frame"] - a0], lc)
        grid = lines.pose_grid([g for i, g in enumerate(good) if i != j], lc)
        p, sc = lines.relocalize(grid, C, lines.distance_map(lm, lc["max_px"]), region, samples, lc)
        pi = np.array([q["img"] for q in s["points"]], float); pp = np.array([kp[q["name"]] for q in s["points"]])
        e = np.inf if p is None else float(np.median(np.linalg.norm(camera.project(p, C, w, h, pp) - pi, axis=1)))
        e_man = float(np.median(np.linalg.norm(camera.project(good[j], C, w, h, pp) - pi, axis=1)))
        errs.append((e, e_man)); oks.append(sc >= lc["reloc_min_score"])
        print(f"  {s['tempo']}:{a:<6d} encaixe {sc:.2f} {'aceita ' if oks[-1] else 'recusada'}  erro {e:6.1f} px (manual {e_man:.1f})")
    e, e_man, ok = np.array(errs)[:, 0], np.array(errs)[:, 1], np.array(oks)
    print(f"  aceitas {ok.sum()}/{len(ok)}; erro mediano das aceitas {np.median(e[ok]):.1f} px (manual {np.median(e_man[ok]):.1f});"
          f" aceitas com erro > 10 px: {(e[ok] > 10).sum()}; recusadas com erro <= 10 px: {(e[~ok] <= 10).sum()}")

    print("(b) quadros com câmera aceita × Labels-cameras do SoccerNet:")
    cams = json.load(open(f"outputs/{name}/cameras.json"))
    for auto in (False, True):
        by, n_ok, hit, n_gt = {}, 0, 0, 0
        for half, v in M["tempos"].items():
            cap = cv2.VideoCapture(v["video"]); fps = cap.get(cv2.CAP_PROP_FPS)
            gt, lab, labels = gt_main_frames(M["soccernet_cameras"], half, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps)
            okf = np.array([f for c in cams if c["tempo"] == half and c["auto"] == auto for f in c["ok"]], int)
            okf = okf[lab[okf] >= 0]                       # só onde há gabarito
            n_ok += len(okf); hit += int(gt[okf].sum())
            for i, c in zip(*np.unique(lab[okf], return_counts=True)):
                key = labels[i % len(labels)] + (" (replay)" if i >= len(labels) else "")
                by[key] = by.get(key, 0) + int(c)
        print(f"  {'sem marcação' if auto else 'marcadas':13s}: {n_ok} quadros aceitos, {hit / max(n_ok, 1):.1%} são câmera principal "
              f"ao vivo no gabarito; por rótulo: {dict(sorted(by.items(), key=lambda kv: -kv[1]))}")


def eval_goalkeeper(args: argparse.Namespace, cfg: dict) -> None:
    """Regra do goleiro pela posição (`match.goalkeeper_tracks`) sobre o gabarito do SoccerNet GSR: posições e IDs do
    próprio gabarito, 1 a cada `match.frame_step` quadros, e cada sequência como uma tomada. Mede se o ID escolhido em
    cada gol é o goleiro e se o time atribuído (o que defende aquele gol) é o do gabarito, cuja chave `team` é o
    lado que o time defende. Compara a regra só do mais recuado com a regra completa."""
    from src import match
    L, W, cm = 105.0, 68.0, cfg["match"]
    rules = {"só o mais recuado": cm | {"gk_min_rows": np.inf}, "completa": cm}
    tot = {k: np.zeros(4, int) for k in rules}           # IDs escolhidos, time certo, são goleiro, goleiros achados
    n_gk, wrong = 0, {k: [] for k in rules}
    n_field, field_wrong = 0, {k: 0 for k in rules}      # posições de jogadores de linha; quantas a regra erra
    for gt in sorted(Path(args.gt_dir).glob("SNGS-*/Labels-GameState.json")):
        rows, info = [], {}
        for a in json.load(open(gt))["annotations"]:
            f = int(a["image_id"]) % 1_000_000 - 1
            if a["category_id"] not in (1, 2, 3) or not a.get("bbox_pitch") or f % cm["frame_step"]:
                continue
            p = a["bbox_pitch"]
            rows.append((f, a["track_id"], 0, p["x_bottom_middle"] + L / 2, p["y_bottom_middle"] + W / 2))
            info[a["track_id"]] = (a["attributes"]["role"], a["attributes"]["team"])
        pos = np.array(rows, float)
        near = np.abs(pos[:, 4] - W / 2) < pitch.PENALTY_WIDTH / 2
        # goleiros que dá para achar: pelo menos gk_min_rows quadros junto ao próprio gol
        for tid, (role, team) in info.items():
            d = pos[:, 3] if team == "left" else L - pos[:, 3]
            if role == "goalkeeper" and ((pos[:, 1] == tid) & near & (d < cm["gk_zone_m"])).sum() >= cm["gk_min_rows"]:
                n_gk += 1
            if role == "player":
                n_field += int((pos[:, 1] == tid).sum())
        for k, c in rules.items():
            for tid, side in match.goalkeeper_tracks(pos, L, W, c).items():
                role, team = info[tid]
                ok = team == ("left" if side == 0 else "right")
                tot[k] += [1, ok, role == "goalkeeper", role == "goalkeeper" and ok]
                if not ok:
                    wrong[k].append(f"{gt.parent.name}:{tid} ({role}, {team})")
                    field_wrong[k] += int((pos[:, 1] == tid).sum()) if role == "player" else 0
    print(f"{n_gk} goleiros com pelo menos {cm['gk_min_rows']} quadros junto ao próprio gol, em "
          f"{len(list(Path(args.gt_dir).glob('SNGS-*')))} sequências")
    for k, (n, ok, is_gk, found) in tot.items():
        print(f"{k:18s}: {n} IDs escolhidos; time certo {ok}/{n} ({ok / max(n, 1):.1%}); são o goleiro {is_gk}/{n}; "
              f"goleiros achados {found}/{n_gk} ({found / max(n_gk, 1):.1%}); posições de jogadores de linha "
              f"passadas ao time errado {field_wrong[k]}/{n_field} ({field_wrong[k] / max(n_field, 1):.2%})")
        if wrong[k]:
            print("   time errado:", ", ".join(wrong[k]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("homography")
    h.add_argument("annotation"); h.add_argument("--gt", required=True); h.add_argument("--gt-reference", action="store_true")
    h.add_argument("--propagate", action="store_true"); h.add_argument("--video", default=None)
    t = sub.add_parser("team")
    t.add_argument("video"); t.add_argument("--gt", required=True)
    t.add_argument("--tracks", default="outputs/clip01/cache/tracks.npz")
    hm = sub.add_parser("heatmap")
    hm.add_argument("video"); hm.add_argument("--gt", required=True)
    sh = sub.add_parser("shots")
    sh.add_argument("match")
    ca = sub.add_parser("camera")
    ca.add_argument("match")
    gk = sub.add_parser("goleiro")
    gk.add_argument("--gt-dir", default="data/raw/SoccerNetGS/valid")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.cmd == "homography":
        eval_homography(args, cfg)
    elif args.cmd == "team":
        eval_team(args, cfg)
    elif args.cmd == "heatmap":
        eval_heatmap(args, cfg)
    elif args.cmd == "shots":
        eval_shots(args, cfg)
    elif args.cmd == "camera":
        eval_camera(args, cfg)
    elif args.cmd == "goleiro":
        eval_goalkeeper(args, cfg)


if __name__ == "__main__":
    main()
