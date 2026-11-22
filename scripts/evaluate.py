"""Métricas contra o gabarito do SoccerNet.

  python scripts/evaluate.py homography data/annotations/clip01.json --gt data/raw/SoccerNetGS/valid/SNGS-056/Labels-GameState.json

`homography`: projeta os pés dos jogadores do GT (imagem) com a H anotada e compara com a posição
real no campo (GT, metros), frame a frame. Mostra quanto uma H fixa se degrada quando a câmera mexe.
`--gt-reference` calcula a H a partir dos próprios pés do GT no frame anotado (referência ideal).

  python scripts/evaluate.py team data/clips/clip01.mp4 --gt .../Labels-GameState.json

`team`: casa os tracks em cache com o GT (IoU >= 0,5) e mede a acurácia do time (left/right) dos jogadores
de linha, por descritor e com/sem remoção de grama e votação temporal (ablações 1 e 2).
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
    err = [homography.reprojection_error(H, *feet[i]) for i in frames]
    med = np.array([np.median(e) for e in err]); p90 = np.array([np.percentile(e, 90) for e in err])
    print(f"H {label} (frame {f0}): erro no frame {f0} = {med[frames.index(f0)]:.2f} m (mediana) | "
          f"ao longo do clipe: mediana {np.median(med):.2f} m, pior frame {med.max():.2f} m (frame {frames[int(med.argmax())]})")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(frames, med, label="mediana"); ax.plot(frames, p90, alpha=.6, label="percentil 90")
    ax.set_xlabel("frame"); ax.set_ylabel("erro (m)"); ax.set_title(f"Erro dos pés projetados com H fixa ({label})")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    out = Path("outputs/figures") / f"05_erro_homografia_{Path(args.annotation).stem}{'_gtref' if args.gt_reference else ''}.png"
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
        err = np.linalg.norm(homography.project(r["H"], p) - ref, axis=1)
        print(f"ponto {name}: erro mediano {np.nanmedian(err):.2f} m, p90 {np.nanpercentile(err, 90):.2f} m ({len(err)} detecções)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("homography")
    h.add_argument("annotation"); h.add_argument("--gt", required=True); h.add_argument("--gt-reference", action="store_true")
    t = sub.add_parser("team")
    t.add_argument("video"); t.add_argument("--gt", required=True)
    t.add_argument("--tracks", default="outputs/clip01/cache/tracks.npz")
    hm = sub.add_parser("heatmap")
    hm.add_argument("video"); hm.add_argument("--gt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.cmd == "homography":
        eval_homography(args, cfg)
    elif args.cmd == "team":
        eval_team(args, cfg)
    elif args.cmd == "heatmap":
        eval_heatmap(args, cfg)


if __name__ == "__main__":
    main()
