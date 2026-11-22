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


def gt_roles(gt_path: str) -> dict[int, tuple[np.ndarray, list[str]]]:
    """{frame: (caixas GT (N, 4), rótulo)} com rótulo = left | right (jogador), gk_<time> ou referee."""
    out: dict[int, tuple[list, list]] = {}
    for a in json.load(open(gt_path))["annotations"]:
        if a["category_id"] in (1, 2, 3):
            b, at = a["bbox_image"], a["attributes"]
            lab = at["team"] if a["category_id"] == 1 else f"gk_{at['team']}" if a["category_id"] == 2 else "referee"
            f = out.setdefault(int(a["image_id"][-6:]) - 1, ([], []))
            f[0].append((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])); f[1].append(lab)
    return {i: (np.array(v[0], float), v[1]) for i, v in out.items()}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("homography")
    h.add_argument("annotation"); h.add_argument("--gt", required=True); h.add_argument("--gt-reference", action="store_true")
    t = sub.add_parser("team")
    t.add_argument("video"); t.add_argument("--gt", required=True)
    t.add_argument("--tracks", default="outputs/clip01/cache/tracks.npz")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.cmd == "homography":
        eval_homography(args, cfg)
    elif args.cmd == "team":
        eval_team(args, cfg)


if __name__ == "__main__":
    main()
