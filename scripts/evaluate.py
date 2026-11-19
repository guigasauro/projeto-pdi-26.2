"""Métricas contra o gabarito do SoccerNet.

  python scripts/evaluate.py homography data/annotations/clip01.json --gt data/raw/SoccerNetGS/valid/SNGS-056/Labels-GameState.json

`homography`: projeta os pés dos jogadores do GT (imagem) com a H anotada e compara com a posição
real no campo (GT, metros), frame a frame. Mostra quanto uma H fixa se degrada quando a câmera mexe.
`--gt-reference` calcula a H a partir dos próprios pés do GT no frame anotado (referência ideal).
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
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import homography, pitch  # noqa: E402


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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("homography")
    h.add_argument("annotation"); h.add_argument("--gt", required=True); h.add_argument("--gt-reference", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.cmd == "homography":
        eval_homography(args, cfg)


if __name__ == "__main__":
    main()
