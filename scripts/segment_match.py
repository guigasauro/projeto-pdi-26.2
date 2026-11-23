"""Separa uma partida completa em tomadas e marca quais são da câmera principal ao vivo.

  python scripts/segment_match.py data/annotations/barcelona-x-real_madrid.json

O JSON da partida traz os vídeos de cada tempo, início/fim (mm:ss) e o recorte do placar.
Saída: outputs/<partida>/<tempo>/shots.json (tomadas com índices de quadro do vídeo e `main`)
e outputs/figures/08_tomadas_<partida>.png (linha do tempo de cada tempo).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detection, shots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    sc = cfg["shots"]
    M = json.load(open(args.match))
    name = Path(args.match).stem
    template = cv2.imread(M["placar"]["template"])
    model = detection.load_model(cfg["detection"])
    device = detection.resolve_device(cfg["detection"]["device"])
    lines = []
    for half, v in M["tempos"].items():
        out = Path("outputs") / name / half
        R = shots.scan(v["video"], shots.mmss(v["inicio"]), shots.mmss(v["fim"]), cfg, template,
                       M["placar"]["roi"], out / "cache" / "shots.npz")
        cap = cv2.VideoCapture(v["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        table = []
        for a, b in shots.segments(R["dist"], R["board"], sc):
            grass = float(np.median(R["grass"][a:b]))
            board = float((R["board"][a:b] > sc["board_thresh"]).mean())
            people, height = shots.probe(cap, int(R["frame"][(a + b) // 2]), model, cfg, device) \
                if b - a >= sc["probe_min_frames"] else (0, 0.0)
            table.append({"a": a, "b": b, "frame_a": int(R["frame"][a]), "frame_b": int(R["frame"][b - 1]) + 1,
                          "grass": grass, "board": board, "people": people, "height": height,
                          "main": shots.is_main(grass, board, people, height / cap.get(cv2.CAP_PROP_FRAME_HEIGHT), sc)})
        (out / "shots.json").write_text(json.dumps(table, indent=1))
        n_main = sum(s["main"] for s in table)
        t_main = sum(s["b"] - s["a"] for s in table if s["main"]) / fps
        print(f"tempo {half}: {len(table)} tomadas, {n_main} da câmera principal = {t_main / 60:.1f} de "
              f"{len(R['dist']) / fps / 60:.1f} min ({t_main * fps / len(R['dist']):.0%})")
        tl = shots.timeline_view(table, len(R["dist"]), fps)
        cv2.putText(tl, f"{half}o tempo", (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        lines.append(tl)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"outputs/figures/08_tomadas_{name}.png", np.vstack(lines))


if __name__ == "__main__":
    main()
