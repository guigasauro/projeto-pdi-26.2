"""CLI principal.

  python scripts/run_pipeline.py data/clips/clip01.mp4 [--config configs/default.yaml] [--out outputs/clip01]

Usa a homografia de data/annotations/<clip>.json (scripts/annotate_homography.py) e salva os heatmaps
em <out>/; detecções, tracks e times ficam em cache em <out>/cache/ (apague para recalcular).
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--annotation", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stem = Path(args.video).stem
    pipeline.run(args.video, yaml.safe_load(open(args.config)), args.out or f"outputs/{stem}",
                 args.annotation or f"data/annotations/{stem}.json")


if __name__ == "__main__":
    main()
