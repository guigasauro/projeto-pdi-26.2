"""Heatmaps de uma partida completa a partir das tomadas marcadas.

  python scripts/run_match.py data/annotations/barcelona-x-real_madrid.json

Pré-requisitos: scripts/segment_match.py e scripts/annotate_homography.py --match. Pode rodar de novo
conforme mais tomadas forem marcadas: só as novas são processadas. Saída em outputs/<partida>/.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import match  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    match.run(args.match, yaml.safe_load(open(args.config)), args.out)


if __name__ == "__main__":
    main()
