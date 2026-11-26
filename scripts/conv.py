"""Quantas tomadas bastam: confiabilidade metade-metade dos heatmaps por time (Spearman-Brown).

  python scripts/conv.py data/annotations/barcelona-x-real_madrid.json

Roda o pipeline da partida (tudo vem do cache de outputs/<partida>/) e repete a análise para as tomadas
marcadas e para todas (marcadas + sem marcação, `match.auto_shots`). Para cada grupo: correlação média
(mínima) entre o mapa de 25/50/75% das tomadas sorteadas e o mapa do grupo inteiro; correlação entre duas
metades disjuntas e a confiabilidade do grupo inteiro pela fórmula de Spearman-Brown.
"""
import json
import logging
import sys

import numpy as np
import yaml

sys.path.insert(0, ".")
from src import match, pipeline

logging.basicConfig(level=logging.WARNING)
match_json = sys.argv[1]
base = yaml.safe_load(open("configs/default.yaml"))
cfg = match.match_cfg(base, json.load(open(match_json)))
r = match.run(match_json, base)
L, W = cfg["pitch"]["length_m"], cfg["pitch"]["width_m"]
ref = r["left"][min(r["left"])]
S_all = [(match.rotate_half(s["pos"], L, W) if r["left"][s["tempo"]] != ref else s["pos"],
          int(np.isfinite(s["Hs"]).all((1, 2)).sum()), bool(s.get("auto"))) for s in r["shots"]]


def corr(a: dict, b: dict) -> float:
    return float(np.mean([np.corrcoef(a[t].ravel(), b[t].ravel())[0, 1] for t in (0, 1)]))


for nome, S in (("marcadas", [s for s in S_all if not s[2]]), ("todas", S_all)):
    def maps(idx):
        return pipeline.team_heatmaps(np.vstack([S[i][0] for i in idx]), sum(S[i][1] for i in idx), cfg)
    n = len(S); full = maps(range(n)); rng = np.random.default_rng(0)
    print(f"\n== {nome}: {n} tomadas")
    for frac in (0.25, 0.5, 0.75):
        c = [corr(maps(rng.choice(n, int(frac * n), replace=False)), full) for _ in range(30)]
        print(f"{frac:.0%} das tomadas vs todas: correlação média {np.mean(c):.3f} (min {np.min(c):.3f})")
    sh = []
    for _ in range(30):
        p = rng.permutation(n); sh.append(corr(maps(p[:n // 2]), maps(p[n // 2:])))
    rh = np.mean(sh); rfull = 2 * rh / (1 + rh)
    print(f"metade × outra metade: {rh:.3f} → Spearman-Brown, todas as {n} tomadas: {rfull:.3f}")
    for k in (2, 3):
        print(f"previsto com {k}x as tomadas: {k * rfull / (1 + (k - 1) * rfull):.3f}")
