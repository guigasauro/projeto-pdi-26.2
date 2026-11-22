# Mapa de calor de jogadores em vídeos de futebol

Projeto final de GDSCO0055 – Introdução ao PDI (UFPB, 2026.2). Detalhes em `CLAUDE.md`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
# torch primeiro (fora do requirements.txt, depende da máquina):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # sem GPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # GPU NVIDIA (só precisa do driver)
pip install -r requirements.txt
# pesos do YOLO (~5 MB) em models/:
cd models && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" && cd ..
pytest -q
```

## Dados
Fonte: **SoccerNet Game State Reconstruction** (`gamestate-2024`, split `valid`, v1.3): 58 sequências de 30 s,
1920×1080, 25 fps, JPG, com bbox, posição no campo (m), time e papel de cada pessoa e linhas do campo por frame.
```python
from SoccerNet.Downloader import SoccerNetDownloader
SoccerNetDownloader(LocalDirectory="data/raw/SoccerNetGS").downloadDataTask(task="gamestate-2024", split=["valid"])
# depois: unzip data/raw/SoccerNetGS/gamestate-2024/valid.zip -d data/raw/SoccerNetGS/valid
```

| Clipe | Origem | Por quê |
|---|---|---|
| `data/clips/clip01.mp4` | `valid/SNGS-056` (pênalti, 750 frames), recodificado com `cv2.VideoWriter` (mp4v, PSNR ≈ 30 dB vs. JPG original) | Menor movimento de câmera do split (~395 px acumulados em 30 s; mediana 4900 px) e grande área inteira visível |

Coordenadas do SoccerNet no campo têm origem no **centro** (x ∈ [-52.5, 52.5], y ∈ [-34, 34]); este projeto usa origem no canto superior esquerdo (somar 52,5 e 34).

## Uso
```bash
python scripts/annotate_homography.py data/clips/clip01.mp4     # marca >= 4 pontos do campo (v1, uma vez por clipe)
python scripts/run_pipeline.py data/clips/clip01.mp4            # heatmaps em outputs/clip01/ (cache em outputs/clip01/cache/)
# métricas e figuras vs. GT do SoccerNet (outputs/figures/):
GT=data/raw/SoccerNetGS/valid/SNGS-056/Labels-GameState.json
python scripts/evaluate.py homography data/annotations/clip01.json --gt $GT
python scripts/evaluate.py team data/clips/clip01.mp4 --gt $GT
python scripts/evaluate.py heatmap data/clips/clip01.mp4 --gt $GT
```
A homografia v1 é fixa: só vale enquanto a câmera está parada (`homography.valid_until_frame` no YAML).
