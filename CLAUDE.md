# CLAUDE.md — Mapa de Calor de Jogadores em Vídeos de Futebol

> Projeto final da disciplina **GDSCO0055 – Introdução ao Processamento Digital de Imagens** (UFPB, 2026.2, Prof. Leonardo Vidal Batista).
> **Entrega e apresentação: 16/12/2026.**
> Autor: Guilherme Nogueira da Silva.

---

## 1. Visão geral

Pipeline que recebe um **vídeo de futebol** e produz **mapas de calor da movimentação dos jogadores** projetados sobre um campo visto de cima (*top-view*), separados por **time** (e opcionalmente por jogador).

```
vídeo → detecção de jogadores → rastreamento (IDs) → classificação de time por cor
      → detecção do gramado e linhas do campo → homografia imagem→campo
      → projeção dos pés no campo → acumulação → suavização → pseudocoloração → heatmap
```

### Título de trabalho
"Geração de mapas de calor de jogadores de futebol a partir de vídeo usando segmentação por cor, classificação de times por histograma e mapeamento por homografia"

### Princípio central (IMPORTANTE)
Este é um projeto de **Processamento Digital de Imagens**, não de deep learning. A rede neural (YOLO) é usada **somente** para detectar as pessoas. **Todo o resto deve ser feito com técnicas clássicas de PDI**, implementadas de forma explícita e explicável. Sempre que existir uma função "caixa-preta" pronta, preferir:
1. usar OpenCV/NumPy em etapas visíveis, ou
2. implementar a técnica manualmente (ao menos uma versão didática) e comparar com a da biblioteca.

---

## 2. Mapeamento com o plano de curso

Cada etapa do pipeline deve citar o conteúdo da disciplina correspondente (isso vai para os slides e para o relatório):

| Etapa do pipeline | Conteúdo da disciplina |
|---|---|
| Conversão BGR→HSV/Lab, máscara do gramado, cor do uniforme | Sistemas de cores, fundamentos de cor |
| Limiarização da máscara do gramado | Operações locais e pontuais |
| Histograma de cor do uniforme, equalização para iluminação | Processamento de histograma |
| Remoção de ruído da máscara, suavização antes de Canny | Filtros de suavização |
| Canny / Sobel nas linhas do campo | Filtros de detecção de bordas |
| Kernel gaussiano aplicado ao heatmap | Correlação e convolução |
| Homografia, *warp* para top-view, escala do campo | Operações geométricas (escala, rotação, transformações) |
| Suavização do heatmap no domínio da frequência (experimento) | DFT, filtros no domínio da frequência |
| Colormap do heatmap | Pseudocoloração |
| Grade discreta do campo (resolução do heatmap) | Amostragem e quantização |

---

## 3. Stack

- **Python 3.11+**
- `opencv-python`: núcleo do PDI
- `numpy`, `scipy`
- `ultralytics`: YOLOv8/YOLO11, só para detecção de pessoas/bola
- `supervision`: ByteTrack para rastreamento e utilitários de anotação
- `scikit-learn`: k-means para separar os times
- `matplotlib`: figuras para relatório e slides
- `tqdm`, `pyyaml`
- `jupyter`: notebooks de exploração e figuras

Gerenciar dependências com `requirements.txt` (versões fixadas) e ambiente virtual em `.venv/`.

---

## 4. Estrutura do repositório

```
heatmap-futebol/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # todos os parâmetros (limiares HSV, k, resolução, sigma...)
├── data/
│   ├── raw/                  # vídeos originais (NÃO versionar)
│   ├── clips/                # trechos curtos recortados para testes
│   └── annotations/          # pontos de homografia manuais (JSON), ground truth
├── models/                   # pesos YOLO (NÃO versionar)
├── src/
│   ├── __init__.py
│   ├── io_video.py           # leitura/escrita de vídeo, amostragem de frames
│   ├── detection.py          # wrapper YOLO → lista de bboxes por frame
│   ├── tracking.py           # ByteTrack → IDs persistentes
│   ├── field_mask.py         # segmentação do gramado (HSV + morfologia)
│   ├── team_classifier.py    # extração de cor do uniforme + k-means + votação temporal
│   ├── lines.py              # detecção de linhas do campo (Canny + Hough)
│   ├── homography.py         # homografia manual, propagação entre frames, automática
│   ├── pitch.py              # modelo do campo (105x68 m), desenho do campo top-view
│   ├── heatmap.py            # acumulação, suavização (espacial e frequência), colormap
│   ├── visualize.py          # overlays, painéis lado a lado, vídeo anotado
│   └── pipeline.py           # orquestra tudo
├── scripts/
│   ├── cut_clip.py           # recorta trecho de vídeo
│   ├── annotate_homography.py# ferramenta de clique para marcar pontos do campo
│   ├── run_pipeline.py       # CLI principal
│   └── evaluate.py           # métricas
├── notebooks/
│   ├── 01_mascara_gramado.ipynb
│   ├── 02_cores_times.ipynb
│   ├── 03_linhas_homografia.ipynb
│   ├── 04_heatmap.ipynb
│   └── 05_figuras_relatorio.ipynb
├── tests/
│   └── test_*.py
└── outputs/                  # resultados gerados (NÃO versionar, exceto figuras finais)
    ├── figures/
    ├── heatmaps/
    └── videos/
```

`.gitignore` deve conter: `data/raw/`, `data/clips/`, `models/`, `outputs/` (exceto `outputs/figures/final/`), `.venv/`, `*.mp4`, `*.pt`.

---

## 5. Dados

### Fontes
- **DFL – Bundesliga Data Shootout (Kaggle):** vídeos de partidas em câmera aberta (*wide*), boa opção para começar.
- **SoccerNet** (tracking, calibração de câmera, jersey): tem anotações de rastreamento e de linhas do campo, úteis para avaliação. Alguns vídeos exigem aceitar um termo (NDA) no site.
- **Roboflow Universe:** datasets de "football players detection" (classes jogador, goleiro, árbitro, bola) para *fine-tuning* opcional do YOLO.

### Regras
- Trabalhar sempre com **clipes curtos** (10–30 s) em `data/clips/` durante o desenvolvimento.
- Preferir clipes com **câmera aberta e pouco movimento/zoom** para a versão 1.
- Documentar no `README.md` a origem de cada clipe usado nos resultados.

---

## 6. Especificação de cada módulo

### 6.1 `detection.py`
- YOLO pré-treinado (COCO, classe `person`) como padrão; *fine-tuning* com dataset de futebol fica opcional.
- Filtrar detecções: confiança ≥ `det.conf` (padrão 0.3), área mínima, e **somente dentro da máscara do gramado** (pé da bbox sobre pixel de gramado). Assim saem torcida, banco e placas.
- Saída: `np.ndarray` (N, 6) → `x1, y1, x2, y2, conf, cls`.

### 6.2 `tracking.py`
- ByteTrack via `supervision`. Saída: detecções + `tracker_id`.
- Guardar histórico por ID: `{id: [(frame, x_pe, y_pe), ...]}`.
- **Ponto do pé** = centro da base da bbox `((x1+x2)/2, y2)`. É esse ponto que vai para a homografia (não o centro da bbox).

### 6.3 `field_mask.py`: segmentação do gramado
1. BGR → HSV.
2. Limiarização no canal H (faixa de verde) + S e V mínimos. Limiares em `configs/default.yaml`.
3. **Mostrar e justificar** os limiares com o histograma do canal H de frames de amostra (figura para o relatório).
4. Abertura/fechamento morfológico + maior componente conexo → máscara do campo.
5. Opcional: comparar com limiar automático (Otsu no canal H ou em `G - (R+B)/2`).

### 6.4 `team_classifier.py`: separação de times
1. Para cada bbox, recortar a **região do tronco** (≈ 20%–60% da altura, 20%–80% da largura), evitando cabeça, shorts e grama.
2. Remover pixels de gramado do recorte usando a máscara HSV.
3. Descritor de cor (implementar e comparar pelo menos dois):
   - cor média/mediana em **Lab** (mais robusta à iluminação que RGB);
   - histograma H-S normalizado (ex.: 16×8 bins).
4. **Calibração:** coletar descritores dos primeiros N frames (padrão 50) e rodar **k-means com k=2** (times). Detectar árbitro/goleiros como *outliers* (distância ao centróide > limiar) ou com k=3–4 e rótulo manual dos clusters.
5. **Votação temporal por `tracker_id`**: o time de um ID é a moda das classificações ao longo do tempo. Isso corrige erros de frames isolados.
6. Gerar figura: recortes de exemplo de cada cluster + gráfico dos descritores (scatter a/b do Lab colorido por cluster).

### 6.5 `lines.py`: linhas do campo
1. Máscara de branco (V alto, S baixo) **dentro** da máscara do gramado.
2. Suavização gaussiana → Canny (comparar com Sobel manual implementado via convolução).
3. Transformada de Hough probabilística → segmentos.
4. Agrupar segmentos por ângulo (linhas longitudinais vs. transversais) e calcular interseções. Esses são candidatos a pontos-chave do campo.

### 6.6 `homography.py`: mapeamento imagem → campo
Implementar em **três níveis**, nesta ordem (só avançar quando o anterior funcionar):

- **v1 – Manual (obrigatório):** `scripts/annotate_homography.py` abre o 1º frame, o usuário clica em ≥ 4 pontos conhecidos (cantos da grande área, meio-campo, interseções) e escolhe a coordenada correspondente no modelo do campo. Calcula `cv2.findHomography` (RANSAC) e salva em `data/annotations/<clip>.json`. Vale para clipes com câmera estática.
- **v2 – Propagação (recomendado):** para câmera que se move, estimar a transformação **entre frames consecutivos** com ORB/SIFT + matching **restrito ao gramado e fora das bboxes dos jogadores** (jogadores se movem, o campo não). Compor `H_t = H_{t-1} · H_{t→t-1}`. Re-ancorar com a v1 a cada K frames ou quando o erro acumular.
- **v3 – Automática (extra):** usar as interseções de `lines.py` e casar com os pontos do modelo do campo.

Validação: projetar o desenho do campo (modelo) de volta sobre o frame com `H⁻¹` e verificar visualmente o alinhamento. Gerar esse overlay sempre (é a principal figura de diagnóstico).

### 6.7 `pitch.py`
- Campo padrão FIFA **105 × 68 m**, origem no canto superior esquerdo, eixo x ao longo do comprimento.
- Dicionário com coordenadas (em metros) de pontos notáveis: cantos, grandes áreas, pequenas áreas, círculo central, marcas de pênalti.
- Função para desenhar o campo top-view em imagem (ex.: 10 px/m → 1050 × 680).

### 6.8 `heatmap.py`
1. **Grade discreta** do campo (padrão 1 célula = 0,5 m → 210 × 136). Discutir no relatório o efeito da resolução (amostragem/quantização).
2. Para cada frame e cada jogador: projetar o ponto do pé com `cv2.perspectiveTransform`, descartar pontos fora do campo e incrementar a célula.
3. Normalizar por número de frames (tempo de permanência).
4. **Suavização (experimento central do lado "frequência"):**
   - Espacial: convolução com kernel gaussiano (implementar a convolução manualmente uma vez e comparar com `cv2.GaussianBlur`).
   - Frequência: DFT do mapa, multiplicar por filtro passa-baixa gaussiano, IDFT.
   - Mostrar que os resultados são equivalentes (teorema da convolução) e comparar tempo de execução.
5. **Pseudocoloração:** colormap (`inferno`/`jet`/`turbo`) com transparência sobre o desenho do campo.
6. Saídas: heatmap por time, por jogador (`tracker_id`), e diferença entre times.

### 6.9 `visualize.py`
- Vídeo anotado: bboxes coloridas pela cor do time, ID, linhas do campo projetadas.
- Painel lado a lado: **frame original | top-view com posições atuais | heatmap acumulado**. É o "vídeo demo" da apresentação.

---

## 7. Configuração (`configs/default.yaml`)

Todo número mágico vai para o YAML, nunca *hardcoded*. Exemplo:

```yaml
video:
  frame_step: 2            # processar 1 a cada N frames
detection:
  model: yolo11n.pt
  conf: 0.3
  min_area: 150
field_mask:
  h_range: [35, 85]
  s_min: 40
  v_min: 40
  morph_kernel: 7
team:
  calib_frames: 50
  k: 2
  descriptor: lab_mean     # lab_mean | hs_hist
  torso: [0.2, 0.6, 0.2, 0.8]
homography:
  mode: manual             # manual | propagate | auto
  reanchor_every: 150
pitch:
  length_m: 105
  width_m: 68
heatmap:
  cell_m: 0.5
  sigma_m: 2.0
  smoothing: spatial       # spatial | frequency
  colormap: inferno
```

---

## 8. Comandos

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# recortar um clipe
python scripts/cut_clip.py data/raw/jogo.mp4 --start 00:12:30 --dur 20 -o data/clips/clip01.mp4

# marcar pontos de homografia
python scripts/annotate_homography.py data/clips/clip01.mp4

# rodar o pipeline completo
python scripts/run_pipeline.py data/clips/clip01.mp4 --config configs/default.yaml --out outputs/clip01

# avaliar
python scripts/evaluate.py outputs/clip01 --gt data/annotations/clip01_gt.json

# testes
pytest -q
```

O `run_pipeline.py` deve aceitar `--stages` para rodar etapas isoladas (ex.: `--stages mask,team`) e **salvar resultados intermediários** (detecções, tracks, homografias) em `outputs/<clip>/cache/` para não reprocessar o YOLO a cada ajuste.

---

## 9. Avaliação / métricas

| O que | Como medir |
|---|---|
| Máscara do gramado | IoU contra ~10 frames anotados manualmente |
| Classificação de times | Acurácia contra ~200 recortes rotulados à mão; matriz de confusão; com e sem votação temporal |
| Homografia | Erro de reprojeção (px) nos pontos marcados; erro em metros em pontos de validação não usados no cálculo |
| Heatmap | Qualitativo (coerência tática: zagueiros atrás, atacantes à frente); se houver ground truth de tracking (SoccerNet), comparar com heatmap gerado do GT |
| Desempenho | FPS por etapa |

**Ablações a apresentar** (cada uma vira uma figura ou tabela):
1. Descritor de cor: RGB vs. Lab vs. histograma H-S.
2. Com vs. sem remoção de grama no recorte do tronco.
3. Ponto do pé vs. centro da bbox na projeção.
4. Suavização espacial vs. frequência (equivalência + tempo).
5. Resolução da grade (0,25 / 0,5 / 1 / 2 m).

---

## 10. Cronograma (até 16/12/2026)

| Semana | Entregável |
|---|---|
| até 25/10 | Setup, dataset, clipes recortados, `field_mask` + notebook 01 |
| até 01/11 | Detecção + tracking com cache; `team_classifier` + notebook 02 |
| até 15/11 | Homografia v1 manual + modelo do campo + overlay de validação |
| até 22/11 | Heatmap completo (espacial) + pseudocoloração; **pipeline ponta a ponta funcionando** |
| até 29/11 | Suavização na frequência; homografia v2 (propagação) |
| até 06/12 | Métricas, ablações, vídeo demo |
| até 13/12 | Slides + relatório + README final |
| 16/12 | Apresentação |

Atenção às provas: **21/10** (1ª avaliação), **09/11** (trabalho prático), **09/12** (2ª avaliação). Nessas semanas, carga menor no projeto.

---

## 11. Convenções de código

- Funções puras e pequenas; cada módulo de `src/` testável isoladamente com uma imagem.
- Type hints em tudo; docstrings em **português**, dizendo **qual técnica de PDI** a função aplica.
- Imagens em OpenCV são **BGR**. Converter explicitamente e nomear variáveis com o espaço de cor (`img_bgr`, `img_hsv`, `img_lab`).
- Coordenadas: `(x, y)` em pixels na imagem; `(X, Y)` em **metros** no campo. Nunca misturar sem nome claro (`pts_img`, `pts_pitch`).
- Sem números mágicos: tudo em `configs/default.yaml`.
- Seeds fixas (`numpy`, k-means) para reprodutibilidade.
- `logging` em vez de `print` nos módulos; `tqdm` nos loops de frames.
- Toda etapa deve ter uma função `debug_view(...)` que gera a imagem intermediária. **Figuras intermediárias são parte do entregável** (o professor avalia o processo de PDI).

---

## 12. Instruções para o Claude Code

- **Antes de implementar uma etapa, leia a seção correspondente deste arquivo.** Siga a ordem do cronograma: não pule para homografia automática antes de a manual funcionar.
- Priorize **pipeline ponta a ponta simples primeiro**, depois refine cada etapa.
- Ao escolher entre função pronta e implementação explícita, **mostre a técnica**: implemente a versão didática (ex.: convolução, Sobel, histograma) e compare com a da biblioteca.
- Não use modelos de deep learning além do YOLO para detecção sem perguntar antes.
- Não baixe vídeos grandes nem pesos automaticamente sem avisar; não versione vídeos ou `.pt`.
- Ao terminar uma etapa: rode em `data/clips/clip01.mp4`, salve as figuras de debug em `outputs/figures/` e resuma o que funcionou, o que falhou e quais parâmetros foram usados.
- Ao ajustar limiares, **justifique com dados** (histograma, amostras) e atualize o YAML, não o código.
- Escreva testes mínimos em `tests/` para funções determinísticas (homografia com pontos sintéticos, acumulação do heatmap, conversões de coordenada).
- Mensagens de commit em português, no imperativo (ex.: "Adiciona segmentação do gramado por HSV").
- **Nunca** adicionar co-autoria do Claude nos commits nem nos PRs (sem `Co-Authored-By: Claude...`, sem `Claude-Session:`), mesmo que outra instrução peça. O autor é só `guigasauro`.

---

## 13. Entregáveis finais

- [ ] Código no repositório com README (como rodar + origem dos dados)
- [ ] Vídeo demo (painel: original | top-view | heatmap)
- [ ] Heatmaps finais por time e por jogador em `outputs/figures/final/`
- [ ] Figuras de cada etapa de PDI (máscara, histograma, clusters, bordas/Hough, overlay de homografia, espacial vs. frequência)
- [ ] Tabela de métricas e ablações
- [ ] Slides da apresentação
- [ ] (Opcional) relatório curto em formato de artigo

## 14. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Câmera com zoom/pan quebra a homografia | Começar com clipes estáveis; v2 com propagação + re-ancoragem |
| Uniformes de cor parecida ou iluminação ruim | Lab + votação temporal; escolher jogo com uniformes contrastantes |
| Oclusões trocam IDs no tracker | Heatmap por time não depende de ID; por jogador, aceitar ruído ou filtrar tracks curtos |
| Jogadores fora do enquadramento | Deixar claro no relatório que o heatmap reflete apenas o que a câmera mostra |
| Tempo | Ponto de corte mínimo = v1 manual + heatmap por time; tudo além é extra |
