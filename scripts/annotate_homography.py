"""Marcação manual da homografia.

Um vídeo (v1):     python scripts/annotate_homography.py data/clips/clip01.mp4 [--frame 0]
Partida completa:  python scripts/annotate_homography.py --match data/annotations/barcelona-x-real_madrid.json
                   (rode antes scripts/segment_match.py; percorre as tomadas da câmera principal, da mais
                   longa para a mais curta, e retoma de onde parou)

Janela "frame": clique num ponto do campo na imagem (círculo amarelo).
Janela "campo": clique no ponto correspondente do modelo (é ajustado ao ponto notável mais próximo).
Repita para >= 4 pontos bem espalhados (não colineares). Com 4+, as linhas do modelo são projetadas
sobre o frame em magenta: devem coincidir com as linhas reais.
Modo partida: depois de algumas marcações, a posição da câmera da transmissão é estimada e a marcação
usa o modelo de câmera fixa (src/camera.py, 4 parâmetros): bastam 3 pontos, mesmo colineares.
Teclas:  u = desfaz o último par   s = salva   q/Esc = sai   h = mostra/esconde as linhas projetadas
         a / d = volta / avança 1 s (escolha um quadro com linhas visíveis; os pares são descartados)
         n = pula a tomada (só no modo partida; ex.: nenhuma marcação do campo visível)
         z = desfaz a última tomada salva ou pulada e volta a ela (só no modo partida)
Saída: data/annotations/<clip>.json  ou  data/annotations/<partida>_homografias.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import camera, homography, pitch  # noqa: E402

SNAP_M = 3.0      # distância máxima (m) do clique no modelo até um ponto notável
VIEW_W = 1280     # largura da janela do frame (a imagem é só reescalada para exibir)
KEYS = {ord("s"): "save", ord("n"): "skip", ord("a"): "back", ord("d"): "fwd", ord("q"): "quit", 27: "quit", ord("z"): "undo"}


def annotate(img_bgr: np.ndarray, cfg: dict, header: str, allow_skip: bool,
             cam: tuple[np.ndarray, list[np.ndarray]] | None = None) -> tuple[str, np.ndarray | None, list, list]:
    """Janela de marcação de um quadro. Devolve (ação, H, pontos na imagem, nomes dos pontos do modelo);
    ação ∈ save | skip | back | fwd | quit. Com `cam` = (centro da câmera, câmeras de outras marcações),
    H vem do modelo de câmera fixa a partir de 3 pontos; sem, da homografia geral com 4+."""
    pc, hc = cfg["pitch"], cfg["homography"]
    scale = VIEW_W / img_bgr.shape[1]
    kp = pitch.keypoints(pc["length_m"], pc["width_m"])
    lines = pitch.lines(pc["length_m"], pc["width_m"])
    pitch_ref = pitch.debug_view(pc)
    pts_img: list[tuple[float, float]] = []
    names: list[str] = []
    pending: list[tuple[float, float] | None] = [None]
    H: list[np.ndarray | None] = [None]
    overlay = True

    def update_h() -> None:
        H[0] = None
        if cam is not None and len(pts_img) >= 3:
            P, I = np.array([kp[n] for n in names]), np.array(pts_img)
            h, w = img_bgr.shape[:2]
            try:
                H8 = homography.compute_homography(I, P, hc["ransac_thresh_m"])[0] if len(I) >= 4 else None
            except ValueError:
                H8 = None                                    # pontos colineares: só a câmera fixa resolve
            p, _ = camera.anchor(I, P, H8, cam[0], w, h, cam[1])
            if p is None:
                print("câmera não ajustou: confira os pares")
                return
            H[0] = camera.to_homography(p, cam[0], w, h)
            px = np.linalg.norm(camera.project(p, cam[0], w, h, P) - I, axis=1)
            print("erro por ponto (px):", ", ".join(f"{n}={e:.1f}{' [ERRO?]' if e > 10 else ''}" for n, e in zip(names, px)))
        elif cam is None and len(pts_img) >= 4:
            P = np.array([kp[n] for n in names])
            try:
                H[0], inl = homography.compute_homography(np.array(pts_img), P, hc["ransac_thresh_m"])
                err = homography.reprojection_error(H[0], np.array(pts_img), P)
                print("erro por ponto (m):", ", ".join(f"{n}={e:.2f}{'' if i else ' [OUTLIER]'}" for n, e, i in zip(names, err, inl)))
            except ValueError as e:
                print("homografia:", e)

    def on_frame(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending[0] = (x / scale, y / scale)

    def on_pitch(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if pending[0] is None:
            print("clique primeiro no frame")
            return
        XY = (x / pc["px_per_m"] - pc["margin_m"], y / pc["px_per_m"] - pc["margin_m"])
        name = pitch.nearest_keypoint(kp, XY, SNAP_M)
        if name is None or name in names:
            print("nenhum ponto notável livre perto do clique" if name is None else f"{name} já foi usado")
            return
        pts_img.append(pending[0]); names.append(name); pending[0] = None
        print(f"par {len(names)}: {name} <- ({pts_img[-1][0]:.0f}, {pts_img[-1][1]:.0f})")
        update_h()

    cv2.namedWindow("frame"); cv2.setMouseCallback("frame", on_frame)
    cv2.namedWindow("campo"); cv2.setMouseCallback("campo", on_pitch)
    help_txt = "u=desfaz s=salva h=linhas a/d=-/+1s " + ("n=pula z=desfaz tomada " if allow_skip else "") + "q=sai"
    while True:
        P = np.array([kp[n] for n in names]) if names else None
        view = homography.debug_view(img_bgr, H[0], lines, np.array(pts_img), P) if H[0] is not None and overlay else img_bgr.copy()
        for (x, y), n in zip(pts_img, names):
            cv2.circle(view, (int(x), int(y)), 7, (0, 255, 0), 2)
            cv2.putText(view, n, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if pending[0] is not None:
            cv2.circle(view, (int(pending[0][0]), int(pending[0][1])), 9, (0, 255, 255), 3)
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for i, txt in enumerate((header, f"pares: {len(names)} | {help_txt}")):
            cv2.putText(view, txt, (15, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(view, txt, (15, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("frame", view)
        ref = pitch_ref.copy()
        for n in names:
            x, y = np.round(pitch.to_px(kp[n], pc["px_per_m"], pc["margin_m"])).astype(int)
            cv2.circle(ref, (x, y), 8, (0, 255, 0), 2)
        cv2.imshow("campo", ref)
        k = cv2.waitKey(30) & 0xFF
        action = KEYS.get(k)
        if k == ord("u") and names:
            pts_img.pop(); print("desfeito:", names.pop()); update_h()
        elif k == ord("h"):
            overlay = not overlay
        elif action == "save" and H[0] is None:
            print(f"precisa de pelo menos {3 if cam else 4} pares válidos para salvar")
        elif action in ("skip", "undo") and not allow_skip:
            continue
        elif action:
            return action, H[0], pts_img, names


def read_frame(video: str, frame: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    return img if ok else None


def single(args: argparse.Namespace, cfg: dict) -> None:
    out = Path(args.out or f"data/annotations/{Path(args.video).stem}.json")
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    f = args.frame
    while True:
        img = read_frame(args.video, f)
        if img is None:
            sys.exit(f"não consegui ler o frame {f} de {args.video}")
        action, H, pts, names = annotate(img, cfg, f"{Path(args.video).name} | quadro {f}", allow_skip=False)
        if action in ("back", "fwd"):
            f = max(0, f + int(fps) * (1 if action == "fwd" else -1))
            continue
        if action == "save":
            homography.save(out, H, np.array(pts), names, f)
            kp = pitch.keypoints(cfg["pitch"]["length_m"], cfg["pitch"]["width_m"])
            cv2.imwrite(str(Path("outputs/figures") / f"05_homografia_{out.stem}.png"),
                        homography.debug_view(img, H, pitch.lines(cfg["pitch"]["length_m"], cfg["pitch"]["width_m"]),
                                              np.array(pts), np.array([kp[n] for n in names])))
            print(f"salvo em {out}")
        break


def match(args: argparse.Namespace, cfg: dict) -> None:
    M = json.load(open(args.match))
    cfg = pitch.for_match(cfg, M)
    name = Path(args.match).stem
    out = Path(f"data/annotations/{name}_homografias.json")
    done = json.loads(out.read_text()) if out.exists() else {}
    cam = match_camera(done, M, cfg)
    todo = []
    for half, v in M["tempos"].items():
        for s in json.load(open(f"outputs/{name}/{half}/shots.json")):
            if s["main"]:
                todo.append((s["frame_b"] - s["frame_a"], half, v["video"], s["frame_a"], s["frame_b"]))
    todo.sort(reverse=True)
    total = sum(t[0] for t in todo)
    i = 0
    while i < len(todo):
        n, half, video, a, b = todo[i]
        key = f"{half}:{a}"
        if key in done:
            i += 1
            continue
        fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
        f = a
        while True:
            covered = sum(t[0] for t in todo if "H_img_to_pitch" in done.get(f"{t[1]}:{t[3]}", {}))
            header = (f"{half}o tempo | tomada {i + 1}/{len(todo)} ({n / fps:.0f} s) | +{(f - a) / fps:.0f} s | "
                      f"marcado {covered / total:.0%} do tempo")
            action, H, pts, names = annotate(read_frame(video, f), cfg, header, allow_skip=True, cam=cam)
            if action in ("back", "fwd"):
                f = int(np.clip(f + int(fps) * (1 if action == "fwd" else -1), a, b - 1))
                continue
            break
        if action == "quit":
            break
        if action == "undo":
            if not done:
                print("nada para desfazer")
                continue
            last = list(done)[-1]                      # o JSON guarda as tomadas na ordem em que foram salvas
            done.pop(last)
            out.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            print(f"tomada {last} desfeita ({out})")
            i = next((j for j, t in enumerate(todo) if f"{t[1]}:{t[3]}" == last), i)
            continue
        done[key] = ({"tempo": half, "frame_a": a, "frame_b": b, "frame": f, "H_img_to_pitch": H.tolist(),
                      "modelo": "ptz" if cam else "homografia",
                      "points": [{"name": nm, "img": [float(x), float(y)]} for nm, (x, y) in zip(names, pts)]}
                     if action == "save" else {"tempo": half, "frame_a": a, "frame_b": b, "pulada": True})
        out.write_text(json.dumps(done, indent=1, ensure_ascii=False))
        print(f"tomada {key}: {'salva' if action == 'save' else 'pulada'} ({out})")
    cv2.destroyAllWindows()


def match_camera(done: dict, M: dict, cfg: dict) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Centro da câmera e câmeras candidatas (`camera.from_marks`). None enquanto houver poucas
    marcações com homografia geral para o chute inicial do centro."""
    lc, pc = cfg["lines"], cfg["pitch"]
    marks = [s for s in done.values() if "H_img_to_pitch" in s]
    cap = cv2.VideoCapture(M["tempos"]["1"]["video"])
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = sum(camera.decompose(np.array(s["H_img_to_pitch"]), w, h)[2] <= lc["center_max_resid"]
            for s in marks if s.get("modelo", "homografia") == "homografia")
    if n < lc["center_min_marks"]:
        print(f"{n} marcações boas: usando a homografia geral (4+ pontos) até ter {lc['center_min_marks']}")
        return None
    C, P = camera.from_marks(marks, pitch.keypoints(pc["length_m"], pc["width_m"]), w, h, lc["center_max_resid"])
    print(f"câmera fixa em {np.round(C, 1)} m (ajuste conjunto de {len(P)} marcações): 3+ pontos bastam")
    return C, P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--match", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.match:
        match(args, cfg)
    elif args.video:
        single(args, cfg)
    else:
        ap.error("informe um vídeo ou --match")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
