"""Marcação manual da homografia (v1).

Uso:  python scripts/annotate_homography.py data/clips/clip01.mp4 [--frame 0]

Janela "frame": clique num ponto do campo na imagem (círculo amarelo).
Janela "campo": clique no ponto correspondente do modelo (é ajustado ao ponto notável mais próximo).
Repita para >= 4 pontos bem espalhados (não colineares). Com 4+, as linhas do modelo são projetadas
sobre o frame em magenta: devem coincidir com as linhas reais.
Teclas:  u = desfaz o último par   s = salva e sai   q/Esc = sai sem salvar
Saída: data/annotations/<clip>.json
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import homography, pitch  # noqa: E402

SNAP_M = 3.0      # distância máxima (m) do clique no modelo até um ponto notável
VIEW_W = 1280     # largura da janela do frame (a imagem é reduzida só para exibir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    pc, hc = cfg["pitch"], cfg["homography"]
    out = Path(args.out or f"data/annotations/{Path(args.video).stem}.json")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img_bgr = cap.read()
    if not ok:
        sys.exit(f"não consegui ler o frame {args.frame} de {args.video}")
    scale = VIEW_W / img_bgr.shape[1]
    kp = pitch.keypoints(pc["length_m"], pc["width_m"])
    lines = pitch.lines(pc["length_m"], pc["width_m"])
    pitch_ref = pitch.debug_view(pc)

    pts_img: list[tuple[float, float]] = []
    names: list[str] = []
    pending: list[tuple[float, float] | None] = [None]
    H: list[np.ndarray | None] = [None]

    def update_h() -> None:
        H[0] = None
        if len(pts_img) >= 4:
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
    print(__doc__)
    while True:
        P = np.array([kp[n] for n in names]) if names else None
        view = homography.debug_view(img_bgr, H[0], lines, np.array(pts_img), P) if H[0] is not None else img_bgr.copy()
        for (x, y), n in zip(pts_img, names):
            cv2.circle(view, (int(x), int(y)), 7, (0, 255, 0), 2)
            cv2.putText(view, n, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if pending[0] is not None:
            cv2.circle(view, (int(pending[0][0]), int(pending[0][1])), 9, (0, 255, 255), 3)
        cv2.putText(view, f"pares: {len(names)}  |  u=desfaz  s=salva  q=sai", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.imshow("frame", cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        ref = pitch_ref.copy()
        for n in names:
            x, y = np.round(pitch.to_px(kp[n], pc["px_per_m"], pc["margin_m"])).astype(int)
            cv2.circle(ref, (x, y), 8, (0, 255, 0), 2)
        cv2.imshow("campo", ref)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and names:
            pts_img.pop(); print("desfeito:", names.pop()); update_h()
        elif k == ord("s"):
            if H[0] is None:
                print("precisa de pelo menos 4 pares válidos para salvar")
                continue
            homography.save(out, H[0], np.array(pts_img), names, args.frame)
            cv2.imwrite(str(Path("outputs/figures") / f"05_homografia_{out.stem}.png"),
                        homography.debug_view(img_bgr, H[0], lines, np.array(pts_img), np.array([kp[n] for n in names])))
            print(f"salvo em {out} e outputs/figures/05_homografia_{out.stem}.png")
            break
        elif k in (ord("q"), 27):
            print("saindo sem salvar")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
