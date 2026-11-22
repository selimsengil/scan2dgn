"""Two-stage candidates: a finder gives region boxes on the whole sheet, GLM-OCR reads each (level crop, 0/180).
usage: python iki_asama.py FINDER KEY [--mg 0.10]    FINDER: craft05 | craft03 | bizim_e1 | from:<regions.json>   (5070 Ti only)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import cv2, numpy as np
H = config.KT / "ocr_test"
import test2
ap = argparse.ArgumentParser(); ap.add_argument("finder"); ap.add_argument("key"); ap.add_argument("--mg", type=float, default=0.10); a = ap.parse_args()
fid = test2.FLOORS[a.key]; t0 = time.time()
if a.finder.startswith("craft"):
    import edocr2
    W8 = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\edocr2"; det = edocr2.Detector(os.path.join(W8, "craft_mlt_25k.pth"), device="cuda"); thr = {"craft05": 0.5, "craft03": 0.3}[a.finder]
    g = cv2.imdecode(np.fromfile(str(H.parent / "dgn_katlar" / fid / "clean.png"), np.uint8), 0); T = 1024; boxes = []
    for y0 in range(0, g.shape[0] - 64, T - 128):
        for x0 in range(0, g.shape[1] - 64, T - 128):
            tile = cv2.cvtColor(g[y0:y0 + T, x0:x0 + T], cv2.COLOR_GRAY2RGB)
            if min(tile.shape[:2]) < 64 or tile.min() > 200: continue
            for b in det.detect([tile], detection_threshold=thr, text_threshold=0.3 if thr < 0.6 else 0.4)[0]: boxes.append(np.asarray(b, float) + [x0, y0])
    keep = []
    for b in boxes:
        if any(np.hypot(*(b.mean(0) - k.mean(0))) < 8 for k in keep): continue
        keep.append(b)
    regs = [{"box": b.tolist()} for b in keep]
elif a.finder.startswith("bizim"):
    import etiket_bulucu as EB
    regs, _ = EB.regions(fid, a.finder.split("_", 1)[1])
else:
    regs = json.load(open(a.finder.split(":", 1)[1], encoding="utf-8"))
tf = time.time() - t0; print(a.finder, a.key, len(regs), "bolge", round(tf), "s", flush=True)
import etiket_oku as EO
res = EO.oku(fid, regs, a.mg, readers=("glm",))
reads = [{"box": r["box"], "text": r["text"]} for r in res]
test2.save(f"{a.finder}+glm", a.key, reads, time.time() - t0, {"bulma_s": round(tf, 1), "bolge": len(regs)})
