"""Alternative without training: CRAFT text detector (eDOCr2 / keras-ocr box decoding) on the whole Malkoc K2 sheet,
each box cut level and read by eDOCr2's dimension CRNN (alphabet 0-9 A a B C D R G H h M m n x Z t ( ) , . + - ± : / ° " ⌀ =).
usage: python craft_test.py [tile] [det_thr]   -> A_craft*.json   (5070 Ti only)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("EDOCR2_WEIGHTS_DIR", r"C:\Users\Selim\Documents\Proje\ocr_modelleri\edocr2")
import numpy as np, torch
import edocr2
from ortak import *
T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
W8 = os.environ["EDOCR2_WEIGHTS_DIR"]
det = edocr2.Detector(os.path.join(W8, "craft_mlt_25k.pth"), device="cuda"); rec = edocr2.Recognizer(os.path.join(W8, "recognizer_dimensions_2.keras"), device="cuda")
g = sheet(); H, W = g.shape; X0, Y0, X1, Y1 = 400, 400, 5300, 10500; boxes = []; t0 = time.time(); ST = T - 128
for y0 in range(Y0, Y1 - 64, ST):
    for x0 in range(X0, X1 - 64, ST):
        tile = cv2.cvtColor(g[y0:y0 + T, x0:x0 + T], cv2.COLOR_GRAY2RGB)
        if tile.shape[0] < 64 or tile.shape[1] < 64: continue
        bb = det.detect([tile], detection_threshold=thr)[0]
        for b in bb: boxes.append(np.asarray(b, float) + [x0, y0])
td = time.time() - t0
# dedupe overlapping boxes from tile overlaps
keep = []
for b in boxes:
    c = b.mean(0)
    if any(np.hypot(*(c - k.mean(0))) < 8 for k in keep): continue
    keep.append(b)
def level(b, pad=3):
    c = np.asarray(b, float); e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e))
    d = c[(k + 1) % 4] - c[k]; ang = np.degrees(np.arctan2(d[1], d[0])); m = c.mean(0); L = max(e) / 2 + pad; Hh = min(e) / 2 + pad
    M = cv2.getRotationMatrix2D(tuple(m), ang, 1.0); M[:, 2] += [L - m[0], Hh - m[1]]
    v = cv2.warpAffine(g, M, (max(4, int(2 * L)), max(4, int(2 * Hh))), borderValue=255); return [v, cv2.rotate(v, cv2.ROTATE_180)]
reads = []; t1 = time.time()
for b in keep:
    for im in level(b):
        t = rec.recognize(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
        if t: reads.append({"box": b.tolist(), "text": t})
tr = time.time() - t1
json.dump({"kutular": [b.tolist() for b in keep], "okumalar": reads}, open(H_ := H, "w") if False else open(config.KT / "ocr_test" / f"A_craft_T{T}_okumalar.json", "w", encoding="utf-8"), ensure_ascii=False)
r = score_spot(reads, f"CRAFT+eDOCr2 CRNN T{T}"); r["sure_s"] = round(time.time() - t0); r["bulma_s"] = round(td, 1); r["okuma_s"] = round(tr, 1); r["kutu"] = len(keep)
json.dump(r, open(config.KT / "ocr_test" / f"A_craft_T{T}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: r[k] for k in ("door", "window", "kutu", "bulma_s", "okuma_s")})
# detector-only recall at the tag places
E = gt(); s = E["px_per_m"]; from shapely.geometry import Polygon as _P
hit = {"door": 0, "window": 0}
for o in E["openings"]:
    B = _P(o["poly"]).buffer(0.10 * s)
    if any(B.intersects(_P(b).buffer(0)) and _P(b).buffer(0).intersection(B).area > 0.3 * _P(b).buffer(0).area for b in keep): hit[o["kind"]] += 1
print("bulucu, etiket yerinde kutu:", hit)
