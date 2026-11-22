"""Word spotting by template matching (no OCR): one tag image as the query, normalized cross-correlation over the
whole sheet in 4 orientations; peaks above a threshold = copies of the tag.  Prototype on Malkoc K2.
usage: python esle_dene.py OPENING_ID [scale 0.5]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys, time
import cv2, numpy as np
from shapely.geometry import Point, Polygon
import test2
qid = sys.argv[1]; sc = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
E = json.load(open(test2.K / "el_etiket" / "malkoc_k2" / "ETIKET.json", encoding="utf-8")); s = E["px_per_m"]
g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / test2.FLOORS["malkoc_k2"] / "clean.png"), np.uint8), 0)
o = [x for x in E["openings"] if x["id"] == qid][0]; box = np.asarray(o.get("etiket_kutu") or o["poly"], float)   # window: the whole symbol
# level crop of the query tag
c = box; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e)); d = c[(k + 1) % 4] - c[k]
ang = np.degrees(np.arctan2(d[1], d[0])); m = c.mean(0); L, Hh = max(e) / 2 + 3, min(e) / 2 + 3
M = cv2.getRotationMatrix2D(tuple(m), ang, 1.0); M[:, 2] += [L - m[0], Hh - m[1]]; q = cv2.warpAffine(g, M, (int(2 * L), int(2 * Hh)), borderValue=255)
cv2.imwrite("sorgu.png", q)
G = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32); t0 = time.time(); peaks = []
for r in range(4):
    Q = cv2.resize(np.rot90(q, r).copy(), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32)
    R = cv2.matchTemplate(G, Q, cv2.TM_CCOEFF_NORMED); h, w = Q.shape
    ys, xs = np.where(R > 0.45)
    for y, x in sorted(zip(ys, xs), key=lambda p: -R[p]):
        if any(abs(y - py) < h * 0.6 and abs(x - px) < w * 0.6 for py, px, _, _, _ in peaks): continue
        peaks.append((y, x, float(R[y, x]), h, w))
print(len(peaks), "tepe", round(time.time() - t0, 1), "s")
# score: which door tags are hit (peak centre inside opening + 10 cm)
hit = {}; noise = 0
for y, x, v, h, w in peaks:
    cx, cy = (x + w / 2) / sc, (y + h / 2) / sc; ok = False
    for op in E["openings"]:
        if Polygon(op["poly"]).buffer(0.10 * s).contains(Point(cx, cy)): hit[op["id"]] = max(hit.get(op["id"], 0), v); ok = True
    noise += not ok
same = [op["id"] for op in E["openings"] if op["etiket"] == o["etiket"]]
print("sorgu", o["etiket"], "| ayni etiketli", len(same), "bulunan", sum(i in hit for i in same), "| baska aciklikta", sum(i not in same for i in hit), "| bosta", noise)
print(sorted([(round(v, 2), i, [op["etiket"] for op in E["openings"] if op["id"] == i][0]) for i, v in hit.items()], reverse=True))
