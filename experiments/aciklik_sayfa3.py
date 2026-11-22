"""Result sheet of an opening measurement (aciklik_son.olc output): green = ready (matched, Citymax-valid, jamb <= 5 cm),
yellow = matched but jamb > 5 cm, red = matched but not valid, orange = not a labelled opening; labels thin grey.
usage: python aciklik_sayfa3.py KEY OLCUM.json OUT.jpg [scale 0.3]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(config.KT / "ocr_test"))
import test2
key, src, out = sys.argv[1:4]; sc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
d = json.load(open(src, encoding="utf-8")); G, s = test2.gt(key)
g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / test2.FLOORS[key] / "clean.png"), np.uint8), 0)
ys, xs = np.where(g < 128); y0, y1, x0, x1 = max(0, ys.min() - 50), ys.max() + 50, max(0, xs.min() - 50), xs.max() + 50
v = cv2.cvtColor(cv2.resize(g[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR); v = (v * 0.55 + 115).astype(np.uint8)
f = lambda p: np.round((np.asarray(p, float) - [x0, y0]) * sc).astype(np.int32)
for gg in G: cv2.polylines(v, [f(list(gg["poly"].exterior.coords))], True, (150, 150, 150), 1)
for r in d["satirlar"]:
    if r["eslesen"]: col = (0, 170, 0) if r["gecerli"] and r["kasa_cm"] <= 5 else ((0, 200, 230) if r["gecerli"] else (0, 0, 230))
    else: col = (0, 140, 255)
    cv2.polylines(v, [f(r["box"])], True, col, 2)
S = d["sonuc"]; txt = f"{key}  bulunan {S['bulunan']}  yanlis {S['yanlis']}  kasa med {S['kasa_med_cm']} cm  gecerli {S['gecerli_eslesen']}  hazir {S['hazir']}"
cv2.rectangle(v, (0, 0), (v.shape[1], 24), (255, 255, 255), -1); cv2.putText(v, txt, (6, 17), 0, 0.5, (0, 0, 0), 1)
cv2.imwrite(out, v, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, v.shape)
