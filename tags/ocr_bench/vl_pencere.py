"""PaddleOCR-VL OCR on window crops at higher scale (sheet px x3, cubic) and tighter margin."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys, time
from ortak import *
sys.argv = ["vl.py", "none"]; exec(open(H / "vl.py", encoding="utf-8").read().split('mode = sys.argv[1]')[0])
for sc, mg in ((3.0, 0.10), (2.0, 0.06)):
    res = {}; t0 = time.time()
    for oid, kind, tag, ims in crops(margin_m=mg, scale=sc):
        if kind != "window": continue
        res[oid] = [run(im, "ocr")[0] for im in ims]
    r = score_crops(res, f"VL pencere x{sc} m{mg}", {"sure_s": round(time.time() - t0, 1)})
    print(sc, mg, r["window"], r["sure_s"], flush=True)
    for x in r["satirlar"]:
        if x["kind"] == "window": print(" ", x["id"], x["etiket"], x["sonuc"], [t.replace("\n", " ")[:40] for t in x["okunan"]])
