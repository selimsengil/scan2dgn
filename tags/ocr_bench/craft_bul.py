"""CRAFT detector only: recall at the 40 tag places of Malkoc K2 for scale x threshold.  python craft_bul.py"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np, edocr2
from shapely.geometry import Polygon
from ortak import *
W8 = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\edocr2"; det = edocr2.Detector(os.path.join(W8, "craft_mlt_25k.pth"), device="cuda")
g0 = sheet()[400:10500, 400:5300]; E = gt(); s = E["px_per_m"]
for sc in (0.75, 1.0, 1.5):
    g = cv2.resize(g0, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC); T = 1024
    for thr in (0.7, 0.5, 0.3):
        t0 = time.time(); boxes = []
        for y0 in range(0, g.shape[0] - 64, T - 128):
            for x0 in range(0, g.shape[1] - 64, T - 128):
                tile = cv2.cvtColor(g[y0:y0 + T, x0:x0 + T], cv2.COLOR_GRAY2RGB)
                if min(tile.shape[:2]) < 64: continue
                for b in det.detect([tile], detection_threshold=thr, text_threshold=0.3 if thr < 0.6 else 0.4)[0]: boxes.append((np.asarray(b, float) + [x0, y0]) / sc + [400, 400])
        P = [Polygon(b).buffer(0) for b in boxes]; hit = {"door": 0, "window": 0}
        for o in E["openings"]:
            B = Polygon(o["poly"]).buffer(0.10 * s)
            if any(B.intersects(p) and p.intersection(B).area > 0.3 * p.area for p in P): hit[o["kind"]] += 1
        print(f"olcek {sc} esik {thr}: kutu {len(boxes)} kapi {hit['door']}/25 pencere {hit['window']}/15 {time.time()-t0:.0f}s", flush=True)
