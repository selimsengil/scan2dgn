"""docTR (db_resnet50 + PARSeq, rotated boxes) on 1024 px tiles of the whole sheet; reads = text lines (words joined).
usage: python e2e_doctr.py KEY  (.venv-ocr2, 5070 Ti)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import cv2, numpy as np, torch
from doctr.models import ocr_predictor
import test2, tile_util as TU
key = sys.argv[1]
m = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True, assume_straight_pages=False, detect_orientation=False, straighten_pages=False).cuda()
g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / test2.FLOORS[key] / "clean.png"), np.uint8), 0); reads = []; t0 = time.time(); n = 0
for r, x0, y0, t, HW in TU.tiles(g, 1024, 900, (0,)):
    h, w = t.shape; res = m([cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)])
    for b in res.pages[0].blocks:
        for l in b.lines:
            txt = " ".join(wd.value for wd in l.words)
            pts = np.vstack([np.asarray(wd.geometry, float).reshape(-1, 2) for wd in l.words]) * [w, h]
            rect = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32)))
            reads.append({"box": TU.back(rect, r, x0, y0, HW), "text": txt})
    n += 1
test2.save("doctr_db50_parseq", key, reads, time.time() - t0, {"parca": n})
