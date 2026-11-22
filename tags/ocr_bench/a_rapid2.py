"""RapidOCR 4 orientations on the whole sheet (CPU).  usage: python a_rapid2.py KEY  (.venv)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import sys, time
import cv2, numpy as np
from rapidocr_onnxruntime import RapidOCR
import test2, tile_util as TU
key = sys.argv[1]; eng = RapidOCR(); g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / test2.FLOORS[key] / "clean.png"), np.uint8), 0); reads = []; t0 = time.time()
for r, x0, y0, t, HW in TU.tiles(g, 1600, 1400, (0, 90, 180, 270)):
    res, _ = eng(t)
    for box, txt, conf in (res or []): reads.append({"box": TU.back(np.asarray(box, float), r, x0, y0, HW), "text": txt})
test2.save("rapidocr_4yon", key, reads, time.time() - t0)
