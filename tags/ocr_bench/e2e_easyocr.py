"""EasyOCR (CRAFT + CRNN) on 1024 px tiles, characters limited to digits / x X P K - and rotation 90/180/270 tried.
usage: python e2e_easyocr.py KEY  (.venv-spec, 5070 Ti)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import cv2, numpy as np, easyocr
import test2, tile_util as TU
key = sys.argv[1]; rd = easyocr.Reader(["en"], gpu=True, model_storage_directory=r"C:\Users\Selim\Documents\Proje\ocr_modelleri\easyocr")
g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / test2.FLOORS[key] / "clean.png"), np.uint8), 0); reads = []; t0 = time.time(); n = 0
for r, x0, y0, t, HW in TU.tiles(g, 1024, 900, (0,)):
    for box, txt, conf in rd.readtext(t, allowlist="0123456789/xXPK- ", rotation_info=[90, 180, 270], width_ths=0.8, low_text=0.3):
        reads.append({"box": TU.back(np.asarray(box, float), r, x0, y0, HW), "text": txt})
    n += 1
test2.save("easyocr_rakam", key, reads, time.time() - t0, {"parca": n})
