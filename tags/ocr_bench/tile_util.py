"""Tiling helper for whole-sheet candidates: yields (rot, x0, y0, tile) over the sheet turned 0/90/180/270 and maps
tile points back to sheet px."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import cv2, numpy as np
ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
def tiles(g, T, ST, rots=(0,)):
    H, W = g.shape
    for r in rots:
        im = g if r == 0 else cv2.rotate(g, ROT[r]); h, w = im.shape
        for y0 in range(0, max(1, h - 64), ST):
            for x0 in range(0, max(1, w - 64), ST):
                t = im[y0:y0 + T, x0:x0 + T]
                if min(t.shape) < 64 or t.min() > 200: continue
                yield r, x0, y0, t, (H, W)
def back(pts, r, x0, y0, HW):
    H, W = HW; p = np.asarray(pts, float) + [x0, y0]; x, y = p[:, 0], p[:, 1]
    if r == 90: p = np.stack([y, H - 1 - x], 1)
    elif r == 180: p = np.stack([W - 1 - x, H - 1 - y], 1)
    elif r == 270: p = np.stack([W - 1 - y, x], 1)
    return p.tolist()
