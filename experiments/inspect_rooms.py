"""Close-ups of the user's DGN room boundaries on the original raster, at native pixels.

usage: python inspect_rooms.py FLOOR NAME_SUBSTR [index] [--pad 0.6]
writes out_inspect/<floor>_<i>.png: the room (thin red line, vertices as dots) on the raster
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import sys
from pathlib import Path

import cv2
import numpy as np

import v0_watershed as V0


def main():
    f, want = sys.argv[1], sys.argv[2].upper()
    pick = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rooms = [r for r in V0.load_rooms(f) if want in (r["name"] or "").upper()]
    r = rooms[pick]
    img = cv2.imread(str(V0.B / f / "clean.png"), cv2.IMREAD_GRAYSCALE)
    s = V0.oracle_scale(V0.load_rooms(f))
    x0, y0, x1, y1 = [int(v) for v in r["poly"].bounds]
    pad = int(0.6 * s)
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    vis = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    up = max(1.0, 1100 / max(vis.shape[:2]))
    vis = cv2.resize(vis, None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
    pts = np.asarray(r["poly"].exterior.coords)
    q = np.round((pts - [x0, y0]) * up).astype(np.int32)
    cv2.polylines(vis, [q], True, (0, 0, 255), 1)
    for p in q[:-1]:
        cv2.circle(vis, tuple(int(v) for v in p), 3, (0, 160, 0), -1)
    out = Path(__file__).with_name("out_inspect"); out.mkdir(exist_ok=True)
    path = out / f"{f}_{want}_{pick}.png"
    cv2.imwrite(str(path), vis)
    print(path, r["name"], r["area_m2"], "vertices", len(pts) - 1, "scale", round(s, 1), "upscale", round(up, 2))


if __name__ == "__main__":
    main()
