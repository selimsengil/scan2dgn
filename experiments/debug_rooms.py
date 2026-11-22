"""Per-room debug mosaic: DGN room (red), predicted region (blue), OCR seeds (green dots)."""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import sys
from pathlib import Path

import cv2
import numpy as np

import araclar as A
import v0_watershed as V0


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else "015"
    worst = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    k = A.Kat(f)
    k.segment()
    sc = k.score()
    gts = V0.load_rooms(f)
    order = np.argsort([r["iou"] for r in sc["per_room"]])[:worst]
    tiles = []
    for j in order:
        g, r = gts[j], sc["per_room"][j]
        x0, y0, x1, y1 = [int(v) for v in g["poly"].bounds]
        m = int(0.8 * k.s)
        x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(k.W, x1 + m), min(k.H, y1 + m)
        vis = cv2.cvtColor(k.img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        gm = V0.poly_mask(g["poly"], (k.H, k.W))[y0:y1, x0:x1]
        inside = [i for i, l in enumerate(k.labels) if gm.shape and V0.poly_mask(g["poly"], (k.H, k.W))[min(int(l["xy"][1]), k.H - 1), min(int(l["xy"][0]), k.W - 1)]]
        if inside:
            pm = k.room_mask(inside[0])[y0:y1, x0:x1]
            vis[pm] = (vis[pm] * 0.5 + np.array([255, 120, 0]) * 0.5).astype(np.uint8)
        wm = k.walls()[y0:y1, x0:x1]
        vis[wm] = (vis[wm] * 0.6 + np.array([90, 90, 90]) * 0.4).astype(np.uint8)
        cnt = np.round(np.asarray(g["poly"].exterior.coords) - [x0, y0]).astype(np.int32)
        cv2.polylines(vis, [cnt], True, (0, 0, 255), 2)
        for l in k.labels:
            if x0 <= l["xy"][0] < x1 and y0 <= l["xy"][1] < y1:
                cv2.circle(vis, (int(l["xy"][0] - x0), int(l["xy"][1] - y0)), 5, (0, 170, 0), -1)
        t = 360 / max(vis.shape[:2])
        vis = cv2.resize(vis, None, fx=t, fy=t, interpolation=cv2.INTER_AREA)
        canvas = np.full((390, 360, 3), 255, np.uint8)
        canvas[:vis.shape[0], :vis.shape[1]] = vis
        cv2.putText(canvas, f"{g['name']} iou {r['iou']:.2f} {r.get('why','')}", (4, 384), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    out = Path(__file__).with_name("out_araclar") / f"{f}_debug_worst.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(out)


if __name__ == "__main__":
    main()
