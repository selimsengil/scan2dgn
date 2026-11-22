"""Developer test of qwen_geo on the dev floors (never 015): seeds = a point inside each DGN room, scale from DGN areas.
Measures what the geometry alone can do when the semantic part (which label is where) is right.
usage: python geo_test.py [--floors 019,013,016] [--save]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

import qwen_geo as G  # noqa: E402
import v0_watershed as V0  # noqa: E402

B = V0.B
OUT = config.KT / "geo_test"


def corners_at(gt, pr, tol=10):
    gp = list(gt.exterior.coords)[:-1]
    return np.mean([min(np.hypot(x - a, y - b) for a, b in pr) <= tol for x, y in gp]) if pr else 0.0


def run(floor, save, **kw):
    gray = G.load_gray(B / floor / "clean.png")
    gts = [g for g in V0.load_rooms(floor) if g["poly"].geom_type == "Polygon"]
    sc = [np.sqrt(g["poly"].area / (g["area_m2"] * 1e4)) for g in gts if (g.get("area_m2") or -1) > 0]
    ppc = float(np.median(sc)) if sc else 0.8
    wall = G.wall_mask(gray, ppc, **kw)
    seeds = [(i, [tuple(map(int, g["poly"].representative_point().coords[0]))]) for i, g in enumerate(gts)]
    cm = G.cells(wall, seeds, px_per_cm=ppc)
    ious, cors = [], []
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis[wall > 0] = (vis[wall > 0] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
    for i, g in enumerate(gts):
        pts = G.mask_to_polygon(cm[i])
        if not pts or len(pts) < 3:
            ious.append(0); cors.append(0); continue
        P = Polygon(pts).buffer(0)
        iou = g["poly"].intersection(P).area / g["poly"].union(P).area
        ious.append(iou); cors.append(corners_at(g["poly"], pts))
        cv2.polylines(vis, [np.array(list(g["poly"].exterior.coords), np.int32)], True, (0, 170, 0), 3)
        cv2.polylines(vis, [np.array(pts, np.int32)], True, (200, 0, 200), 2)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / f"geo_{floor}.png"), vis)
    found = sum(v >= 0.5 for v in ious)
    print(f"{floor}: px/cm {ppc:.3f} rooms {len(gts)} found {found} meanIoU {np.mean(ious):.3f} "
          f"IoU>=.9 {sum(v >= .9 for v in ious)} corners@10 {np.mean([c for c, v in zip(cors, ious) if v >= .5] or [0]):.3f}", flush=True)
    return ious


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="019,013,016")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    for f in a.floors.split(","):
        run(f, a.save)
