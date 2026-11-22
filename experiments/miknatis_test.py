"""Dev test of the qwen_cizim magnet (never 015): GT rooms with every edge shifted by a random reading error,
then snapped. Compares IoU and corners@10 vs GT before / after.
usage: python miknatis_test.py --floors "019,013" --err 5 --snap 6"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

import qwen_cizim as C  # noqa: E402
import qwen_geo as G  # noqa: E402
import v0_watershed as V0  # noqa: E402


def iou(a, b):
    a, b = Polygon(a).buffer(0), Polygon(b).buffer(0)
    return a.intersection(b).area / a.union(b).area


def corners(gt, pr, tol=10):
    return np.mean([min(np.hypot(x - u, y - v) for u, v in pr) <= tol for x, y in gt])


ap = argparse.ArgumentParser()
ap.add_argument("--floors", default="019")
ap.add_argument("--err", type=int, default=5)
ap.add_argument("--snap", type=int, default=6)
a = ap.parse_args()
rng = np.random.default_rng(0)
for f in a.floors.split(","):
    gray = G.load_gray(V0.B / f / "clean.png")
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = gray < min(max(t, 120), 200)
    res = {"exact": [], "noisy": [], "snapped": [], "exact_snapped": []}
    for g in V0.load_rooms(f):
        if g["poly"].geom_type != "Polygon":
            continue
        gt = [[int(round(x)), int(round(y))] for x, y in list(g["poly"].exterior.coords)[:-1]]
        P = G.orthogonalize([p[:] for p in gt])
        if len(P) < 3:
            continue
        n = len(P)
        Q = [p[:] for p in P]
        for i in range(n):   # shift every axis-aligned edge by a random error
            A, B = Q[i], Q[(i + 1) % n]
            e = int(rng.integers(-a.err, a.err + 1))
            if A[1] == B[1]:
                A[1] += e; B[1] = A[1]
            elif A[0] == B[0]:
                A[0] += e; B[0] = A[0]
        try:
            S, _ = C.snap_polygon(Q, ink, a.snap)
            E, _ = C.snap_polygon(P, ink, a.snap)
            for k, poly in (("exact", P), ("noisy", Q), ("snapped", S), ("exact_snapped", E)):
                res[k].append((iou(gt, poly), corners(gt, poly)))
        except Exception as ex:
            print("  skip", g.get("name"), ex)
    print(f"{f} ({len(res['exact'])} rooms, error +-{a.err} px, snap {a.snap}):")
    for k, v in res.items():
        v = np.array(v)
        print(f"   {k:14s} IoU {v[:, 0].mean():.3f}  corners@10 {v[:, 1].mean():.3f}  corners exact-ish@4 n/a")
