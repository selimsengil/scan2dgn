"""Edge-level loss of the line encoder (dev floors, never 015): for every GT room edge, is there an encoded stroke
lying on the same line? A model that picks strokes gets its corners as intersections of the picked lines, so this is
the number that matters. usage: python cizgi_kenar_olc.py --floors 019,017,013"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image

import v0_watershed as V0  # noqa: E402
import cizgi_kod as CK  # noqa: E402


def edge_support(enc, a, b, tol=2.5, near=40, min_len=10):
    """(covered fraction of edge a-b by collinear strokes, longest collinear stroke within `near` px of the edge span)."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = b - a; L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
    s = enc["seg"]; p = s[:, :2] - a; q = s[:, 2:] - a
    ok = (np.abs(p @ n) <= tol) & (np.abs(q @ n) <= tol)
    t0 = np.minimum(p @ u, q @ u)[ok]; t1 = np.maximum(p @ u, q @ u)[ok]
    if not len(t0): return 0.0, 0.0
    nearby = (t1 >= -near) & (t0 <= L + near)
    longest = float((t1 - t0)[nearby].max()) if nearby.any() else 0.0
    cov = np.zeros(int(np.ceil(L)) + 1, bool)
    for x0, x1 in zip(np.clip(t0, 0, L), np.clip(t1, 0, L)):
        if x1 > x0: cov[int(x0):int(np.ceil(x1)) + 1] = True
    return float(cov.mean()), longest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floors", default="019,017,013"); ap.add_argument("--dark", type=int, default=150); ap.add_argument("--faint", type=int, default=200)
    a = ap.parse_args()
    for fl in a.floors.split(","):
        assert fl != "015"
        gray = np.array(Image.open(V0.B / fl / "clean.png").convert("L")); enc = CK.encode(gray, dark=a.dark, faint=a.faint)
        tot = sup = 0; totL = covL = 0.0; bad = []
        for i, g in enumerate(q for q in V0.load_rooms(fl) if q["poly"].geom_type == "Polygon"):
            c = np.array(g["poly"].exterior.coords)
            for k in range(len(c) - 1):
                L = float(np.hypot(*(c[k + 1] - c[k])))
                if L < 8: continue
                cov, longest = edge_support(enc, c[k], c[k + 1])
                tot += 1; totL += L; covL += cov * L
                if longest >= 10: sup += 1
                else: bad.append((i, g["name"], k, round(L), round(cov, 2), round(longest)))
        print(f"{fl}: GT edges {tot} | edges with a collinear stroke {sup} ({sup / tot:.1%}) | boundary length lying on strokes {covL / totL:.1%}")
        for b in bad[:8]: print("    no line:", b)
