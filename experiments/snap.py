"""Snap a polygon's edges onto the nearest parallel ink line (the wall face), corners re-intersected.

Each edge is moved along its normal by at most `max_px`; the offset kept is the one whose 1-px line
(middle 70% of the edge) is darkest, if clearly darker than where it is. Nothing else changes: the
model's decision (which wall, where it turns) stays, only its rounding is cleaned.
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import math

import numpy as np
from PIL import Image


def _line_dark(g: np.ndarray, a, b, frac=0.7, step=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    if L < 4:
        return 0.0
    t0, t1 = (1 - frac) / 2, 1 - (1 - frac) / 2
    n = max(4, int(L * frac / step))
    ts = np.linspace(t0, t1, n)
    pts = a[None] + (b - a)[None] * ts[:, None]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, g.shape[1] - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, g.shape[0] - 1)
    return float((g[ys, xs] < 140).mean())


def _intersect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], float)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, np.asarray(p2, float) - np.asarray(p1, float))
    return np.asarray(p1, float) + t[0] * np.asarray(d1, float)


def snap_polygon(img: Image.Image, poly, max_px=8, min_gain=0.15):
    g = np.asarray(img.convert("L"))
    n = len(poly)
    P = [np.asarray(p, float) for p in poly]
    lines = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        d = b - a; L = np.linalg.norm(d)
        if L < 1e-6:
            lines.append((a, np.array([1.0, 0.0]))); continue
        u = d / L; nrm = np.array([-u[1], u[0]])
        base = _line_dark(g, a, b)
        best, best_off = base, 0
        for off in range(-max_px, max_px + 1):
            v = _line_dark(g, a + nrm * off, b + nrm * off)
            if v > best + 1e-9 or (abs(v - best) < 1e-9 and abs(off) < abs(best_off)):
                best, best_off = v, off
        if best - base < min_gain and base > 0.5:
            best_off = 0   # already on a line
        lines.append((a + nrm * best_off, u))
    out = []
    for i in range(n):
        q = _intersect(lines[i - 1][0], lines[i - 1][1], lines[i][0], lines[i][1])
        out.append(q.tolist() if q is not None and np.linalg.norm(q - P[i]) < 4 * max_px else P[i].tolist())
    return out
