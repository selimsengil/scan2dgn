"""The model draws roughly, the encoding makes it exact: every edge of the model's polygon is matched to the encoded
line it most plausibly meant (same direction, close, overlapping), and the corners become intersections of the matched
lines. The model decides the shape; coordinates come from the plan's own strokes.

align(points, lines) -> (polygon points, per-edge report)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import numpy as np
from shapely.geometry import Polygon


def _match(a, b, lines, reach, ang_tol):
    d = b - a; L = float(np.hypot(*d))
    if L < 1e-6: return None
    u = d / L; n = np.array([-u[1], u[0]]); mid = (a + b) / 2; ang = np.degrees(np.arctan2(u[1], u[0])) % 180.0
    best = None
    for k, l in enumerate(lines):
        da = abs(l["ang"] - ang); da = min(da, 180 - da)
        if da > ang_tol: continue
        lu = (l["p2"] - l["p1"]) / l["len"]; ln = np.array([-lu[1], lu[0]])
        dist = abs((mid - l["p1"]) @ ln)
        if dist > reach: continue
        ta, tb = sorted(((a - l["p1"]) @ lu, (b - l["p1"]) @ lu))
        ov = (min(tb, l["len"] + 8) - max(ta, -8)) / max(L, 1e-6)
        if ov < 0.35: continue
        score = min(ov, 1.0) - 0.6 * dist / reach - 0.01 * da + (0.1 if l["len"] >= 0.6 * L else 0.0)
        if best is None or score > best[0]: best = (score, k, dist)
    return best


def _cross(l1, l2):
    p, r = l1["p1"], l1["p2"] - l1["p1"]; q, s = l2["p1"], l2["p2"] - l2["p1"]
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 0.14 * np.hypot(*r) * np.hypot(*s): return None           # under ~8 degrees: no usable crossing
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    return p + t * r


def align(points, lines, reach=22.0, ang_tol=10.0, min_edge=7.0):
    P = [np.asarray(p, float) for p in points]; n = len(P)
    edges = []                                                      # each: a line (matched or the model's own)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]; L = float(np.hypot(*(b - a)))
        if L < min_edge: continue
        m = _match(a, b, lines, reach if L >= 25 else min(reach, 12.0), ang_tol if L >= 25 else 18.0)
        own = {"p1": a, "p2": b, "len": L, "ang": float(np.degrees(np.arctan2(*(b - a)[::-1])) % 180.0)}
        edges.append({"line": lines[m[1]] if m else own, "id": lines[m[1]]["id"] if m else None, "a": a, "b": b, "moved": m[2] if m else None})
    out = [edges[0]] if edges else []                               # consecutive edges on the same line are one edge
    for e in edges[1:]:
        if e["id"] is not None and e["id"] == out[-1]["id"]: out[-1]["b"] = e["b"]
        else: out.append(e)
    if len(out) > 1 and out[0]["id"] is not None and out[0]["id"] == out[-1]["id"]: out[0]["a"] = out[-1]["a"]; out.pop()
    if len(out) < 3: return [list(p) for p in P], out
    poly = []
    for i in range(len(out)):
        e0, e1 = out[i - 1], out[i]
        c = _cross(e0["line"], e1["line"])
        corner = (e0["b"] + e1["a"]) / 2                             # where the model put this corner
        if c is not None and np.hypot(*(c - corner)) <= 2.2 * reach: poly.append(c)
        else:                                                        # parallel neighbours: a short jog between the two lines
            for e in (e0, e1):
                l = e["line"]; lu = (l["p2"] - l["p1"]) / l["len"]
                poly.append(l["p1"] + lu * ((corner - l["p1"]) @ lu))
    Q = Polygon(poly)
    if not Q.is_valid or Q.area < 0.5 * Polygon(P).buffer(0).area: return [list(p) for p in P], out
    return [list(p) for p in poly], out
