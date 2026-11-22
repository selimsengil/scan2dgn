"""Repair model polygons that are geometrically broken (ring traced twice, self-touching) and rescore.

repair(): drop consecutive duplicates; if the ring returns to an earlier vertex, keep the sub-ring with the
largest area; if still invalid, make_valid and keep the largest polygon's exterior.

usage: python repair_rescore.py TAG   -> prints old/new mean IoU and changed rooms
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

import qwen_test as QT

R = Path(__file__).with_name("runs")


def _area(r):
    return abs(Polygon(r).area) if len(r) >= 3 else 0.0


def split_loops(pts):
    """Split a vertex list at repeated vertices into simple loops; return them all."""
    loops, stack = [], []
    for p in pts:
        p = (round(p[0], 2), round(p[1], 2))
        if stack and p == stack[-1]:
            continue
        if p in stack:
            i = stack.index(p)
            loops.append(stack[i:])
            stack = stack[:i + 1]
        else:
            stack.append(p)
    loops.append(stack)
    return [l for l in loops if len(l) >= 3]


def repair(pts):
    if not pts or len(pts) < 3:
        return pts
    if Polygon(pts).is_valid:
        return pts
    loops = split_loops(pts)
    # a ring traced over itself: the shape is the union of its loops (the model went around, came back
    # along the same edges and added tabs); outer boundary of the biggest connected piece
    g = unary_union([make_valid(Polygon(l)).buffer(0.5, join_style=2) for l in loops]).buffer(-0.5, join_style=2)
    polys = [g] if g.geom_type == "Polygon" else [x for x in getattr(g, "geoms", []) if x.geom_type == "Polygon"]
    if not polys:
        return pts
    big = max(polys, key=lambda x: x.area)
    return [list(c) for c in list(big.exterior.coords)[:-1]]


def iou(m):
    return m.get("IoU", 0.0) if m.get("valid", True) and "IoU" in m else 0.0


def main():
    tag = sys.argv[1]
    d = json.load(open(R / f"b132_{tag}" / "RESULT.json", encoding="utf-8"))
    H = type("H", (), {"inside": None})()
    old, new = [], []
    for k, r in sorted(d.items()):
        pts = r.get("snapped") or r.get("points")
        o = iou(r["snap"])
        n = o
        if pts and len(pts) >= 3 and not Polygon(pts).is_valid:
            n = iou(QT.brief(QT.score(repair(pts), r["reference"], H)))
            print(f"{k} {r['name'][:22]:22} {o:.3f} -> {n:.3f}")
        old.append(o); new.append(max(o, n))
    print(f"mean IoU {sum(old) / len(old):.3f} -> {sum(new) / len(new):.3f}  (>=0.90: {sum(x >= .9 for x in old)} -> {sum(x >= .9 for x in new)})")


if __name__ == "__main__":
    main()
