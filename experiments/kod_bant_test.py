"""Offline check of a growth rule that needs no line darkness (dev floors, never 015): a room grows from the piece
under its label through EVERY drawn line and helper cut, and stops only at BAND pieces (long narrow pieces: wall and
window bands) and FILLED pieces (hatched columns) - and at doorways, which the model has to mark. Doors come from the
labels here, standing in for a model that found the doors first.
usage: python kod_bant_test.py --floor 019 [--no-doors]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, shape
from shapely.ops import unary_union

import v0_watershed as V0, cizgi_yuz as CY, kod_paket as KP, kod_bolge as KB, tek_oda_test as T


def piece_kinds(fs, gray, strip_w=21.0, fill=0.22, min_long=45.0):
    """'bant' (long narrow), 'dolu' (hatched/filled block) or 'zemin' (anything else) for every piece."""
    ink = (gray < 150).astype(np.uint8); out = []
    for f in fs:
        if KB.strip_like(f, strip_w, min_long): out.append("bant"); continue
        x0, y0, x1, y1 = [int(v) for v in f.bounds]
        if f.area < 6000 and (x1 - x0) > 3 and (y1 - y0) > 3:
            m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
            cv2.fillPoly(m, [(np.array(f.exterior.coords) - [x0, y0]).round().astype(np.int32)], 1)
            m = cv2.erode(m, np.ones((5, 5), np.uint8))               # stay off the outline itself
            if m.sum() >= 30 and ink[y0:y1 + 1, x0:x1 + 1][m > 0].mean() >= fill: out.append("dolu"); continue
        out.append("zemin")
    return out


def grow(seed, fs, adj, kind, barriers, min_len=5.0, max_area=None):
    seen = {seed}; q = [seed]; area = fs[seed].area
    while q:
        c = q.pop()
        for j, L in adj.get(c, []):
            if j in seen or L < min_len or kind[j] != "zemin": continue
            if barriers:
                sh = fs[c].exterior.intersection(fs[j].exterior)
                if any(sh.intersects(b) for b in barriers): continue
            if max_area and area + fs[j].area > max_area: continue
            seen.add(j); q.append(j); area += fs[j].area
    return seen


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floor", default="019"); ap.add_argument("--no-doors", action="store_true"); ap.add_argument("--min-long", type=float, default=45); ap.add_argument("--fill", type=float, default=0.22); a = ap.parse_args(); assert a.floor != "015"
    gray = np.array(Image.open(V0.B / a.floor / "clean.png").convert("L")); H, W = gray.shape
    enc, lines = KP.build(gray); fs = CY.faces(lines, (0, 0, W, H))
    fs = [f for f in fs if f.bounds[0] > 0.5 and f.bounds[1] > 0.5 and f.bounds[2] < W - 0.5 and f.bounds[3] < H - 0.5]
    bord = KP.border_info(fs, enc, gray); adj = {}
    for i, j, L, ink, g in bord: adj.setdefault(i, []).append((j, L)); adj.setdefault(j, []).append((i, L))
    kind = piece_kinds(fs, gray, fill=a.fill, min_long=a.min_long)
    print("pieces", len(fs), {k: kind.count(k) for k in ("zemin", "bant", "dolu")})
    lab = json.load(open(V0.B / a.floor / "LABELS_COMMON.json", encoding="utf-8"))
    doors = [] if a.no_doors else [shape(s["geometry"]).buffer(4.0) for s in lab["structures"] if s["kind"] == "door"]
    print("door barriers:", len(doors))
    from shapely.strtree import STRtree
    tree = STRtree(fs); ious = []; rows = []
    for i, g in enumerate(V0.load_rooms(a.floor)):
        if g["poly"].geom_type != "Polygon": continue
        pt = Point(*g["xy"]); hit = [int(j) for j in tree.query(pt) if fs[int(j)].contains(pt)]
        if not hit: ious.append(0.0); rows.append((i, g["name"], 0.0, "label on a line")); continue
        seed = min(hit, key=lambda j: fs[j].area)
        ids = grow(seed, fs, adj, kind, doors)
        U = unary_union([fs[j] for j in ids]).buffer(0.01); U = U if U.geom_type == "Polygon" else max(U.geoms, key=lambda q: q.area)
        sc = T.score([list(p) for p in U.exterior.coords[:-1]], g["poly"]); ious.append(sc["iou"]); rows.append((i, g["name"], sc["iou"], f"{len(ids)} pieces, {U.area / g['poly'].area:.2f}x area"))
    ious = np.array(ious)
    print(f"grow from label, stop at bands/filled/doors: mean IoU {ious.mean():.3f} | >=0.95: {(ious >= 0.95).sum()}/{len(ious)} | >=0.90: {(ious >= 0.9).sum()} | <0.5: {(ious < 0.5).sum()}")
    for r in rows:
        if r[2] < 0.9: print("   ", f"r{r[0]}", r[1], r[2], r[3])
