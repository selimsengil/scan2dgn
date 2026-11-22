"""ZONES: pieces that are only separated by faint drawn lines (floor tiles, dimension lines) are one zone. A tiled
kitchen is then one or two zones instead of fifty pieces. Long narrow pieces (wall bands, window bands) never join a
zone, and helper cuts (nothing drawn) are never crossed, so a zone ends at every wall line and at every opening.
Zones are only bigger words for the same translation; the exact outline still comes from the pieces inside them.

zones(fs, borders) -> list of lists of piece indices.    usage (loss check): python kod_bolge.py --floor 019"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, sys
from pathlib import Path

import numpy as np


def strip_like(f, width=21.0, min_long=0.0):
    """Long and narrow, like a wall or window band. min_long: shortest length that still counts (a cut-off floor
    tile is narrow too, but short)."""
    r = f.minimum_rotated_rectangle
    if r.geom_type != "Polygon": return True
    c = np.array(r.exterior.coords); e = sorted((float(np.hypot(*(c[1] - c[0]))), float(np.hypot(*(c[2] - c[1])))))
    if e[1] < min_long: return False
    return 2.0 * f.area / max(f.exterior.length, 1e-6) < 6.0 or (e[0] <= width and e[1] >= 2.5 * e[0])


def zones(fs, borders, faint=160, min_ink=0.5, min_len=6.0, cut_ink=None):
    """Union-find over faint drawn borders. cut_ink: if given, borders with ink share below it (helper cuts) also join."""
    par = list(range(len(fs))); strip = [strip_like(f) for f in fs]

    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for i, j, L, ink, g in borders:
        if L < min_len or strip[i] or strip[j]: continue
        if (ink >= min_ink and g >= faint) or (cut_ink is not None and ink < cut_ink):
            par[find(i)] = find(j)
    groups = {}
    for i in range(len(fs)): groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: -sum(fs[i].area for i in g))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    import v0_watershed as V0, cizgi_yuz as CY, kod_paket as KP, tek_oda_test as T
    from PIL import Image
    from shapely.ops import unary_union
    ap = argparse.ArgumentParser(); ap.add_argument("--floor", default="019"); a = ap.parse_args(); assert a.floor != "015"
    gray = np.array(Image.open(V0.B / a.floor / "clean.png").convert("L")); H, W = gray.shape
    enc, lines = KP.build(gray); fs = CY.faces(lines, (0, 0, W, H))
    fs = [f for f in fs if f.bounds[0] > 0.5 and f.bounds[1] > 0.5 and f.bounds[2] < W - 0.5 and f.bounds[3] < H - 0.5]
    bord = KP.border_info(fs, enc, gray)
    for name, kw in (("faint only", {}), ("faint + short helper cuts", {"cut_ink": 0.2})):
        zs = zones(fs, bord, **kw); zpoly = [unary_union([fs[i] for i in z]).buffer(0.01) for z in zs]
        print(f"== {name}: pieces {len(fs)} -> zones {len(zs)} (zones over 400 px2: {sum(p.area >= 400 for p in zpoly)})")
        ious = []; nz = []; rows = []
        for i, g in enumerate(V0.load_rooms(a.floor)):
            if g["poly"].geom_type != "Polygon": continue
            sel = [p for p in zpoly if p.intersects(g["poly"]) and p.intersection(g["poly"]).area >= 0.5 * p.area]
            if not sel: ious.append(0.0); nz.append(0); rows.append((i, g["name"], 0.0, 0)); continue
            U = unary_union(sel).buffer(0.01); U = U if U.geom_type == "Polygon" else max(U.geoms, key=lambda q: q.area)
            sc = T.score([list(p) for p in U.exterior.coords[:-1]], g["poly"]); ious.append(sc["iou"])
            big = sum(p.area >= 400 for p in sel); nz.append(big); rows.append((i, g["name"], sc["iou"], big))
        ious = np.array(ious)
        print(f"   best reachable mean IoU {ious.mean():.3f} | >=0.95: {(ious >= 0.95).sum()}/{len(ious)} | >=0.90: {(ious >= 0.9).sum()} | zones (over 400 px2) per room: median {int(np.median(nz))}, max {max(nz)}")
        print("   below 0.90:", [(f"r{i}", n, v, z) for i, n, v, z in rows if v < 0.9])
