"""Upper bound of the face formulation (no model): choose every face that lies mostly inside the GT room and score
the union. Shows how much the cutting into pieces can lose. usage: python yuz_oracle.py [--rooms 11,18,7,21,14 | --all]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.ops import unary_union
import v0_watershed as V0, cizgi_kod as CK, cizgi_hat as CH, cizgi_yuz as CY, tek_oda_test as T

ap = argparse.ArgumentParser(); ap.add_argument("--rooms", default="11,18,7,21,14"); ap.add_argument("--all", action="store_true"); ap.add_argument("--floor", default="019")
ap.add_argument("--min-len", type=float, default=8); ap.add_argument("--reach", type=float, default=10)
a = ap.parse_args(); assert a.floor != "015"
im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); W, H = im.size
enc = CK.encode(gray); gts = V0.load_rooms(a.floor)
ids = [i for i, g in enumerate(gts) if g["poly"].geom_type == "Polygon"] if a.all else [int(v) for v in a.rooms.split(",")]
ious = []; tiles = []
for k, i in enumerate(ids):
    g = gts[i]; box = T.crop_box(g["poly"], W, H, k)
    ls = CH.lines_in(enc, box, min_len=a.min_len); fs = CY.faces(ls, box, reach=a.reach)
    sel = [f for f in fs if f.intersection(g["poly"]).area >= 0.5 * f.area]
    if not sel: print(f"r{i}: no face"); ious.append(0); continue
    U = unary_union(sel).buffer(0.01); U = U if U.geom_type == "Polygon" else max(U.geoms, key=lambda q: q.area)
    pts = [list(p) for p in U.exterior.coords[:-1]]; sc = T.score(pts, g["poly"]); ious.append(sc["iou"])
    seed = [f for f in fs if f.contains(__import__("shapely.geometry", fromlist=["Point"]).Point(*g["xy"]))]
    print(f"r{i} {g['name']}: lines {len(ls)} faces {len(fs)} | room = {len(sel)} faces | oracle IoU {sc['iou']} corners<=6px {sc['corners_hit6']}/{sc['gt_corners']} bnd mean {sc['bnd_mean']} p90 {sc['bnd_p90']} | seed face covers {seed[0].intersection(g['poly']).area / g['poly'].area:.0%} of room, {seed[0].area / g['poly'].area:.2f}x its area" if seed else f"r{i}: IoU {sc['iou']} (label not in a face)")
    if not a.all: tiles.append(T.tile(im, box, g["poly"], pts, f"r{i} parca tavan IoU {sc['iou']} kose {sc['corners_hit6']}/{sc['gt_corners']}"))
print("mean oracle IoU", round(float(np.mean(ious)), 3), "| >=0.95:", sum(v >= 0.95 for v in ious), "of", len(ious), "| min", min(ious))
if tiles:
    n = (len(tiles) + 2) // 3; cw = max(t.width for t in tiles) + 10; ch = max(t.height for t in tiles) + 10
    sh = Image.new("RGB", (cw * 3, ch * n), "white")
    for j, t in enumerate(tiles): sh.paste(t, ((j % 3) * cw, (j // 3) * ch))
    sh.save(T.OUT / "sayfa_yuz_oracle.png")
