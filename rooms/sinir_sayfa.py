"""Picture sheet for the boundary score: for every GT room, the model's line in BLUE where it is the office line
(within tol px) and RED where it is a wrong line; the parts of the office line the model did not draw in ORANGE.
usage: python sinir_sayfa.py <run dir> [--floor 019] [--tol 3]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point
import v0_watershed as V0, sinir_skor as SS

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--floor", default="019"); ap.add_argument("--tol", type=float, default=3.0); ap.add_argument("--cols", type=int, default=6); ap.add_argument("--rooms", default=""); ap.add_argument("--size", type=int, default=460); ap.add_argument("--out", default="sinir_sayfa.png")
a = ap.parse_args(); run = Path(a.run)
im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); W, H = im.size
allg = V0.load_rooms(a.floor); gts = [g for g in allg if g["poly"].geom_type == "Polygon"]
tiles = []
want = [int(v) for v in a.rooms.split(",")] if a.rooms else None
for gi, pred in SS.load_pairs(run, a.floor, allg):
    if want and gi not in want: continue
    g = gts[gi]; x0, y0, x1, y1 = g["poly"].bounds; m = 40
    box = (max(0, int(x0 - m)), max(0, int(y0 - m)), min(W, int(x1 + m)), min(H, int(y1 + m)))
    c = im.crop(box); s = a.size / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    c = Image.blend(c, Image.new("RGB", c.size, "white"), 0.45); d = ImageDraw.Draw(c)
    def dots(P, ok, col_ok, col_bad, r):
        for (x, y), k in zip(P, ok):
            X, Y = (x - box[0]) * s, (y - box[1]) * s; d.ellipse([X - r, Y - r, X + r, Y + r], fill=col_ok if k else col_bad)
    gs = SS.samples(g["poly"], 1.5)
    if pred is None:
        dots(gs, np.zeros(len(gs), bool), None, (255, 140, 0), 2.2); cz = fz = None
    else:
        dg = np.array([pred.exterior.distance(Point(*p)) for p in gs]); ps = SS.samples(pred, 1.5); dp = np.array([g["poly"].exterior.distance(Point(*p)) for p in ps])
        dots(gs[dg > a.tol], np.zeros(int((dg > a.tol).sum()), bool), None, (255, 140, 0), 2.2)
        dots(ps, dp <= a.tol, (30, 70, 255), (230, 0, 0), 1.6); cz, fz = (dg <= a.tol).mean(), (dp > a.tol).mean()
    d.rectangle([0, 0, c.width, 13], fill="white")
    d.text((3, 1), f"r{gi} {g['name']}  cizgi {cz:.2f} fazla {fz:.2f}" if cz is not None else f"r{gi} {g['name']}  CIZILMEDI", fill=(160, 0, 0))
    tiles.append(c)
n = (len(tiles) + a.cols - 1) // a.cols; cw = max(t.width for t in tiles) + 6; ch = max(t.height for t in tiles) + 6
sh = Image.new("RGB", (cw * a.cols, ch * n), "white")
for j, t in enumerate(tiles): sh.paste(t, ((j % a.cols) * cw, (j // a.cols) * ch))
sh.save(run / a.out); print(run / a.out, sh.size)
