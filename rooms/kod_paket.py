"""Builds the ENCODED FLOOR package for a model that can run Python (dev floors; 015 only for the final blind test):
  kod/cizgiler.json   every line of the plan: id, end points, length, thickness, darkness
  kod/parcalar.json   the closed pieces the lines cut the plan into + which pieces touch, and what lies between them
  kod/kod_araclar.py  small helpers over those files (find, grow, merge, draw)
  kod/KOD.md          what the encoding is and how to use it
With --gt it also prints the loss check: the best room boundaries reachable from the pieces, against the labels.
usage: python kod_paket.py --floor 019 --out kat_test/019_kod_test [--gt]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, shutil, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

import v0_watershed as V0  # noqa: E402
import cizgi_kod as CK  # noqa: E402
import cizgi_hat as CH  # noqa: E402
import cizgi_yuz as CY  # noqa: E402

HERE = config.DATA


def build(gray, tile=900, overlap=220):
    """Lines and pieces for the whole floor. Lines are merged per overlapping tile (the merge is quadratic), keeping in
    each tile the lines whose middle lies in the tile's core, so every line appears once."""
    H, W = gray.shape; enc = CK.encode(gray); lines = []
    step = tile - overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            box = (x0, y0, min(W, x0 + tile), min(H, y0 + tile))
            cx0, cy0 = (x0 + overlap / 2 if x0 else 0), (y0 + overlap / 2 if y0 else 0)
            cx1, cy1 = (x0 + tile - overlap / 2 if x0 + tile < W else W), (y0 + tile - overlap / 2 if y0 + tile < H else H)
            for l in CH.lines_in(enc, box, min_len=8, pad=0):
                m = (l["p1"] + l["p2"]) / 2
                if cx0 <= m[0] < cx1 and cy0 <= m[1] < cy1: lines.append(l)
    for k, l in enumerate(lines): l["id"] = k + 1
    return enc, lines


def border_info(fs, enc, gray):
    """For every pair of touching pieces: shared border length, how much of it is drawn ink, and how dark that ink is."""
    line1 = CK.redraw(enc, gray.shape, width=1)
    near = cv2.distanceTransform((~line1).astype(np.uint8), cv2.DIST_L2, 3) <= 1.8
    tree = STRtree(fs); out = []
    for i, f in enumerate(fs):
        for j in tree.query(f.buffer(0.3)):
            j = int(j)
            if j <= i: continue
            sh = f.exterior.intersection(fs[j].exterior)
            L = sh.length if not sh.is_empty else 0.0
            if L < 3.0: continue
            n = max(2, int(L / 2)); geoms = list(sh.geoms) if hasattr(sh, "geoms") else [sh]
            pts = []
            for gm in geoms:
                if gm.length <= 0: continue
                for s in np.linspace(0, gm.length, max(2, int(gm.length / 2))): pts.append(gm.interpolate(s).coords[0])
            if not pts: continue
            P = np.array(pts); x = np.clip(P[:, 0].round().astype(int), 0, gray.shape[1] - 1); y = np.clip(P[:, 1].round().astype(int), 0, gray.shape[0] - 1)
            on = near[y, x]; ink = float(on.mean())
            # darkest pixel across the border at each sample (3x3) -> median
            g = np.minimum.reduce([gray[np.clip(y + dy, 0, gray.shape[0] - 1), np.clip(x + dx, 0, gray.shape[1] - 1)] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            out.append([i, j, round(L, 1), round(ink, 2), int(np.median(g[on])) if on.any() else 255])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floor", default="019"); ap.add_argument("--out", required=True); ap.add_argument("--gt", action="store_true")
    ap.add_argument("--reach", type=float, default=230.0, help="px a wall line is carried on across an opening (about 3 m; windows up to that width get closed)")
    a = ap.parse_args()
    out = Path(a.out); kod = out / "kod"; kod.mkdir(parents=True, exist_ok=True)
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); H, W = gray.shape
    t = time.time(); enc, lines = build(gray); t1 = time.time() - t
    t = time.time(); fs = CY.faces(lines, (0, 0, W, H), min_area=1.0, long_reach=a.reach, max_gap=a.reach)      # tiny pieces are kept: dropping them leaves slits in a room's outline
    # the paper around the building (pieces touching the image frame) is not part of the plan
    fs = [f for f in fs if f.bounds[0] > 0.5 and f.bounds[1] > 0.5 and f.bounds[2] < W - 0.5 and f.bounds[3] < H - 0.5]; t2 = time.time() - t
    t = time.time(); bord = border_info(fs, enc, gray); t3 = time.time() - t
    print(f"lines {len(lines)} ({t1:.0f}s) | pieces {len(fs)} ({t2:.0f}s) | borders {len(bord)} ({t3:.0f}s)")
    json.dump({"image": f"kat_{a.floor}.png", "size": [W, H], "lines": [
        {"id": l["id"], "p1": [round(float(l["p1"][0]), 1), round(float(l["p1"][1]), 1)], "p2": [round(float(l["p2"][0]), 1), round(float(l["p2"][1]), 1)],
         "len": round(l["len"], 1), "thick": round(l["thick"], 1), "gray": int(l["gray"])} for l in lines]}, open(kod / "cizgiler.json", "w", encoding="utf-8"))
    json.dump({"image": f"kat_{a.floor}.png", "size": [W, H],
               "pieces": [{"id": i, "area": round(f.area, 1), "poly": [[round(x, 1), round(y, 1)] for x, y in f.exterior.coords[:-1]],
                           "holes": [[[round(x, 1), round(y, 1)] for x, y in r.coords[:-1]] for r in f.interiors]} for i, f in enumerate(fs)],
               "borders": bord}, open(kod / "parcalar.json", "w", encoding="utf-8"))
    for f in ("kod_araclar.py", "KOD.md"): shutil.copyfile(HERE / "kod_sablon" / f, kod / f)
    im.save(out / f"kat_{a.floor}.png")
    print("package:", kod, "| sizes KB:", {p.name: p.stat().st_size // 1024 for p in kod.iterdir()})
    if a.gt:
        assert a.floor != "015", "no label check on the blind floor"
        import tek_oda_test as T
        ious = []; hit = tot = 0; rows = []
        tree = STRtree(fs)
        for i, g in enumerate(V0.load_rooms(a.floor)):
            if g["poly"].geom_type != "Polygon": continue
            sel = [fs[int(j)] for j in tree.query(g["poly"]) if fs[int(j)].intersection(g["poly"]).area >= 0.5 * fs[int(j)].area]
            if not sel: ious.append(0.0); rows.append((i, g["name"], 0.0)); continue
            U = unary_union(sel).buffer(0.01); U = U if U.geom_type == "Polygon" else max(U.geoms, key=lambda q: q.area)
            sc = T.score([list(p) for p in U.exterior.coords[:-1]], g["poly"]); ious.append(sc["iou"]); hit += sc["corners_hit6"]; tot += sc["gt_corners"]; rows.append((i, g["name"], sc["iou"]))
        ious = np.array(ious)
        print(f"LOSS CHECK {a.floor}: best reachable mean IoU {ious.mean():.3f} | >=0.95: {(ious >= 0.95).sum()}/{len(ious)} | >=0.90: {(ious >= 0.9).sum()} | min {ious.min():.3f} | corners within 6 px {hit}/{tot}")
        print("   below 0.95:", [(f"r{i}", n, v) for i, n, v in rows if v < 0.95])
