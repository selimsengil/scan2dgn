"""The model's rough polygon chooses the pieces: every face lying mostly inside the rough polygon is taken, and the
room is the outline of their union (exact, on the plan's strokes). Scores saved model polygons before/after; no model call.
usage: python yuz_oy_test.py ornekli_o1 [sonuc_a1 ...] [--thr 0.5]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union
import v0_watershed as V0, cizgi_kod as CK, cizgi_hat as CH, cizgi_yuz as CY, tek_oda_test as T


def vote(pred, fs, thr=0.5):
    P = Polygon(pred).buffer(0)
    if P.geom_type != "Polygon": P = max(P.geoms, key=lambda q: q.area)
    sel = [f for f in fs if f.intersection(P).area >= thr * f.area]
    if not sel: return None
    U = unary_union(sel).buffer(0.01)
    if U.geom_type != "Polygon": U = max(U.geoms, key=lambda q: q.area)
    return [list(p) for p in U.exterior.coords[:-1]]


if __name__ == "__main__":
    thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.5
    tags = [t for t in sys.argv[1:] if not t.startswith("--") and not t.replace(".", "").isdigit()]
    im = Image.open(V0.B / "019" / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); W, H = im.size
    enc = CK.encode(gray); gts = V0.load_rooms("019")
    for tag in tags:
        tiles = []; a0 = []; a1 = []
        for k, i in enumerate([11, 18, 7, 21, 14]):
            f = T.OUT / f"{tag}_r{i}.json"; pred = None
            if f.exists():
                d = json.load(open(f, encoding="utf-8")); pred = d.get("pred") or d["hist"][-1]
            elif (T.OUT / f"{tag}.json").exists():
                pred = [r for r in json.load(open(T.OUT / f"{tag}.json", encoding="utf-8")) if r["room"] == i][0]["pred"]
            if not pred or len(pred) < 3: continue
            g = gts[i]; box = T.crop_box(g["poly"], W, H, k); fs = CY.faces(CH.lines_in(enc, box, min_len=8), box)
            new = vote(pred, fs, thr) or pred
            s0 = T.score(pred, g["poly"]); s1 = T.score(new, g["poly"]); a0.append(s0["iou"]); a1.append(s1["iou"])
            print(f"{tag} r{i}: IoU {s0['iou']} -> {s1['iou']} | corners<=6px {s0['corners_hit6']} -> {s1['corners_hit6']} of {s0['gt_corners']} | boundary mean {s0['bnd_mean']} -> {s1['bnd_mean']} px")
            tiles.append(T.tile(im, box, g["poly"], new, f"r{i} parca oyu IoU {s1['iou']} kose {s1['corners_hit6']}/{s1['gt_corners']} (once {s0['iou']}, {s0['corners_hit6']})"))
        if tiles:
            print(f"{tag}: mean IoU {np.mean(a0):.3f} -> {np.mean(a1):.3f}")
            n = (len(tiles) + 2) // 3; cw = max(t.width for t in tiles) + 10; ch = max(t.height for t in tiles) + 10
            sh = Image.new("RGB", (cw * 3, ch * n), "white")
            for j, t in enumerate(tiles): sh.paste(t, ((j % 3) * cw, (j // 3) * ch))
            sh.save(T.OUT / f"sayfa_oy_{tag}.png")
