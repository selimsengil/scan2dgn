"""Shows what a piece classifier would learn from (no training): for a window of a labelled floor, three pictures
  *_kaynak.jpg   the scan
  *_etiket.jpg   the office/assistant labels drawn on it (rooms, walls, columns, doors, windows)
  *_parca.jpg    the encoding's pieces, each coloured by the label covering most of it = the training target
and STATS.json with the piece counts per class.   Never 015.
usage: python veri_goster.py --floor 013 --box 900,1400,2900,3000 --reach 440"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import shape, box as sbox
from shapely.ops import unary_union
from shapely.strtree import STRtree

import v0_watershed as V0, cizgi_kod as CK, cizgi_hat as CH, cizgi_yuz as CY  # noqa: E402

OUT = config.KT / "egitim_veri"
COL = {"oda": (60, 180, 75), "duvar": (0, 110, 220), "kolon": (245, 130, 30), "kapi": (225, 30, 60), "pencere": (0, 190, 200), "bilinmiyor": (150, 150, 150)}
KIND = {"wall": "duvar", "column": "kolon", "door": "kapi", "window": "pencere"}


def paint(im, box, polys, alpha=110, outline=True):
    c = im.crop(box).convert("RGBA"); ov = Image.new("RGBA", c.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    for P, col in polys:
        for Q in ([P] if P.geom_type == "Polygon" else list(getattr(P, "geoms", []))):
            if Q.geom_type != "Polygon" or Q.is_empty: continue
            pts = [(x - box[0], y - box[1]) for x, y in Q.exterior.coords]
            d.polygon(pts, fill=col + (alpha,), outline=(col + (255,)) if outline else None)
    return Image.alpha_composite(c, ov).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floor", required=True); ap.add_argument("--box", default=""); ap.add_argument("--reach", type=float, default=230.0)
    a = ap.parse_args(); assert a.floor != "015"; OUT.mkdir(parents=True, exist_ok=True)
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); W, H = im.size
    box = tuple(int(v) for v in a.box.split(",")) if a.box else (0, 0, W, H); B = sbox(*box)
    L = json.load(open(V0.B / a.floor / "LABELS_COMMON.json", encoding="utf-8"))
    st = {k: [] for k in KIND.values()}
    for s in L["structures"]:
        g = shape(s["geometry"]).buffer(0)
        if g.intersects(B): st[KIND[s["kind"]]].append(g)
    rooms = [g["poly"] for g in V0.load_rooms(a.floor) if g["poly"].intersects(B)]
    lab = [(p, COL["oda"]) for p in rooms] + [(p, COL[k]) for k in ("duvar", "kolon", "kapi", "pencere") for p in st[k]]
    im.crop(box).save(OUT / f"{a.floor}_kaynak.jpg", quality=82); paint(im, box, lab).save(OUT / f"{a.floor}_etiket.jpg", quality=82)
    enc = CK.encode(np.array(im.convert("L"))); fs = CY.faces(CH.lines_in(enc, box, min_len=8), box, min_area=1.0, long_reach=a.reach, max_gap=a.reach)
    U = {k: unary_union(v) if v else None for k, v in st.items()}; U["oda"] = unary_union(rooms) if rooms else None
    cls = []
    for f in fs:
        ar = {k: (f.intersection(u).area if u is not None and f.intersects(u) else 0.0) for k, u in U.items()}
        k = max(ar, key=ar.get); cls.append(k if ar[k] >= 0.5 * f.area else "bilinmiyor")
    paint(im, box, [(f, COL[k]) for f, k in zip(fs, cls)], alpha=120, outline=False).save(OUT / f"{a.floor}_parca.jpg", quality=82)
    cnt = Counter(cls); area = Counter()
    for f, k in zip(fs, cls): area[k] += f.area
    stats = {"floor": a.floor, "box": box, "pieces": len(fs), "count": dict(cnt), "area_px": {k: int(v) for k, v in area.items()},
             "labels_in_window": {"oda": len(rooms), **{k: len(v) for k, v in st.items()}}}
    p = OUT / "STATS.json"; allst = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    allst[a.floor] = stats; json.dump(allst, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(stats)
