"""Offline A/B of how pieces are chosen from saved rough polygons (dev floor, no model call): plain vote against the
minimum cut that prefers drawn lines, on coarse and fine pieces. Boundary score for each.
usage: python kesim_test.py kat_test/gorunum/019_scan_g1 [--floor 019]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import Point, shape
from shapely.ops import unary_union
import v0_watershed as V0, tek_oda_test as T, cizgi_kod as CK, cizgi_hat as CH, cizgi_yuz as CY, kod_paket as KP, oda_parca as OP, sinir_skor as SS, parca_kesim as PK

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--floor", default="019"); ap.add_argument("--lams", default="8,20,40"); a = ap.parse_args(); assert a.floor != "015"
im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); H, W = gray.shape
enc = CK.encode(gray); allg = V0.load_rooms(a.floor); gts = [g for g in allg if g["poly"].geom_type == "Polygon"]; ids = [i for i, g in enumerate(allg) if g["poly"].geom_type == "Polygon"]
lab = json.load(open(V0.B / a.floor / "LABELS_COMMON.json", encoding="utf-8"))
columns = unary_union([shape(s["geometry"]).buffer(0) for s in lab["structures"] if s["kind"] == "column"])
src = json.load(open(Path(a.run) / "SONUC.json", encoding="utf-8"))["rooms"]
variants = [("vote coarse", 150, None), ("vote fine", 230, None)] + [(f"cut fine lam={l}", 230, float(l)) for l in a.lams.split(",")] + [(f"cut coarse lam={l}", 150, float(l)) for l in a.lams.split(",")[:2]]
res = {v[0]: [] for v in variants}; saved = {v[0]: [] for v in variants}
for k, r in enumerate(src):
    gi = ids.index(r["room"]); g = gts[gi]; box = T.crop_box(g["poly"], W, H, k)
    P = OP.clean(r["rough"]) if len(r.get("rough") or []) >= 3 else None
    cache = {}
    for name, reach, lam in variants:
        if reach not in cache:
            fs = CY.faces(CH.lines_in(enc, box, min_len=8), box, min_area=1.0, long_reach=reach, max_gap=reach); cache[reach] = (fs, KP.border_info(fs, enc, gray))
        fs, bord = cache[reach]; U = None
        if P is not None:
            if lam is None: U, _ = OP.vote([P], fs, seed_xy=g["xy"])
            else:
                sel = PK.select(fs, bord, P, g["xy"], lam)
                if sel:
                    U = unary_union([fs[i] for i in sel]).buffer(0.01).buffer(2.0, join_style=2).buffer(-2.0, join_style=2)
                    if U.geom_type != "Polygon":
                        hit = [q for q in U.geoms if q.contains(Point(*g["xy"]))]; U = hit[0] if hit else max(U.geoms, key=lambda q: q.area)
        pred = SS.clean(list(U.simplify(0.4).exterior.coords)) if U is not None else None
        res[name].append(SS.room_score(g["poly"], pred, columns, 3.0)); saved[name].append({"room": r["room"], "boundary": [list(p) for p in pred.exterior.coords[:-1]] if pred is not None else []})
print(f"{'variant':22} {'cizgi':>6} {'fazla':>6} {'kose':>9} {'kolon':>6} {'kabul':>6} {'>=0.90':>7}")
for name, rows in res.items():
    kc = sum(r["kose"] for r in rows); kg = sum(r["kose_gt"] for r in rows); co = sum(r["kolon_ok"] for r in rows) / max(1, sum(r["kolon_len"] for r in rows))
    print(f"{name:22} {np.mean([r['cizgi'] for r in rows]):6.3f} {np.mean([r['fazla'] for r in rows]):6.3f} {str(kc) + '/' + str(kg):>9} {co:6.2f} {sum(r['kabul'] for r in rows):6d} {sum(r['cizgi'] >= 0.9 for r in rows):7d}")
best = max(res, key=lambda n: np.mean([r["cizgi"] for r in res[n]])); out = Path(a.run).parent / (Path(a.run).name + "_kesim"); out.mkdir(exist_ok=True)
json.dump({"rooms": saved[best], "variant": best}, open(out / "SONUC.json", "w")); print("saved best:", best, "->", out)
