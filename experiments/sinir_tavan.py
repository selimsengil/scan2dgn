"""Ceiling of the encoded package under the boundary score (dev floors, never 015): choose every piece lying mostly
inside the GT room and score the union with sinir_skor. Shows how much line detail the encoding itself loses.
usage: python sinir_tavan.py --floor 019 [--tol 3]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
import v0_watershed as V0, cizgi_yuz as CY, kod_paket as KP, sinir_skor as SS

ap = argparse.ArgumentParser(); ap.add_argument("--floor", default="019"); ap.add_argument("--tol", type=float, default=3.0); ap.add_argument("--save", default=""); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--min-area", type=float, default=25.0); ap.add_argument("--close", type=float, default=0.0); ap.add_argument("--long-reach", type=float, default=150); ap.add_argument("--long-len", type=float, default=30); ap.add_argument("--max-gap", type=float, default=150); a = ap.parse_args(); assert a.floor != "015"
gray = np.array(Image.open(V0.B / a.floor / "clean.png").convert("L")); H, W = gray.shape
enc, lines = KP.build(gray); fs = CY.faces(lines, (0, 0, W, H), min_area=a.min_area, long_reach=a.long_reach, long_len=a.long_len, max_gap=a.max_gap); tree = STRtree(fs); print('pieces', len(fs))
lab = json.load(open(V0.B / a.floor / "LABELS_COMMON.json", encoding="utf-8"))
columns = unary_union([shape(s["geometry"]).buffer(0) for s in lab["structures"] if s["kind"] == "column"])
rows = []
for i, g in enumerate(q for q in V0.load_rooms(a.floor) if q["poly"].geom_type == "Polygon"):
    sel = [fs[int(j)] for j in tree.query(g["poly"]) if fs[int(j)].intersection(g["poly"]).area >= a.thr * fs[int(j)].area]
    U = None
    if sel:
        U = unary_union(sel).buffer(0.01)
        if a.close > 0: U = U.buffer(a.close, join_style=2).buffer(-a.close, join_style=2)      # close slits narrower than 2*close, corners stay sharp
        U = U if U.geom_type == "Polygon" else max(U.geoms, key=lambda q: q.area)
        U = SS.clean(list(U.simplify(0.4).exterior.coords))
    r = SS.room_score(g["poly"], U, columns, a.tol); r.update({"room": i, "name": g["name"]}); rows.append(r)
    r["_b"] = [list(p) for p in U.exterior.coords[:-1]] if U is not None else []
    print(f"  r{i:<2} {str(g['name'])[:10]:10} cizgi {r['cizgi']:.2f}  fazla {r['fazla']:.2f}  kose {r['kose']}/{r['kose_gt']}  kolon {r['kolon_ok']}/{r['kolon_len']}  p95 {r['sapma_p95']}  {'KABUL' if r['kabul'] else ''}")
print({"tol": a.tol, "cizgi": round(float(np.mean([r["cizgi"] for r in rows])), 3), "fazla": round(float(np.mean([r["fazla"] for r in rows])), 3),
       "kose": f"{sum(r['kose'] for r in rows)}/{sum(r['kose_gt'] for r in rows)}", "kolon": f"{sum(r['kolon_ok'] for r in rows)}/{sum(r['kolon_len'] for r in rows)}",
       "kabul": sum(r["kabul"] for r in rows), "cizgi>=0.90": sum(r["cizgi"] >= 0.9 for r in rows), "rooms": len(rows)})

if a.save:
    out = Path(a.save); out.mkdir(parents=True, exist_ok=True)
    allg = V0.load_rooms(a.floor); ids = [i for i, g in enumerate(allg) if g["poly"].geom_type == "Polygon"]
    json.dump({"rooms": [{"room": ids[r["room"]], "boundary": r["_b"]} for r in rows]}, open(out / "SONUC.json", "w"))
