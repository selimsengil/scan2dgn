"""Sample sheet of the opening audit: crops around DGN doors/windows. green = reliable, red = jamb not on ink,
orange = jamb ok but a touched room is not clean.  Room outlines thin blue.
usage: python aciklik_sayfa.py --floor F [--n 24] [--out file.jpg]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json
from pathlib import Path
import cv2, numpy as np
from shapely.geometry import shape
import gt_denetim as GD
ap = argparse.ArgumentParser(); ap.add_argument("--floor", required=True); ap.add_argument("--n", type=int, default=24); ap.add_argument("--out", default=""); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
A = json.load(open(GD.OUT / "ACIKLIK_DENETIM.json", encoding="utf-8"))[a.floor]
L = json.load(open(GD.R / a.floor / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
g = cv2.imdecode(np.fromfile(str(GD.R / a.floor / "clean.png"), np.uint8), 0); v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
for r in L["rooms"]:
    P = shape(r["interior_polygon"]).buffer(0)
    for p in ([P] if P.geom_type == "Polygon" else list(getattr(P, "geoms", []))): cv2.polylines(v, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], True, (255, 120, 0), 1)
geo = {x["id"]: shape(x["geometry"]) for x in L["structures"]}
rng = np.random.default_rng(a.seed); items = list(A); rng.shuffle(items); items = items[:a.n]; tiles = []; T = 260
for x in items:
    P = geo[x["id"]]; col = (0, 170, 0) if x["guvenilir"] else ((0, 0, 230) if x["jamb"] < 0.8 else (0, 140, 255))
    c = v.copy(); cv2.polylines(c, [np.round(np.asarray(P.exterior.coords)).astype(np.int32)], True, col, 2)
    cx, cy = P.centroid.x, P.centroid.y; h = int(1.2 * s); y0, x0 = max(0, int(cy - h)), max(0, int(cx - h))
    t = c[y0:int(cy + h), x0:int(cx + h)]; t = cv2.resize(t, (T, T))
    cv2.putText(t, f"{x['kind'][0]} j{x['jamb']:.2f}", (4, 16), 0, 0.5, col, 2); tiles.append(t)
while len(tiles) % 6: tiles.append(np.full((T, T, 3), 255, np.uint8))
img = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
cv2.imwrite(a.out or str(GD.OUT / f"aciklik_{a.floor}.jpg"), img)
