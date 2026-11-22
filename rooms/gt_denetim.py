"""How well does each DGN room sit on the drawn line of the scan?  A room drawn on the inner wall face runs along ink
all the way round, except where it crosses a door / window opening.  A room that leaves the ink for a long run is
shifted, on the wrong face or misses a column: a drawing (label) error, or a picture that does not match the DGN.

Per room: oturma = share of the boundary (outside openings) within tol of ink, kacak_m = longest run off the ink (m).
Output: kat_test/gt_denetim/DENETIM.json (+ per-floor thumbnails for the page).
usage: python gt_denetim.py [--only Kisla] [--tol-m 0.025]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, os
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

R = config.KT / "dgn_katlar"; OUT = config.KT / "gt_denetim"


def ink_of(g):
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink > 0


def runs(bad):
    """Longest run of True in a closed ring (wraps around)."""
    if not bad.any(): return 0
    if bad.all(): return len(bad)
    k = int(np.argmin(bad)); b = np.roll(bad, -k); best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best


def audit_floor(fid, tol_m):
    L = json.load(open(R / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
    g = cv2.imdecode(np.fromfile(str(R / fid / "clean.png"), np.uint8), 0); ink = ink_of(g)
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5); tol = max(2.0, tol_m * s)
    opens = [shape(x["geometry"]) for x in L["structures"] if x["kind"] in ("door", "window")]
    op = unary_union([o.buffer(0.05 * s) for o in opens if o.is_valid]) if opens else None
    opm = np.zeros(g.shape, np.uint8)
    if op is not None and not op.is_empty:
        for p in ([op] if op.geom_type == "Polygon" else list(op.geoms)): cv2.fillPoly(opm, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], 1)
    H, W = g.shape; out = []
    for r in L["rooms"]:
        P = shape(r["interior_polygon"]); P = P if P.is_valid else P.buffer(0)
        if P.is_empty: continue
        if P.geom_type != "Polygon": P = max(P.geoms, key=lambda q: q.area)
        ring = P.exterior; n = max(8, int(ring.length)); pts = np.array([ring.interpolate(t).coords[0] for t in np.linspace(0, ring.length, n, endpoint=False)])
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        inop = opm[yi, xi] > 0; d = dist[yi, xi]; off = (d > tol) & ~inop
        valid = (~inop).sum()
        out.append({"id": r["id"], "name": r["name"], "m2": r.get("area_text_value"), "area_m2": round(P.area / s / s, 2),
                    "oturma": round(float(1 - off.sum() / max(valid, 1)), 3), "kacak_m": round(runs(off) / s, 2), "acik_pay": round(float(inop.mean()), 3),
                    "bbox": [float(v) for v in P.bounds]})
    return L, s, g, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--tol-m", type=float, default=0.025); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); allr = {}
    for i in json.load(open(R / "INDEX.json", encoding="utf-8")):
        fid = i["floor"]
        if not i.get("room") or (a.only and a.only.lower() not in fid.lower()): continue
        L, s, g, rooms = audit_floor(fid, a.tol_m)
        allr[fid] = {"project": L["project"], "kat_no": L["kat_no"], "px_per_m": s, "rooms": rooms}
        o = np.array([r["oturma"] for r in rooms]); k = np.array([r["kacak_m"] for r in rooms])
        print(f"{fid[:48]:50} rooms {len(rooms):3}  oturma med {np.median(o):.3f}  temiz {(o >= 0.97).mean():.2f}  kacak>0.3m {(k > 0.3).sum()}", flush=True)
    json.dump(allr, open(OUT / "DENETIM.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
