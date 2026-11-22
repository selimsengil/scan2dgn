"""Floor outline (kat alani, outer wall face) of a floor from its rooms: union closed over 45 cm, each outer edge pushed out to
the first parallel ink line 12..45 cm away (>= 60 % on ink), corners re-cut.  Rooms: aciklik_ag/veri rooms (clean + snapped)
plus the DGN rooms that were left out.  Output kat_test/kat_alani/<floor>.json {kat_alani, rooms}.
usage: python kat_alani_uret.py FLOOR [sheet.jpg]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import cv2, numpy as np
from shapely.geometry import Point, Polygon, mapping, shape
from shapely.ops import unary_union
import gt_denetim as GD  # noqa: E402
K = config.KT; OUT = K / "kat_alani"; OUT.mkdir(exist_ok=True)


def uret(fid):
    L = json.load(open(GD.R / fid / "LABELS_COMMON.json", encoding="utf-8")); V = json.load(open(K / "aciklik_ag" / "veri" / f"{fid}.json", encoding="utf-8")); s = V["px_per_m"]
    have = {r["id"] for r in V["rooms"]}; rooms = [{"id": r["id"], "name": r["name"], "poly": shape(r["poly"])} for r in V["rooms"]]
    for r in L["rooms"]:
        if r["id"] not in have:
            P = shape(r["interior_polygon"]).buffer(0); P = P if P.geom_type == "Polygon" else max(P.geoms, key=lambda q: q.area)
            rooms.append({"id": r["id"], "name": r["name"], "poly": P, "not": "DGN (oturmayan)"})
    bal = [shape(x["geometry"]).buffer(0) for x in L["structures"] if x["kind"] == "balcony"]
    g = cv2.imdecode(np.fromfile(str(GD.R / fid / "clean.png"), np.uint8), 0); ink = GD.ink_of(g); dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5); H, W = g.shape
    def cov(a, b):
        n = max(6, int(np.hypot(*(b - a)))); t = np.linspace(0.05, 0.95, n)[:, None]; p = a + t * (b - a)
        return float((dist[np.clip(np.round(p[:, 1]).astype(int), 0, H - 1), np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)] <= 2.0).mean())
    def inter(p1, d1, p2, d2):
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6: return None
        return p1 + np.linalg.solve(A, p2 - p1)[0] * d1
    U = unary_union([r["poly"] for r in rooms] + bal); C = U.buffer(0.45 * s, join_style=2).buffer(-0.45 * s, join_style=2)
    C = C if C.geom_type == "Polygon" else max(C.geoms, key=lambda q: q.area); C = C.simplify(6); P = np.array(C.exterior.coords)[:-1]
    if Polygon(P).exterior.is_ccw: P = P[::-1]
    n = len(P); Q = P.copy(); pushed = 0
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        if np.hypot(*(b - a)) < 30: continue
        d = (b - a) / np.hypot(*(b - a)); nv = np.array([-d[1], d[0]])
        if Polygon(P).contains(Point(*((a + b) / 2 + 5 * nv))): nv = -nv
        best = None
        for kk in range(int(0.12 * s), int(0.45 * s)):
            if cov(a + kk * nv, b + kk * nv) >= 0.6: best = kk; break
        if best is None: continue
        pushed += 1; a2 = a + best * nv
        na = inter(a2, d, Q[(k - 1) % n], P[k] - P[(k - 1) % n]); nb = inter(a2, d, Q[(k + 1) % n], P[(k + 2) % n] - P[(k + 1) % n])
        if na is not None: Q[k] = na
        if nb is not None: Q[(k + 1) % n] = nb
    KA = Polygon(Q).buffer(0)
    out = {"floor": fid, "kat_alani": [list(map(float, p)) for p in np.asarray(KA.exterior.coords)[:-1]], "itilen_kenar": pushed, "kenar": n,
           "rooms": [{"id": r["id"], "name": r["name"], "poly": mapping(r["poly"]), "not": r.get("not", "")} for r in rooms], "balkonlar": [mapping(b) for b in bal]}
    json.dump(out, open(OUT / f"{fid}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(fid, "kat alani", round(KA.area / s / s, 1), "m2, itilen", pushed, "/", n)
    if len(sys.argv) > 2:
        sc = 0.3; ys, xs = np.where(ink); y0, x0 = max(0, ys.min() - 50), max(0, xs.min() - 50)
        v = cv2.cvtColor(cv2.resize(g[y0:, x0:], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR); v = (v * 0.55 + 115).astype(np.uint8)
        f = lambda p: np.round((np.asarray(p, float) - [x0, y0]) * sc).astype(np.int32)
        for r in rooms: cv2.polylines(v, [f(np.asarray(r["poly"].exterior.coords))], True, (0, 150, 0), 1)
        cv2.polylines(v, [f(out["kat_alani"])], True, (0, 0, 230), 2); cv2.imwrite(sys.argv[2], v)


if __name__ == "__main__": uret(sys.argv[1])
