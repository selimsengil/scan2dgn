"""Is an opening box valid the way Citymax checks it?  A box may be drawn many ways; it is valid when
  - it touches or crosses the boundary of exactly two areas (two rooms, or a room and the floor outline / a balcony),
  - on each of them it lies on ONE face: that face runs through the box from one short side to the other
    (touching within `tol` px counts), and
  - no corner (vertex) of any area lies inside the box  -> Citymax 'vertex' / 'birden fazla oda yuzeyine' error,
  - it does not reach a third area.
gecerli(box, areas, tol=1.5) -> (ok: bool, reasons: [str], touched: [names])
areas: [(name, shapely Polygon)], the floor outline included (its ring is the outer face).
Also: dogru(box, gt_box, px_per_m) -> jamb error in cm (both ends along the wall axis) for matching a prediction to a label."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import numpy as np
from shapely.geometry import LineString, Point, Polygon


def _frame(box):
    c = np.asarray(box, float)[:4]; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e))
    a, b = c[k], c[(k + 1) % 4]; u = (b - a) / e[k]; n = np.array([-u[1], u[0]]); L = e[k]
    t = [float((p - a) @ n) for p in c]; return a, u, n, L, min(t), max(t)


def gecerli(box, areas, tol=1.5):
    B = Polygon(box); a, u, n, L, t0, t1 = _frame(box); reasons = []; touched = []
    for name, P in areas:
        ring = P.exterior
        if ring.distance(B) > tol: continue
        touched.append(name)
        # vertex inside the box (strictly, shrunk by 0.5 px)
        inner = B.buffer(-0.5)
        vin = [q for q in np.asarray(ring.coords)[:-1] if inner.contains(Point(*q))]
        if vin: reasons.append(f"{name}: kose kutunun icinde ({len(vin)})"); continue
        # the part of the ring near the box must be one straight face parallel to the box axis, spanning its full length
        near = ring.intersection(B.buffer(tol, join_style=2))
        lines = [g for g in getattr(near, "geoms", [near]) if g.geom_type == "LineString" and g.length > 0.5]
        if not lines: reasons.append(f"{name}: yalniz noktada degiyor"); continue
        segs = [LineString([p, q]) for g in lines for p, q in zip(g.coords[:-1], g.coords[1:]) if np.hypot(*np.subtract(q, p)) > 0.5]   # piece by piece
        par = [s for s in segs if abs(abs(np.dot(np.subtract(s.coords[-1], s.coords[0]) / max(s.length, 1e-9), u)) - 1) < 0.02]
        nonpar = [s for s in segs if s not in par and s.length > tol]
        if nonpar:                                   # a corner of the area touches the box (at its end): Citymax vertex error
            ts = [float((np.asarray(p) - a) @ u) for s in nonpar for p in s.coords]
            reasons.append(f"{name}: kose kutunun ucuna degiyor (t {min(ts):.0f}..{max(ts):.0f} / {L:.0f})"); continue
        if len(lines) != 1:
            reasons.append(f"{name}: kutuda {len(lines)} ayri sinir parcasi"); continue
        par = [LineString(lines[0].coords)]
        s = par[0]; ts = sorted(float((np.asarray(p) - a) @ u) for p in s.coords)
        if ts[0] > tol or ts[-1] < L - tol: reasons.append(f"{name}: yuz kutunun boyunu ortmuyor ({ts[0]:.0f}..{ts[-1]:.0f} / {L:.0f})")
    if len(touched) != 2: reasons.append(f"{len(touched)} alana degiyor ({', '.join(touched)})")
    return not reasons, reasons, touched


def kasa_hata_cm(box, gt, px_per_m):
    """Jamb error: the two ends of the predicted box vs the label box, along the label's wall axis (cm, mean of both ends)."""
    a, u, n, L, _, _ = _frame(gt); ga, gb = 0.0, L
    c = np.asarray(box, float)[:4]; ts = sorted(float((p - a) @ u) for p in c)
    pa, pb = (ts[0] + ts[1]) / 2, (ts[2] + ts[3]) / 2
    return (abs(pa - ga) + abs(pb - gb)) / 2 / px_per_m * 100


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    E = json.load(open(config.KT / "el_etiket" / "malkoc_k2" / "ETIKET.json", encoding="utf-8"))
    areas = [(f'{r["name"]}:{r["id"]}', Polygon(r["poly"]).buffer(0)) for r in E["rooms"]] + [(b["id"], Polygon(b["poly"]).buffer(0)) for b in E["balkonlar"]]
    areas.append(("KAT_ALANI", Polygon(E["kat_alani"]).buffer(0)))
    ok = 0
    for o in E["openings"]:
        v, why, t = gecerli(o["poly"], areas); ok += v
        if not v: print(o["id"], o["kind"], why)
    print("gecerli", ok, "/", len(E["openings"]))
