"""Can the engine's face snapping (motor.oturt) clean shifted DGN room labels automatically?
For each room of the given floors: snap the DGN outline to the scan's inner wall face (search radius --r-m),
then re-score with the gt_denetim measure (share of the outline on ink, openings excluded).
usage: python etiket_oturt_test.py --floors "a,b" [--r-m 0.05]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from citymax.motor import oturt  # noqa: E402
import gt_denetim as GD  # noqa: E402


def score(P, ink_dist, opm, tol):
    H, W = opm.shape; ring = P.exterior; n = max(8, int(ring.length))
    pts = np.array([ring.interpolate(t).coords[0] for t in np.linspace(0, ring.length, n, endpoint=False)])
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    inop = opm[yi, xi] > 0; off = (ink_dist[yi, xi] > tol) & ~inop
    return float(1 - off.sum() / max((~inop).sum(), 1))


if __name__ == "__main__":
    import cv2
    from shapely.geometry import shape
    from shapely.ops import unary_union
    ap = argparse.ArgumentParser(); ap.add_argument("--floors", required=True); ap.add_argument("--r-m", type=float, default=0.05); a = ap.parse_args()
    res = {}
    for fid in a.floors.split(","):
        L, s, g, aud = GD.audit_floor(fid, 0.025); ink = GD.ink_of(g)
        dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5); tol = max(2.0, 0.025 * s)
        opens = [shape(x["geometry"]) for x in L["structures"] if x["kind"] in ("door", "window")]
        op = unary_union([o.buffer(0.05 * s) for o in opens if o.is_valid]) if opens else None; opm = np.zeros(g.shape, np.uint8)
        if op is not None and not op.is_empty:
            for p in ([op] if op.geom_type == "Polygon" else list(op.geoms)): cv2.fillPoly(opm, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], 1)
        inkf = ink.astype(np.uint8)
        before = after = n = 0; rows = []
        for r, au in zip([r for r in L["rooms"] if not shape(r["interior_polygon"]).is_empty], aud):
            P = shape(r["interior_polygon"]); P = P if P.is_valid else P.buffer(0)
            if P.geom_type != "Polygon": P = max(P.geoms, key=lambda q: q.area)
            pts = [list(c) for c in list(P.exterior.coords)[:-1]]
            try: new, er, mv = oturt(inkf, pts, R=max(4, int(round(a.r_m * s))))
            except Exception: continue
            Q = Polygon(new)
            if not Q.is_valid or Q.area < 0.5 * P.area: Q = P
            s0, s1 = au["oturma"], score(Q, dist, opm, tol); n += 1; before += s0 >= 0.95; after += s1 >= 0.95
            rows.append([r["name"], s0, round(s1, 3), round(mv, 1)])
        res[fid] = {"oda": n, "temiz_once": before, "temiz_sonra": after, "odalar": rows}
        print(fid, n, "temiz", before, "->", after, flush=True)
    config.KT.joinpath("gt_denetim", "OTURT_TEST.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
