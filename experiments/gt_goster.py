"""Picture of the label audit for one floor: boundary green where it lies on ink, red where it leaves it (doors/windows blue)."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import cv2, numpy as np
from shapely.geometry import shape
sys.path.insert(0, str(Path(__file__).parent)); import gt_denetim as GD
def floor_pic(fid, out, width=2400, crop=None):
    L, s, g, rooms = GD.audit_floor(fid, 0.025); ink = GD.ink_of(g); dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5); tol = max(2.0, 0.025 * s)
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); v = (v * 0.55 + 115).astype(np.uint8)
    for x in L["structures"]:
        if x["kind"] in ("door", "window"): cv2.polylines(v, [np.round(np.asarray(shape(x["geometry"]).exterior.coords)).astype(np.int32)], True, (230, 120, 0), max(1, int(s * 0.012)))
    for r in L["rooms"]:
        P = shape(r["interior_polygon"]).buffer(0)
        if P.geom_type != "Polygon": P = max(P.geoms, key=lambda q: q.area)
        c = np.asarray(P.exterior.coords)
        for a, b in zip(c[:-1], c[1:]):
            n = max(2, int(np.hypot(*(b - a))))
            for t0, t1 in zip(np.linspace(0, 1, n)[:-1], np.linspace(0, 1, n)[1:]):
                p = a + (b - a) * t0; q = a + (b - a) * t1; m = (p + q) / 2
                ok = dist[min(int(m[1]), g.shape[0] - 1), min(int(m[0]), g.shape[1] - 1)] <= tol
                cv2.line(v, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (0, 170, 0) if ok else (0, 0, 255), max(2, int(s * 0.015)))
    for rr in rooms:
        x0, y0, x1, y1 = rr["bbox"]; cv2.putText(v, f"{rr['oturma']:.2f}", (int(x0 + 6), int(y0 + 0.25 * s)), cv2.FONT_HERSHEY_SIMPLEX, s / 160, (0, 0, 200) if rr["oturma"] < 0.97 else (0, 120, 0), max(1, int(s / 90)))
    if crop: v = v[crop[1]:crop[3], crop[0]:crop[2]]
    k = min(1.0, width / max(v.shape[:2])); cv2.imwrite(str(out), cv2.resize(v, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
if __name__ == "__main__":
    floor_pic(sys.argv[1], sys.argv[2])
