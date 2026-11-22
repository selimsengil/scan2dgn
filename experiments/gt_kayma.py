"""Is a floor's misfit one shift of the whole drawing (scan placed slightly off) or room-by-room drawing errors?
Tries shifts of the DGN rooms within +-R cm and reports the best floor-wide shift and the oturma after it."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import cv2, numpy as np
from shapely.geometry import shape
sys.path.insert(0, str(Path(__file__).parent)); import gt_denetim as GD
R = GD.R
def ring_pts(P):
    ring = P.exterior; n = max(8, int(ring.length)); return np.array([ring.interpolate(t).coords[0] for t in np.linspace(0, ring.length, n, endpoint=False)])
def fit(fid, rad_m=0.12):
    L = json.load(open(R / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
    g = cv2.imdecode(np.fromfile(str(R / fid / "clean.png"), np.uint8), 0); ink = GD.ink_of(g); dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5); tol = max(2.0, 0.025 * s)
    H, W = g.shape; rooms = []
    for r in L["rooms"]:
        P = shape(r["interior_polygon"]).buffer(0)
        if P.is_empty: continue
        if P.geom_type != "Polygon": P = max(P.geoms, key=lambda q: q.area)
        rooms.append(ring_pts(P))
    allp = np.vstack(rooms); rad = int(rad_m * s); step = max(1, rad // 24)
    def score(p, dx, dy):
        xi = np.clip(np.round(p[:, 0] + dx).astype(int), 0, W - 1); yi = np.clip(np.round(p[:, 1] + dy).astype(int), 0, H - 1); return (dist[yi, xi] <= tol).mean()
    base = score(allp, 0, 0); best = (base, 0, 0)
    for dx in range(-rad, rad + 1, step):
        for dy in range(-rad, rad + 1, step):
            v = score(allp, dx, dy)
            if v > best[0]: best = (v, dx, dy)
    v, dx, dy = best
    for ddx in range(-step, step + 1):
        for ddy in range(-step, step + 1):
            w = score(allp, dx + ddx, dy + ddy)
            if w > best[0]: best = (w, dx + ddx, dy + ddy)
    per = [score(p, best[1], best[2]) for p in rooms]
    return {"floor": fid, "s": round(s, 1), "base": round(base, 3), "shifted": round(best[0], 3), "dx_cm": round(best[1] / s * 100, 1), "dy_cm": round(best[2] / s * 100, 1), "rooms_ok_after": round(float(np.mean([q >= 0.9 for q in per])), 2)}
if __name__ == "__main__":
    idx = [i["floor"] for i in json.load(open(R / "INDEX.json", encoding="utf-8")) if i.get("room")]
    out = []
    for f in idx:
        if len(sys.argv) > 1 and sys.argv[1].lower() not in f.lower(): continue
        r = fit(f); out.append(r); print(r, flush=True)
    json.dump(out, open(GD.OUT / "KAYMA.json", "w"), indent=0)
