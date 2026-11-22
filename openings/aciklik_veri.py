"""Training data of the opening net, per floor (pixels of clean.png):
  rooms    = room outlines that sit on the scan: clean as drawn (oturma >= 0.95) or clean after the engine's face
             snapping (motor.oturt, R = 5 cm); the others are left out of the room input.
  openings = DGN doors/windows whose jambs lie on ink (jamb >= 0.8), or that do after snapping (aciklik_oturt.snap,
             +-8 cm, jamb >= 0.9 and the wall goes on beyond both jambs);  'bad' = the rest (ignored in training).
Output: kat_test/aciklik_ag/veri/<floor>.json + VERI_OZET.json.   Run with .venv (motor deps).
usage: python aciklik_veri.py [--only substr]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping, shape

import gt_denetim as GD  # noqa: E402
from aciklik_denetim import audit  # noqa: E402
from aciklik_oturt import snap  # noqa: E402
from etiket_oturt_test import score  # noqa: E402
from citymax.motor import oturt  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

OUT = config.KT / "aciklik_ag" / "veri"


def floor(fid, aud):
    L, s, g, ra = GD.audit_floor(fid, 0.025); ink = GD.ink_of(g); inkf = ink.astype(np.uint8)
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5); tol = max(2.0, 0.025 * s)
    opens = [shape(x["geometry"]) for x in L["structures"] if x["kind"] in ("door", "window")]
    op = unary_union([o.buffer(0.05 * s) for o in opens if o.is_valid]) if opens else None; opm = np.zeros(g.shape, np.uint8)
    if op is not None and not op.is_empty:
        for p in ([op] if op.geom_type == "Polygon" else list(op.geoms)): cv2.fillPoly(opm, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], 1)
    rooms = []; nr = {"temiz": 0, "oturtuldu": 0, "dislandi": 0}
    for r, au in zip([r for r in L["rooms"] if not shape(r["interior_polygon"]).is_empty], ra):
        P = shape(r["interior_polygon"]); P = P if P.is_valid else P.buffer(0)
        if P.geom_type != "Polygon": P = max(P.geoms, key=lambda q: q.area)
        if au["oturma"] >= 0.95: rooms.append({"id": r["id"], "name": r["name"], "how": "temiz", "poly": mapping(P)}); nr["temiz"] += 1; continue
        try:
            new, _, _ = oturt(inkf, [list(c) for c in list(P.exterior.coords)[:-1]], R=max(4, int(round(0.05 * s)))); Q = Polygon(new)
        except Exception: Q = None
        if Q is not None and Q.is_valid and Q.area > 0.5 * P.area and score(Q, dist, opm, tol) >= 0.95:
            rooms.append({"id": r["id"], "name": r["name"], "how": "oturtuldu", "poly": mapping(Q)}); nr["oturtuldu"] += 1
        else: nr["dislandi"] += 1
    A = {x["id"]: x for x in audit(fid, 0.025, aud, 0.8)}; ops = []; bad = []; no = {"temiz": 0, "oturtuldu": 0, "dislandi": 0}
    for x in L["structures"]:
        if x["id"] not in A: continue
        P = shape(x["geometry"]).buffer(0); a = A[x["id"]]
        if a["jamb"] >= 0.8: ops.append({"id": x["id"], "kind": x["kind"], "how": "temiz", "poly": mapping(P)}); no["temiz"] += 1; continue
        P2, j0, j1, d = snap(P, dist, tol, int(round(0.08 * s)), s=s)
        if j1 > j0: ops.append({"id": x["id"], "kind": x["kind"], "how": "oturtuldu", "kayma": d, "poly": mapping(P2)}); no["oturtuldu"] += 1
        else: bad.append({"id": x["id"], "kind": x["kind"], "poly": mapping(P)}); no["dislandi"] += 1
    return {"floor": fid, "px_per_m": s, "rooms": rooms, "openings": ops, "bad": bad}, nr, no


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    aud = json.load(open(GD.OUT / "DENETIM.json", encoding="utf-8")); ozet = {}
    for i in json.load(open(GD.R / "INDEX.json", encoding="utf-8")):
        fid = i["floor"]
        if not i.get("room") or (a.only and a.only.lower() not in fid.lower()): continue
        d, nr, no = floor(fid, aud); json.dump(d, open(OUT / f"{fid}.json", "w", encoding="utf-8"), ensure_ascii=False)
        kd = {k: sum(1 for o in d["openings"] if o["kind"] == k) for k in ("door", "window")}
        ozet[fid] = {"oda": nr, "aciklik": no, "guvenilir_kapi": kd["door"], "guvenilir_pencere": kd["window"]}
        print(f"{fid[:46]:48} oda {nr}  aciklik {no}", flush=True)
    json.dump(ozet, open(OUT.parent / "VERI_OZET.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
