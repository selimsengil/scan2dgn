"""Which DGN doors/windows are reliable training targets?  An opening box is drawn from jamb to jamb across the wall,
so in the scan both short sides of the box (the jambs, across the wall) lie on ink.  The rooms it touches must be
clean too (gt_denetim oturma >= 0.95), because those room outlines are the second input of the opening net.

Per opening: jamb = share of each short side within tol of ink (min of the two), rooms = ids of the rooms it touches,
guvenilir = jamb >= --min-jamb and every touched room clean and at least one touched.
Output: kat_test/gt_denetim/ACIKLIK_DENETIM.json
usage: python aciklik_denetim.py [--skip Kisla] [--tol-m 0.025] [--min-jamb 0.8]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, shape

import gt_denetim as GD

R = GD.R; OUT = GD.OUT


def sides(P):
    r = P.minimum_rotated_rectangle; c = np.array(r.exterior.coords)[:4]
    e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmin(e))
    return (c[k], c[(k + 1) % 4]), (c[(k + 2) % 4], c[(k + 3) % 4]), max(e), min(e)


def on_ink(seg, dist, tol):
    a, b = np.asarray(seg[0]), np.asarray(seg[1]); n = max(4, int(np.hypot(*(b - a))))
    t = np.linspace(0.1, 0.9, n)[:, None]; p = a + t * (b - a); H, W = dist.shape          # ends left out: corners
    xi = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
    return float((dist[yi, xi] <= tol).mean())


def audit(fid, tol_m, aud, min_jamb, min_ot=0.95):
    L = json.load(open(R / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
    g = cv2.imdecode(np.fromfile(str(R / fid / "clean.png"), np.uint8), 0); ink = GD.ink_of(g)
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 5); tol = max(2.0, tol_m * s)
    ot = {a["id"]: a["oturma"] for a in aud.get(fid, {}).get("rooms", [])}
    rooms = []
    for r in L["rooms"]:
        P = shape(r["interior_polygon"]).buffer(0)
        if not P.is_empty: rooms.append((r["id"], P))
    out = []
    for x in L["structures"]:
        if x["kind"] not in ("door", "window"): continue
        P = shape(x["geometry"]).buffer(0)
        if P.is_empty or P.area < 4: continue
        s1, s2, ln, th = sides(P); j = min(on_ink(s1, dist, tol), on_ink(s2, dist, tol))
        B = P.buffer(0.05 * s); tr = [rid for rid, Q in rooms if Q.exterior.intersects(B) or Q.intersects(P.buffer(1))]
        ok = j >= min_jamb and len(tr) > 0 and all(ot.get(t, 0) >= min_ot for t in tr)
        out.append({"id": x["id"], "kind": x["kind"], "jamb": round(j, 3), "genislik_m": round(ln / s, 2), "kalinlik_m": round(th / s, 2),
                    "odalar": tr, "oda_temiz": all(ot.get(t, 0) >= min_ot for t in tr), "guvenilir": bool(ok)})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip", default="Kisla"); ap.add_argument("--tol-m", type=float, default=0.025); ap.add_argument("--min-jamb", type=float, default=0.8)
    a = ap.parse_args(); aud = json.load(open(OUT / "DENETIM.json", encoding="utf-8")); res = {}; tot = {}
    for i in json.load(open(R / "INDEX.json", encoding="utf-8")):
        fid = i["floor"]
        if not i.get("room"): continue
        o = audit(fid, a.tol_m, aud, a.min_jamb); res[fid] = o
        k = "test" if "kisla" in fid.lower() else "egitim"
        for x in o:
            t = tot.setdefault(k, {}).setdefault(x["kind"], [0, 0, 0, 0]); t[0] += 1; t[1] += x["jamb"] >= a.min_jamb; t[2] += x["oda_temiz"]; t[3] += x["guvenilir"]
        d = [x for x in o if x["kind"] == "door"]; w = [x for x in o if x["kind"] == "window"]
        print(f"{fid[:46]:48} kapi {sum(x['guvenilir'] for x in d):3}/{len(d):3}  pencere {sum(x['guvenilir'] for x in w):3}/{len(w):3}", flush=True)
    print("toplam [hepsi, jamb ok, odalar temiz, guvenilir]:", json.dumps(tot))
    json.dump(res, open(OUT / "ACIKLIK_DENETIM.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
