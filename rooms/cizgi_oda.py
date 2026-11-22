"""Rooms from the boundary-line map of cizgi_ag.py: watershed of the line probability from one seed per room
(the room's inside point, standing in for the printed label) plus the page border as outside.  The room outline is
the watershed ridge, i.e. the middle of the predicted line.  Scored with sinir_skor.room_score (line itself, 3 px).
usage: python cizgi_oda.py --tag c1 --floors <floors> [--seed-r 0.3]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, shape
from skimage.segmentation import watershed

import cizgi_ag as CA  # noqa: E402
import sinir_skor as SS  # noqa: E402


def rooms_from_lines(p, seeds, rr, line_thr=0.5, other=(), edge=None, wall=None):
    """other: inside points of labelled non-room spaces (balcony, common area): their own 'not a room' markers."""
    H, W = p.shape; mk = np.zeros((H, W), np.int32); mk[0, :] = mk[-1, :] = mk[:, 0] = mk[:, -1] = 1; free = p < line_thr
    for k, (x, y) in [(-1, q) for q in other] + list(enumerate(seeds)):
        m = np.zeros((H, W), np.uint8); cv2.circle(m, (int(round(x)), int(round(y))), rr, 1, -1); m = (m > 0) & free
        if not m.any(): m[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)] = True
        mk[m] = k + 2
    if wall is not None:     # predicted line band away from ink (wall body between two faces, door closures): not a room
        mk[(mk == 0) & wall] = 1
    lab = watershed(p.astype(np.float32), mk)
    out = []
    for k in range(len(seeds)):
        m = (lab == k + 2)
        if edge is not None: m = m & (p < line_thr)          # stop at the room-side edge of the line band, not its middle
        m = m.astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs: out.append(None); continue
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        P = Polygon(c + 0.5).buffer(0 if edge is None else edge)
        out.append(P if P.geom_type == "Polygon" else max(P.geoms, key=lambda q: q.area))
    return lab, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="c1"); ap.add_argument("--floors", required=True); ap.add_argument("--seed-r", type=float, default=0.3); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--edge", type=float, default=-1, help="px: outline = room-side edge of the line band pushed out by this much (-1 = band middle)"); ap.add_argument("--wall", type=float, default=0, help="line probability above which an ink-free pixel is wall body (0 = off)"); ap.add_argument("--other", action="store_true", help="balcony / common area label points as not-a-room markers")
    a = ap.parse_args(); out = CA.OUT / a.tag; summ = {}
    for f in a.floors.split(","):
        L = json.load(open(CA.R / f / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]; r = 100.0 / s if s > 130 else 1.0; se = s * r
        p = np.load(out / f"{f}_cizgi.npy").astype(np.float32); g = CA.load(f, 0.95)[0]
        ot = {x["id"]: x["oturma"] for x in CA.AUD[f]["rooms"]}; gts, seeds, ids = [], [], []
        for rm in L["rooms"]:
            P = shape(rm["interior_polygon"]).buffer(0)
            if P.is_empty: continue
            P = P if P.geom_type == "Polygon" else max(P.geoms, key=lambda q: q.area)
            from shapely import affinity
            P = affinity.scale(P, r, r, origin=(0, 0)); gts.append(P); seeds.append((rm["point"]["xy"][0] * r, rm["point"]["xy"][1] * r)); ids.append(rm["id"])
        oth = []; wall = None
        if a.wall > 0:
            _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU); ink = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
            wall = (p > a.wall) & ~ink
        if a.other:
            for x in L["structures"]:
                if x["kind"] in ("balcony", "common"):
                    q = shape(x["geometry"]).buffer(0)
                    if not q.is_empty: c = q.representative_point(); oth.append((c.x * r, c.y * r))
        lab, preds = rooms_from_lines(p, seeds, max(2, int(a.seed_r * se)), a.thr, oth, None if a.edge < 0 else a.edge, wall)
        rows = []
        for k, (gt, pr) in enumerate(zip(gts, preds)):
            sc = SS.room_score(gt, pr, None, 3.0); sc.update(name=L["rooms"][k]["name"], temiz=ot.get(ids[k], 0) >= 0.97); rows.append(sc)
        for grp, R_ in (("hepsi", rows), ("temiz", [x for x in rows if x["temiz"]])):
            summ.setdefault(f, {})[grp] = {"oda": len(R_), "cizgi": round(float(np.mean([x["cizgi"] for x in R_])), 3), "fazla": round(float(np.mean([x["fazla"] for x in R_])), 3),
                                           ">=0.90": int(sum(x["cizgi"] >= 0.9 for x in R_)), "kabul": int(sum(x["kabul"] for x in R_)), "kose": f"{sum(x['kose'] for x in R_)}/{sum(x['kose_gt'] for x in R_)}"}
        print(f, json.dumps(summ[f]), flush=True)
        v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); v = (v * 0.55 + 115).astype(np.uint8)
        for gt, pr, sc in zip(gts, preds, rows):
            cv2.polylines(v, [np.round(np.asarray(gt.exterior.coords)).astype(np.int32)], True, (0, 140, 255), 3)
            if pr is not None: cv2.polylines(v, [np.round(np.asarray(pr.exterior.coords)).astype(np.int32)], True, (200, 60, 0) if sc["cizgi"] >= 0.9 else (0, 0, 220), 1)
            x0, y0 = gt.bounds[:2]; cv2.putText(v, f"{sc['cizgi']:.2f}", (int(x0 + 4), int(y0 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 160), 1)
        cv2.imwrite(str(out / f"{f}_oda.png"), v); json.dump(rows, open(out / f"{f}_oda.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    json.dump(summ, open(out / "ODA_OZET.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
