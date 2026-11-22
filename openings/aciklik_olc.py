"""Doors and windows: label loading, matching and scoring (dev floors, never 015).

  gt_openings(floor)            -> [{"kind": "door"|"window", "poly", "p1", "p2", "width"}]  (p1/p2 = jamb ends of the long axis)
  score(pred, gt)               -> recall/precision per kind, jamb error, width error
  python aciklik_olc.py --floor 019 [--run DIR]   without --run: loss check of the package's opening candidates;
                                                  with --run: scores DIR/SONUC_<floor>.json "openings"
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, shape

import v0_watershed as V0  # noqa: E402


def _axis(poly):
    """Long axis of a thin opening polygon: its two end points (middles of the two short sides)."""
    r = poly.minimum_rotated_rectangle; c = np.array(r.exterior.coords)[:4]
    e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]
    k = int(np.argmin(e))                                             # a short side
    a = (c[k] + c[(k + 1) % 4]) / 2; b = (c[(k + 2) % 4] + c[(k + 3) % 4]) / 2
    return a, b


def gt_openings(floor):
    assert floor != "015"
    lab = json.load(open(V0.B / floor / "LABELS_COMMON.json", encoding="utf-8"))
    out = []
    for s in lab["structures"]:
        if s["kind"] not in ("door", "window"): continue
        P = shape(s["geometry"]); a, b = _axis(P)
        out.append({"kind": s["kind"], "poly": P, "p1": a, "p2": b, "width": float(np.hypot(*(b - a)))})
    return out


def _match(seg, g, tol=10.0):
    """A predicted opening matches a labelled one when its middle lies in the label (widened by tol) and it runs
    along the label's long axis."""
    p1, p2 = np.asarray(seg[0], float), np.asarray(seg[1], float); m = (p1 + p2) / 2
    if not g["poly"].buffer(tol).contains(Point(*m)): return None
    u = (g["p2"] - g["p1"]) / max(g["width"], 1e-6); d = p2 - p1; L = np.hypot(*d)
    if L < 1e-6: return None
    if abs(d @ u) / L < 0.8: return None                               # across the wall, not along it
    e = sorted([np.hypot(*(p1 - g["p1"])) + np.hypot(*(p2 - g["p2"])), np.hypot(*(p1 - g["p2"])) + np.hypot(*(p2 - g["p1"]))])[0] / 2
    return {"jamb_err": float(e), "width_err": float(L - g["width"])}


def score(pred, gt):
    rows = {}
    for kind in ("door", "window"):
        G = [g for g in gt if g["kind"] == kind]; P = [p for p in pred if p.get("kind") == kind]
        used = set(); hits = []
        for gi, g in enumerate(G):
            best = None
            for pi, p in enumerate(P):
                if pi in used: continue
                m = _match((p["p1"], p["p2"]), g)
                if m and (best is None or m["jamb_err"] < best[1]["jamb_err"]): best = (pi, m)
            if best: used.add(best[0]); hits.append(best[1])
        je = [h["jamb_err"] for h in hits]; we = [abs(h["width_err"]) for h in hits]
        rows[kind] = {"labelled": len(G), "predicted": len(P), "found": len(hits), "recall": round(len(hits) / max(1, len(G)), 3),
                      "precision": round(len(hits) / max(1, len(P)), 3), "jamb_err_median_px": round(float(np.median(je)), 1) if je else None,
                      "jamb_within_5px": int(sum(v <= 5 for v in je)), "width_err_median_px": round(float(np.median(we)), 1) if we else None}
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floor", default="019"); ap.add_argument("--run", default=""); a = ap.parse_args()
    gt = gt_openings(a.floor)
    if a.run:
        res = json.load(open(Path(a.run) / f"SONUC_{a.floor}.json", encoding="utf-8-sig"))
        pred = [{"kind": o.get("kind"), "p1": o["p1"], "p2": o["p2"]} for o in res.get("openings", []) if o.get("p1") and o.get("p2")]
        s = score(pred, gt); print(json.dumps(s, ensure_ascii=False))
        json.dump(s, open(Path(a.run) / "SKOR_aciklik.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    else:
        import numpy as np
        from PIL import Image
        import cizgi_yuz as CY, kod_paket as KP
        gray = np.array(Image.open(V0.B / a.floor / "clean.png").convert("L"))
        enc, lines = KP.build(gray)
        for mx in (150, 260):
            cl = CY._closings(lines, max_gap=mx)
            pred = [{"kind": k, "p1": q1, "p2": q2} for q1, q2 in cl for k in ("door", "window")]
            s = score(pred, gt)
            print(f"opening candidates (gaps in collinear lines, max {mx} px): {len(cl)} | doors found {s['door']['found']}/{s['door']['labelled']}, "
                  f"jamb median {s['door']['jamb_err_median_px']} px, within 5 px {s['door']['jamb_within_5px']} | windows found {s['window']['found']}/{s['window']['labelled']}")
