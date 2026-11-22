"""Boundary score: how much of the drawn LINE is the same line as the office drawing. Area overlap hides exactly the
mistakes that matter (a boundary on the wrong wall face, a column not gone around), so this scores the line itself.

Per room (GT room matched to the model's room as in SKOR.json, or by room index for oda_parca runs):
  cizgi@T     share of the GT boundary length that has the model's boundary within T px          (recall)
  fazla@T     share of the model's boundary length that lies farther than T px from the GT line  (wrong line drawn)
  kose        GT corners with a model corner within 4 px / all GT corners
  kolon       share of the GT boundary that runs along a labelled column and is matched within T px
  kabul       the room needs no correction: cizgi@T >= 0.97, fazla@T <= 0.03 and every GT corner matched
A room the model did not draw counts as 0 everywhere.

usage: python sinir_skor.py <run dir> [--floor 019] [--tol 3]      (dev floors; the blind floor only for final scoring)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union

import v0_watershed as V0  # noqa: E402


def samples(poly, step=1.0):
    L = poly.exterior.length
    return np.array([poly.exterior.interpolate(s).coords[0] for s in np.arange(0, L, step)])


def corners(poly, tol=1.0):
    """Corner points after dropping points that lie on a straight run."""
    c = np.array(poly.simplify(tol).exterior.coords)[:-1]; out = []
    for i in range(len(c)):
        a, b, d = c[i - 1], c[i], c[(i + 1) % len(c)]
        u, v = b - a, d - b
        if np.hypot(*u) < 1e-6 or np.hypot(*v) < 1e-6: continue
        ang = np.degrees(np.arccos(np.clip(u @ v / (np.hypot(*u) * np.hypot(*v)), -1, 1)))
        if ang > 12: out.append(b)
    return np.array(out).reshape(-1, 2)


def clean(pts):
    try:
        P = Polygon(pts).buffer(0)
        if P.is_empty: return None
        return P if P.geom_type == "Polygon" else max(P.geoms, key=lambda q: q.area)
    except Exception:
        return None


def room_score(gt, pred, columns, tol):
    gs = samples(gt); gc = corners(gt)
    near_col = np.zeros(len(gs), bool)
    if columns is not None and not columns.is_empty:
        near_col = np.array([columns.distance(Point(*p)) <= 2.5 for p in gs])
    if pred is None:
        return {"cizgi": 0.0, "fazla": 1.0, "kose": 0, "kose_gt": len(gc), "kolon_len": int(near_col.sum()), "kolon_ok": 0, "sapma_p95": None, "kabul": False, "cizildi": False}
    dg = np.array([pred.exterior.distance(Point(*p)) for p in gs])
    ps = samples(pred); dp = np.array([gt.exterior.distance(Point(*p)) for p in ps])
    pc = corners(pred)
    hit = int(sum((np.hypot(*(pc - c).T).min() <= 4.0) if len(pc) else False for c in gc))
    cizgi = float((dg <= tol).mean()); fazla = float((dp > tol).mean())
    return {"cizgi": round(cizgi, 3), "fazla": round(fazla, 3), "kose": hit, "kose_gt": len(gc), "kose_model": len(pc),
            "kolon_len": int(near_col.sum()), "kolon_ok": int((dg[near_col] <= tol).sum()), "sapma_p95": round(float(np.percentile(dg, 95)), 1),
            "kabul": bool(cizgi >= 0.97 and fazla <= 0.03 and hit == len(gc)), "cizildi": True}


def load_pairs(run, floor, gts):
    """[(gt index, predicted polygon or None)] for a kat_score run (SKOR.json + SONUC_<floor>.json) or an oda_parca run."""
    if (run / "SKOR.json").exists() and (run / f"SONUC_{floor}.json").exists():
        res = [r for r in json.load(open(run / f"SONUC_{floor}.json", encoding="utf-8"))["rooms"] if r.get("type") != "balcony" and r.get("boundary") and len(r["boundary"]) >= 3]
        sk = json.load(open(run / "SKOR.json", encoding="utf-8"))
        return [(r["dgn"], clean(res[r["pred"]]["boundary"]) if r.get("pred") is not None else None) for r in sk["rooms"]]
    d = json.load(open(run / "SONUC.json", encoding="utf-8"))["rooms"]; ids = [i for i, g in enumerate(gts) if g["poly"].geom_type == "Polygon"]
    return [(ids.index(r["room"]), clean(r["boundary"]) if len(r["boundary"]) >= 3 else None) for r in d]


def score_run(run, floor="019", tol=3.0, quiet=False):
    allg = V0.load_rooms(floor); gts = [g for g in allg if g["poly"].geom_type == "Polygon"]
    lab = json.load(open(V0.B / floor / "LABELS_COMMON.json", encoding="utf-8"))
    cols = [shape(s["geometry"]) for s in lab["structures"] if s["kind"] == "column"]
    columns = unary_union([c.buffer(0) if c.geom_type in ("Polygon", "MultiPolygon") else c for c in cols]) if cols else None
    rows = []
    for gi, pred in load_pairs(Path(run), floor, allg):
        g = gts[gi]; r = room_score(g["poly"], pred, columns, tol); r.update({"room": gi, "name": g["name"]}); rows.append(r)
    n = len(rows); drawn = [r for r in rows if r["cizildi"]]
    S = {"run": str(run), "floor": floor, "tol_px": tol, "rooms": n, "drawn": len(drawn),
         "cizgi": round(float(np.mean([r["cizgi"] for r in rows])), 3), "fazla_drawn": round(float(np.mean([r["fazla"] for r in drawn])), 3) if drawn else None,
         "kose": int(sum(r["kose"] for r in rows)), "kose_gt": int(sum(r["kose_gt"] for r in rows)),
         "kolon_ok": int(sum(r["kolon_ok"] for r in rows)), "kolon_len": int(sum(r["kolon_len"] for r in rows)),
         "kabul": int(sum(r["kabul"] for r in rows)), "cizgi>=0.90": int(sum(r["cizgi"] >= 0.9 for r in rows))}
    S["kose_oran"] = round(S["kose"] / max(1, S["kose_gt"]), 3); S["kolon_oran"] = round(S["kolon_ok"] / max(1, S["kolon_len"]), 3)
    if not quiet:
        for r in rows: print(f"  r{r['room']:<2} {str(r['name'])[:10]:10} cizgi {r['cizgi']:.2f}  fazla {r['fazla']:.2f}  kose {r['kose']}/{r['kose_gt']}  kolon {r['kolon_ok']}/{r['kolon_len']}  p95 {r['sapma_p95']}  {'KABUL' if r['kabul'] else ''}")
    print(S)
    json.dump({"summary": S, "rooms": rows}, open(Path(run) / "SINIR_SKOR.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return S, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--floor", default="019"); ap.add_argument("--tol", type=float, default=3.0); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(); score_run(a.run, a.floor, a.tol, a.quiet)
