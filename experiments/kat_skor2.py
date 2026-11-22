"""Scores a whole-floor answer (SONUC_<floor>.json: rooms[{name, type, boundary:[[x,y],..]}]) against the DGN rooms of
KAT_ROOT/<floor>/LABELS_COMMON.json without the deleted egitim_verisi helpers.  Each DGN room gets the answer polygon
with the highest IoU (one-to-one, Hungarian); balconies in the answer are left out (the DGN room set has none).
Per room: IoU and the boundary score of sinir_skor (cizgi / fazla / corners / kabul, tol 3 px).
usage: python kat_skor2.py <run dir> [--floor 015]   (prints one line, writes SKOR2.json in the run dir)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import sinir_skor as SS  # noqa: E402
import v0_watershed as V0  # noqa: E402


def load_answer(p):
    b = Path(p).read_bytes()
    for enc in ("utf-8-sig", "cp1254"):
        try: return json.loads(b.decode(enc))
        except UnicodeDecodeError: pass
    return json.loads(b.decode("utf-8", "replace"))


def score(run, floor="015"):
    run = Path(run); ans = load_answer(run / f"SONUC_{floor}.json")
    gts = [g for g in V0.load_rooms(floor) if g["poly"].geom_type == "Polygon"]
    preds = [SS.clean(r["boundary"]) for r in ans.get("rooms", []) if r.get("type") != "balcony" and r.get("boundary") and len(r["boundary"]) >= 3]
    preds = [p for p in preds if p is not None]
    M = np.zeros((len(gts), max(1, len(preds))))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            u = g["poly"].union(p).area; M[i, j] = g["poly"].intersection(p).area / u if u else 0
    gi, pj = linear_sum_assignment(-M) if preds else ([], [])
    best = {i: j for i, j in zip(gi, pj) if M[i, j] > 0.3}
    rows = []
    for i, g in enumerate(gts):
        p = preds[best[i]] if i in best else None; s = SS.room_score(g["poly"], p, None, 3.0)
        s.update(name=g["name"], iou=round(float(M[i, best[i]]), 3) if i in best else 0.0); rows.append(s)
    S = {"rooms": len(gts), "answer_polygons": len(preds), "found": len(best), "iou": round(float(np.mean([r["iou"] for r in rows])), 3),
         "iou>=0.90": sum(r["iou"] >= 0.9 for r in rows), "cizgi": round(float(np.mean([r["cizgi"] for r in rows])), 3),
         "fazla": round(float(np.mean([r["fazla"] for r in rows])), 3), "cizgi>=0.90": sum(r["cizgi"] >= 0.9 for r in rows),
         "kabul": sum(r["kabul"] for r in rows), "kose": f"{sum(r['kose'] for r in rows)}/{sum(r['kose_gt'] for r in rows)}"}
    json.dump({"summary": S, "rooms": rows}, open(run / "SKOR2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return S


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--floor", default="015"); a = ap.parse_args()
    for r in a.runs:
        try: S = score(r, a.floor); print(f"{Path(r).name:30} " + " ".join(f"{k} {v}" for k, v in S.items()), flush=True)
        except Exception as e: print(Path(r).name, "ERR", repr(e)[:200])
