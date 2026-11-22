"""Classify the errors of a bench132 run: for every room compare the predicted polygon with the DGN reference.

kind of error (snapped polygon P vs reference R):
  ok        IoU >= 0.90
  outer     P covers R but is much bigger (wall outer face / spilled into neighbour)
  inner     R covers P but P is much smaller (only part of the room, or label box)
  shifted   similar area but in the wrong place
  shape     similar area and place, wrong corners (notches/columns missed or invented)
  invalid   no polygon

usage: python error_analysis.py TAG      (runs/b132_<TAG>/RESULT.json) -> prints tables, writes ERRORS.json
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from shapely.geometry import Polygon

from compare_runs import kind as room_kind

R = Path(__file__).with_name("runs")


def poly(pts):
    p = Polygon(pts)
    return p if p.is_valid else p.buffer(0)


def classify(r):
    pts = r.get("snapped") or r.get("points")
    if not pts or len(pts) < 3:
        return "invalid", {}
    P, G = poly(pts), poly(r["reference"])
    if P.area <= 0:
        return "invalid", {}
    inter = P.intersection(G).area
    iou = inter / P.union(G).area
    ratio = P.area / G.area
    cover_ref = inter / G.area      # how much of the reference P covers
    cover_pred = inter / P.area     # how much of P lies inside the reference
    m = {"IoU": round(iou, 3), "area_ratio": round(ratio, 2), "ref_covered": round(cover_ref, 2),
         "pred_inside": round(cover_pred, 2), "n_pred": len(pts), "n_ref": len(r["reference"])}
    if iou >= 0.90:
        return "ok", m
    if cover_ref >= 0.85 and ratio > 1.15:
        return "outer", m
    if cover_pred >= 0.85 and ratio < 0.87:
        return "inner", m
    if iou < 0.5 and 0.6 < ratio < 1.6:
        return "shifted", m
    return "shape", m


def main():
    tag = sys.argv[1]
    d = json.load(open(R / f"b132_{tag}" / "RESULT.json", encoding="utf-8"))
    out = {}
    for k, r in d.items():
        c, m = classify(r)
        out[k] = {"name": r["name"], "floor": r["floor"], "type": room_kind(r["name"]), "error": c, **m}
    json.dump(out, open(R / f"b132_{tag}" / "ERRORS.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("overall:", dict(Counter(v["error"] for v in out.values())))
    by = defaultdict(Counter)
    for v in out.values():
        by[v["type"]][v["error"]] += 1
    cols = ["ok", "shape", "outer", "inner", "shifted", "invalid"]
    print(f"\n{'type':10}{'n':>4}" + "".join(f"{c:>9}" for c in cols))
    for t, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"{t:10}{sum(c.values()):>4}" + "".join(f"{c[x]:>9}" for x in cols))
    byf = defaultdict(Counter)
    for v in out.values():
        byf[v["floor"]][v["error"]] += 1
    print(f"\n{'floor':10}{'n':>4}" + "".join(f"{c:>9}" for c in cols))
    for f, c in sorted(byf.items()):
        print(f"{f:10}{sum(c.values()):>4}" + "".join(f"{c[x]:>9}" for x in cols))
    print("\ncorner count (pred vs ref) for non-ok 'shape' rooms:")
    for k, v in sorted(out.items(), key=lambda kv: kv[1].get("IoU", 0)):
        if v["error"] != "ok":
            print(f"  {k} {v['name'][:22]:22} {v['error']:8} IoU {v.get('IoU', 0):.3f} area x{v.get('area_ratio', 0)}"
                  f" corners {v.get('n_pred', 0)}/{v.get('n_ref', 0)}")


if __name__ == "__main__":
    main()
