"""Compare bench132 runs: overall, by room type and by floor (snapped IoU).

usage: python compare_runs.py TAG [TAG ...]     (tags as in runs/b132_<tag>)
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).with_name("runs")
TYPES = ("N.O.A", "HOL", "SIĞINAK", "BANYO", "TUVALET", "WC", "MUTFAK", "SALON", "ODA")


def kind(name):
    n = name.upper()
    return next((k for k in TYPES if k in n), "DİĞER")


def iou(r):
    x = r.get("snap", {})
    return x.get("IoU", 0.0) if x.get("valid", True) and "IoU" in x else 0.0


def main():
    tags = sys.argv[1:]
    runs = {t: json.load(open(R / f"b132_{t}" / "RESULT.json", encoding="utf-8")) for t in tags}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    print(f"{len(keys)} rooms in all runs\n")
    w = max(len(t) for t in tags) + 2
    print(f"{'':16}" + "".join(f"{t:>{w}}" for t in tags))
    rows = [("mean IoU", lambda ks, r: sum(iou(r[k]) for k in ks) / len(ks)),
            (">=0.95", lambda ks, r: sum(iou(r[k]) >= .95 for k in ks)),
            (">=0.90", lambda ks, r: sum(iou(r[k]) >= .90 for k in ks)),
            ("s/oda", lambda ks, r: sum(r[k]["seconds"] for k in ks) / len(ks))]
    for label, f in rows:
        print(f"{label:16}" + "".join(f"{f(keys, runs[t]):>{w}.3f}" if label in ('mean IoU', 's/oda') else f"{f(keys, runs[t]):>{w}d}" for t in tags))
    first = runs[tags[0]]
    for title, group in (("tip", lambda k: kind(first[k]["name"])), ("kat", lambda k: first[k]["floor"])):
        g = defaultdict(list)
        for k in keys:
            g[group(k)].append(k)
        print(f"\n{title:10}{'oda':>6}" + "".join(f"{t:>{w}}" for t in tags))
        for name, ks in sorted(g.items(), key=lambda kv: -len(kv[1])):
            print(f"{name:10}{len(ks):>6}" + "".join(f"{sum(iou(runs[t][k]) for k in ks) / len(ks):>{w}.3f}" for t in tags))


if __name__ == "__main__":
    main()
