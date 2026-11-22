"""Location check with a random baseline: an opening is located when a candidate box overlaps it (>= 30 % of the smaller);
the same candidates are then scattered at random 5 times.   python yer_olc.py NAME"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys, numpy as np, test2
from shapely.geometry import Polygon
name = sys.argv[1]; rng = np.random.default_rng(0)
def loc(k, R):
    G, s = test2.gt(k); P = [Polygon(r["box"]).buffer(0) for r in R]; hit = {"door": 0, "window": 0}
    for g in G:
        B = g["poly"].buffer(0.05 * s)
        if any(p.area > 0 and B.intersects(p) and B.intersection(p).area >= 0.3 * min(p.area, B.area) for p in P): hit[g["kind"]] += 1
    return hit
for k in ("malkoc_k2", "kisla_85_1"):
    d = json.load(open(f"SONUC2/{name}__{k}.json", encoding="utf-8")); R = d["okumalar"]
    if not R: print(k, "aday yok"); continue
    c = np.array([np.mean(np.asarray(r["box"]), 0) for r in R]); lo, hi = np.percentile(c, 2, 0), np.percentile(c, 98, 0); f = []
    for _ in range(5):
        f.append(loc(k, [{"box": (np.asarray(r["box"], float) - np.mean(r["box"], 0) + rng.uniform(lo, hi)).tolist()} for r in R]))
    s = d["sonuc"]
    print(f"{k}: aday {len(R)} | yer {loc(k, R)} rastgele kapi~{np.mean([x['door'] for x in f]):.0f} pencere~{np.mean([x['window'] for x in f]):.0f} | genislik {s['genislik']} | tam {s['bulunan']} / {s['n']}")
