"""Dev test of qwen_araclar auto calibration with GT seeds (never 015).
usage: python kalib_test.py --floors "019,013" """
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import qwen_araclar as A  # noqa: E402
import v0_watershed as V0  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--floors", default="019")
a = ap.parse_args()
for f in a.floors.split(","):
    gts = [g for g in V0.load_rooms(f) if g["poly"].geom_type == "Polygon" and (g.get("area_m2") or -1) > 0]
    true = float(np.median([np.sqrt(g["poly"].area / (g["area_m2"] * 1e4)) for g in gts]))
    ctx = {"image": V0.B / f / "clean.png", "out": Path("."), "state": {}}
    items = [{"id": f"r{i}", "name": g.get("name", "ODA"), "m2": g["area_m2"],
              "x": int(g["poly"].representative_point.x if False else g["poly"].representative_point().x),
              "y": int(g["poly"].representative_point().y)} for i, g in enumerate(gts)]
    A.handle("rooms", {"items": items}, ctx)
    txt, _, _ = A.handle("calibrate", {"mode": "auto"}, ctx)
    print(f"{f}: true {true:.4f} | {txt}")
    t2, _, _ = A.handle("walls", {}, ctx)
    flags = [ln.rsplit("|", 1)[-1].strip() for ln in t2.splitlines()[1:] if ln.count("|") >= 6]
    print("   flags:", {k.split(" - ")[0]: sum(x.split(" - ")[0] == k.split(" - ")[0] for x in flags) for k in set(flags)})
    t3, _, _ = A.handle("calibrate", {"mode": "dimension", "cm": 592, "x1": 0, "y1": 0, "x2": 280, "y2": 0}, ctx)
    print("   wrong dim ->", t3)
    t4, _, _ = A.handle("walls", {}, ctx)
    print("   walls tail ->", "\n".join(t4.splitlines()[-2:]))
