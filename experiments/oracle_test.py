"""Where does the room loss come from? Swap one part at a time for the user's labels.

  A  tools only (walls, openings, OCR seeds)                         = production
  B  + the labelled doors and windows as barriers (openings closed exactly), tool openings off
  C  + the labelled walls and columns as barriers too
  D  + the DGN rooms' own label points as seeds (no OCR misses or duplicates)

Rooms are scored against the user's DGN rooms (IoU per room).
usage: python oracle_test.py [--floors 001,002,...]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
from pathlib import Path

import numpy as np
from shapely.geometry import shape as shp

import araclar as A
import v0_watershed as V0


def label_mask(floor: str, kinds: set, hw) -> np.ndarray:
    L = json.load(open(V0.B / floor / "LABELS_COMMON.json", encoding="utf-8"))
    m = np.zeros(hw, bool)
    for st in L["structures"]:
        if st.get("kind") in kinds:
            try:
                m |= V0.poly_mask(shp(st["geometry"]), hw)
            except Exception:
                pass
    return m


def summary(sc: dict) -> dict:
    return {k: sc[k] for k in ("rooms", "ge95", "ge90", "ge80", "mean")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="001,002,010,013,015,016,019")
    ap.add_argument("--out", default=str(Path(__file__).with_name("out_oracle")))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    tot = {v: {"rooms": 0, "ge95": 0, "ge90": 0, "ge80": 0} for v in "ABCD"}
    res = {}
    for f in a.floors.split(","):
        k = A.Kat(f)
        hw = (k.H, k.W)
        doors = label_mask(f, {"door", "window"}, hw)
        walls = label_mask(f, {"wall", "column"}, hw)
        ocr_labels, closed0 = k.labels, set(k.closed)
        r = {}
        # A
        k.extra = None; k.closed = set(closed0); k.labels = ocr_labels
        k.segment(); r["A"] = k.score()
        # B: exact openings, the tools' own closing lines off
        k.extra = doors; k.closed = set()
        k.segment(); r["B"] = k.score()
        # C
        k.extra = doors | walls
        k.segment(); r["C"] = k.score()
        # D: the DGN rooms' label points as seeds
        gts = V0.load_rooms(f)
        k.labels = [{"name": g["name"], "area_m2": g["area_m2"], "xy": g["xy"], "box": [g["xy"][0], g["xy"][1], g["xy"][0], g["xy"][1]]} for g in gts]
        k.segment(); r["D"] = k.score()
        k.render(out / f"{f}_D.png", gt=True)
        res[f] = {v: summary(r[v]) for v in "ABCD"}
        res[f]["D_per_room"] = r["D"]["per_room"]
        print(f, {v: (res[f][v]["ge95"], res[f][v]["ge90"], res[f][v]["mean"]) for v in "ABCD"}, flush=True)
        for v in "ABCD":
            for kk in ("rooms", "ge95", "ge90", "ge80"):
                tot[v][kk] += res[f][v][kk]
        k.labels = ocr_labels
    print("TOTAL (rooms, >=0.95, >=0.90, >=0.80):")
    for v in "ABCD":
        print(" ", v, tot[v])
    json.dump({"total": tot, "floors": res}, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
