"""Raster2Seq's polygon (right topology, coarse 256-px edges) snapped to the wall faces with snap.py, the same
snap every other run uses. Writes runs/spec_inputs/r2s_ctx_snap.json (crop frame) for all_compare.

usage: python spec_snap.py
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
from pathlib import Path

import all_compare as AC
import bench132 as B
import snap

SPEC = Path(__file__).with_name("runs") / "spec_inputs"


def main():
    src = AC.load("SPEC:r2s_ctx")
    crops = {k: c for k, f, n, c, r, l in B.rooms(1400) if k in src}
    out = {}
    for k, v in src.items():
        pts = v["pts"]
        sn = snap.snap_polygon(crops[k], pts) if len(pts) >= 3 else []
        out[k] = {"pick": sn or pts}
    json.dump(out, open(SPEC / "r2s_ctx_snap.json", "w"), indent=1)
    print(len(out), "rooms")


if __name__ == "__main__":
    main()
