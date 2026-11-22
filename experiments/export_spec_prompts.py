"""Rough polygons from other runs, in the benchmark crop frame, as prompts for SAM 3 (box = their bounding box).

usage: python export_spec_prompts.py  ->  runs/spec_inputs/prompts.json
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
from pathlib import Path

import all_compare as AC

SPEC = Path(__file__).with_name("runs") / "spec_inputs"


def main():
    out = {}
    for name, tag in (("r2s", "SPEC:r2s_ctx"), ("qwen", "qwenq4_noshot132")):
        d = AC.load(tag)
        for k, v in d.items():
            pts = v["pts"]
            if pts and len(pts) >= 3:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                out.setdefault(k, {})[name] = {"poly": pts, "box": [min(xs), min(ys), max(xs), max(ys)]}
    json.dump(out, open(SPEC / "prompts.json", "w"), indent=1)
    print(len(out), "rooms")


if __name__ == "__main__":
    main()
