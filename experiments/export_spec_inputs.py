"""Export the comparison rooms for the specialist models (they run in another Python env).

For every room: the benchmark crop (same frame as all other tests) and a wider context window (the room's
bbox x3, like a small floor plan), with the reference polygon and label point in each frame.

usage: python export_spec_inputs.py  ->  runs/spec_inputs/{key}_crop.png, {key}_ctx.png, rooms.json
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
from pathlib import Path

from PIL import Image

import bench132 as B
import v0_watershed as V0

R = Path(__file__).with_name("runs")
OUT = R / "spec_inputs"
SUB22 = sorted(json.load(open(R / "b132_ds500_sub22" / "RESULT.json", encoding="utf-8")))
THINK = ["013_05", "016_01", "019_21", "001_26", "019_07", "015_07", "015_06"]
KEYS = list(dict.fromkeys(SUB22 + THINK))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for key, f, name, crop, ref, label in B.rooms(1400):
        if key not in KEYS:
            continue
        crop.save(OUT / f"{key}_crop.png")
        # context window in floor pixels: room bbox x3 around its centre
        img = Image.open(V0.B / f / "clean.png").convert("RGB")
        g = V0.load_rooms(f)[int(key.split("_")[1])]
        x0, y0, x1, y1 = g["poly"].bounds
        cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, 1.5 * max(x1 - x0, y1 - y0)
        bx = [max(0, int(cx - half)), max(0, int(cy - half)), min(img.width, int(cx + half)), min(img.height, int(cy + half))]
        ctx = img.crop(bx)
        s = min(1.0, 1400 / max(ctx.size))
        if s < 1:
            ctx = ctx.resize((int(ctx.width * s), int(ctx.height * s)), Image.Resampling.LANCZOS)
        ctx.save(OUT / f"{key}_ctx.png")
        ref_ctx = [[(x - bx[0]) * s, (y - bx[1]) * s] for x, y in list(g["poly"].exterior.coords)[:-1]]
        lab_ctx = [(g["xy"][0] - bx[0]) * s, (g["xy"][1] - bx[1]) * s]
        meta[key] = {"name": name, "floor": f, "crop": {"size": list(crop.size), "ref": ref, "label": label},
                     "ctx": {"size": list(ctx.size), "ref": ref_ctx, "label": lab_ctx, "box": bx, "scale": s}}
    json.dump(meta, open(OUT / "rooms.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(meta), "rooms ->", OUT)


if __name__ == "__main__":
    main()
