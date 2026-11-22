"""Mosaic of the worst rooms of a bench132 run: crop, DGN room (green), model (orange), snapped (magenta).

usage: python b132_mosaic.py RUN_TAG [n] [floor]
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

run = Path(__file__).with_name("runs") / ("b132_" + sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
floor = sys.argv[3] if len(sys.argv) > 3 else None
res = json.load(open(run / "RESULT.json", encoding="utf-8"))
items = [(k, r) for k, r in res.items() if not floor or r["floor"] == floor]
items.sort(key=lambda kr: kr[1]["snap"].get("IoU", 0) if kr[1]["snap"].get("valid", True) else 0)
tiles = []
for k, r in items[:n]:
    im = Image.open(run / f"{k}.png").convert("RGB")
    s = 380 / max(im.size)
    im = im.resize((int(im.width * s), int(im.height * s)))
    d = ImageDraw.Draw(im)
    for key, col in (("reference", (0, 170, 60)), ("points", (255, 130, 0)), ("snapped", (200, 0, 200))):
        p = r.get(key) or []
        if len(p) >= 3:
            q = [(x * s, y * s) for x, y in p]
            d.line(q + q[:1], fill=col, width=2)
    c = Image.new("RGB", (390, 410), "white"); c.paste(im, (0, 0))
    ImageDraw.Draw(c).text((4, 392), f"{k} {r['name'][:26]} IoU {r['snap'].get('IoU')}", fill=(0, 0, 0))
    tiles.append(np.asarray(c))
while len(tiles) % 4:
    tiles.append(np.full_like(tiles[0], 255))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
out = run / f"worst{('_' + floor) if floor else ''}.png"
Image.fromarray(np.vstack(rows)).save(out)
print(out)
