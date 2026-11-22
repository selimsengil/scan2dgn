"""Door (and window) detection with Qwen's native boxes on a room crop, no thinking. A door is where a room ends
although no wall line is drawn across, so the boxes tell the geometry where to close an opening.
doors(crop_image) -> list of [x1, y1, x2, y2] in 0-1000 of the crop.   usage (picture check): python kapi_bul.py 12,20,13,22"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, re, sys, urllib.request
from pathlib import Path

import qwen_ornekli as QO  # noqa: E402

P_DOOR = ("Detect every door in this floor plan part. A door is an opening in a wall, drawn with a thin door leaf and a "
          "quarter-circle swing arc; it usually has a tag like K1-90/210 next to it. Box each door from one wall end "
          "(jamb) of its opening to the other, including the leaf and the arc. "
          'Output JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "door"}, ...] or [] if there is none.')


def boxes(txt):
    return [[float(v) for v in m] for m in re.findall(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', txt)]


def doors(img, prompt=P_DOOR):
    body = {"messages": [{"role": "user", "content": [QO.img_part(img), {"type": "text", "text": prompt}]}], "temperature": 0.1, "max_tokens": 700,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(QO.URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return boxes(r["choices"][0]["message"].get("content") or "")[:16]


if __name__ == "__main__":
    import time
    import numpy as np
    from PIL import Image, ImageDraw
    import v0_watershed as V0, tek_oda_test as T
    im = Image.open(V0.B / "019" / "clean.png").convert("RGB"); W, H = im.size; gts = V0.load_rooms("019")
    ids = [i for i, g in enumerate(gts) if g["poly"].geom_type == "Polygon"]; tiles = []
    for i in [int(v) for v in sys.argv[1].split(",")]:
        g = gts[i]; box = T.crop_box(g["poly"], W, H, ids.index(i))
        c = im.crop(box); s = 896 / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
        t = time.time(); bx = doors(c); print(f"r{i}: {len(bx)} doors in {time.time() - t:.1f}s", bx)
        d = ImageDraw.Draw(c)
        for b in bx: d.rectangle([b[0] / 1000 * c.width, b[1] / 1000 * c.height, b[2] / 1000 * c.width, b[3] / 1000 * c.height], outline=(255, 0, 0), width=3)
        c.thumbnail((600, 600)); tiles.append(c)
    sh = Image.new("RGB", (sum(t.width for t in tiles) + 10 * len(tiles), max(t.height for t in tiles)), "white"); x = 0
    for t in tiles: sh.paste(t, (x, 0)); x += t.width + 10
    sh.save(T.OUT / "sayfa_kapi.png")
