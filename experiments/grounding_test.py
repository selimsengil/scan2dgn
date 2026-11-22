"""Which coordinate frame does local Qwen use natively for grounding? (dev floors only, never 015)
Crops a region around GT rooms, asks for each room's bbox in Qwen's native format, compares with GT in three frames:
0-1000 relative, pixels of the sent image, and 0-1 relative.
usage: python grounding_test.py --floor 019 --n 6"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import v0_watershed as V0  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--floor", default="019")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--port", type=int, default=8091)
ap.add_argument("--think", action="store_true")
a = ap.parse_args()

img = Image.open(V0.B / a.floor / "clean.png").convert("RGB")
gts = [g for g in V0.load_rooms(a.floor) if g["poly"].geom_type == "Polygon" and g.get("name")]
gts.sort(key=lambda g: -g["poly"].area)
rng = np.random.default_rng(0)
picks = [gts[i] for i in rng.choice(len(gts), size=min(a.n, len(gts)), replace=False)]


def iou(b1, b2):
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])); iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    u = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


res = {"rel1000": [], "pixel": [], "rel1": []}
for g in picks:
    x0, y0, x1, y1 = g["poly"].bounds
    m = 0.6 * max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, x0 - m)), int(max(0, y0 - m)); cx1, cy1 = int(min(img.width, x1 + m)), int(min(img.height, y1 + m))
    crop = img.crop((cx0, cy0, cx1, cy1))
    buf = io.BytesIO(); crop.save(buf, "PNG")
    q = (f"This is part of an architectural floor plan. Locate the room labelled '{g['name']}' "
         f"(the area enclosed by its walls, inner wall faces). Output its bounding box in JSON: "
         f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "{g["name"]}"}}]')
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
        {"type": "text", "text": q}]}], "max_tokens": 3000 if a.think else 300, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": a.think}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                                                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    txt = r["choices"][0]["message"].get("content") or ""
    mm = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', txt)
    gt = [x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]   # in crop pixels
    W, H = crop.size
    if not mm:
        print(f"{g['name']}: no bbox  ({txt[:120]!r})"); continue
    b = [float(v) for v in mm.group(1).split(",")[:4]]
    frames = {"rel1000": [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000],
              "pixel": b, "rel1": [b[0] * W, b[1] * H, b[2] * W, b[3] * H]}
    line = f"{g['name']:12s} crop {W}x{H} raw {b} ({time.time() - t:.0f}s)"
    for k, fb in frames.items():
        v = iou(fb, gt); res[k].append(v); line += f" | {k} IoU {v:.2f}"
    print(line, flush=True)
print({k: round(float(np.mean(v)), 3) if v else None for k, v in res.items()})
