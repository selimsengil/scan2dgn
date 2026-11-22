"""NorthCUA-Mini (a computer-use model trained for pixel-precise clicking) traces room corners by clicking.

For every room crop (the same 22 + 7 comparison rooms, same frame as all other runs) the model sees the crop with
the red label dot and is asked to click the room's inner corners one by one, clockwise. After every click the crop
is shown again with the corners clicked so far (numbered dots + the path), like a screenshot after an action.
It stops by clicking the first corner again or calling `done`. Coordinates are the model's native 0-1000 frame
of the shown image; they are converted to crop pixels.

usage: python mini_trace.py [--url http://127.0.0.1:8092/v1/chat/completions] [--max-corners 24] [--tag mini]
       -> runs/spec_inputs/<tag>_trace.json   (then all_compare.py picks it up as SPEC:<tag>_trace)
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import base64
import io
import json
import math
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SPEC = Path(__file__).with_name("runs") / "spec_inputs"
SHOW = 1000  # the longer side of the image the model sees

SYSTEM = ("You are a computer-use agent operating a CAD tracing tool. The screenshot shows a crop of a scanned floor "
          "plan. Coordinates for every action are normalized 0-1000 over the screenshot (0,0 top-left, 1000,1000 "
          "bottom-right). Use the click tool exactly on the target point. Rules of the drawing office: a room's "
          "boundary runs on the INNER face of its walls (the wall line facing into the room, not the outer line or "
          "the middle of the wall); door and window openings are closed straight across along that inner face; "
          "columns and wall returns that project into the room are followed exactly (each shoulder is a corner); "
          "furniture, text, dimension lines, stamps and hatching are not boundaries.")

TASK = ("Trace the room that contains the RED DOT. Click its inner-boundary corners one at a time, going clockwise; "
        "each click is one corner where the boundary changes direction. Corners clicked so far are shown as "
        "numbered blue dots joined by a blue line. Click the next corner. When all corners are clicked, click "
        "corner 1 again (or call done).")

TOOLS = [{"type": "function", "function": {"name": "click", "description": "Click at a point (normalized 0-1000).",
          "parameters": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}}},
         {"type": "function", "function": {"name": "done", "description": "All corners have been clicked.",
          "parameters": {"type": "object", "properties": {}}}}]


def b64(img):
    b = io.BytesIO(); img.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def shown(crop, label, pts_px, s):
    im = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.Resampling.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im)
    r = 7
    d.ellipse((label[0] * s - r, label[1] * s - r, label[0] * s + r, label[1] * s + r), fill=(230, 0, 0))
    q = [(x * s, y * s) for x, y in pts_px]
    if len(q) >= 2:
        d.line(q, fill=(0, 80, 255), width=2)
    for i, (x, y) in enumerate(q, 1):
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 80, 255))
        d.text((x + 6, y - 12), str(i), fill=(0, 40, 200), font=ImageFont.load_default())
    return im


def call(url, msgs):
    body = {"messages": msgs, "tools": TOOLS, "max_tokens": 512, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def parse_click(m):
    for tc in m.get("tool_calls") or []:
        f = tc.get("function", {})
        if f.get("name") == "done":
            return "done"
        if f.get("name") == "click":
            try:
                a = json.loads(f.get("arguments") or "{}")
                return float(a["x"]), float(a["y"])
            except Exception:
                pass
    txt = m.get("content") or ""   # some builds return the call as text
    import re
    mm = re.search(r"click\D*?(\d+(?:\.\d+)?)\D+(\d+(?:\.\d+)?)", txt)
    if mm:
        return float(mm.group(1)), float(mm.group(2))
    return "done" if "done" in txt.lower() else None


def trace(url, crop, label, max_corners):
    s = SHOW / max(crop.size)
    W, H = crop.size
    pts, log = [], []
    for k in range(max_corners + 1):
        im = shown(crop, label, pts, s)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(im)}},
                                             {"type": "text", "text": TASK + f" ({len(pts)} corners clicked so far.)"}]}]
        r = call(url, msgs)
        c = parse_click(r["choices"][0]["message"])
        log.append(c)
        if c is None or c == "done":
            break
        x, y = c[0] / 1000 * im.width / s, c[1] / 1000 * im.height / s   # -> crop pixels
        if len(pts) >= 3 and math.hypot(x - pts[0][0], y - pts[0][1]) < 0.02 * max(W, H):
            break   # closed the ring
        if pts and math.hypot(x - pts[-1][0], y - pts[-1][1]) < 2:
            break   # repeated the same corner
        pts.append([x, y])
    return pts, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8092/v1/chat/completions")
    ap.add_argument("--max-corners", dest="max_corners", type=int, default=24)
    ap.add_argument("--tag", default="mini")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    meta = json.load(open(SPEC / "rooms.json", encoding="utf-8"))
    out_f = SPEC / f"{a.tag}_trace.json"
    out = json.load(open(out_f)) if out_f.exists() else {}
    for key, m in meta.items():
        if key in out or (a.only and key not in a.only.split(",")):
            continue
        crop = Image.open(SPEC / f"{key}_crop.png").convert("RGB")
        t0 = time.time()
        pts, log = trace(a.url, crop, m["crop"]["label"], a.max_corners)
        out[key] = {"pick": pts, "log": log, "seconds": round(time.time() - t0, 1)}
        print(key, m["name"][:20], len(pts), "corners", f"{time.time() - t0:.0f} s", flush=True)
        json.dump(out, open(out_f, "w"), indent=1)


if __name__ == "__main__":
    main()
