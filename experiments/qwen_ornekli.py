"""Single-room test, teaching by example (never 015 for development): before the question the model sees solved rooms
from OTHER floors (crop + the correct corner list), then draws the asked room the same way. No shape tool.
usage: python qwen_ornekli.py --rooms 11,18,7,21,14 --tag o1 [--think 800] [--examples 017:18,013:0,017:15]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, base64, io, json, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import v0_watershed as V0  # noqa: E402
import tek_oda_test as T  # noqa: E402
import qwen_izgara as QI  # noqa: E402

URL = "http://127.0.0.1:8091/v1/chat/completions"
Q = 'Give the boundary of the room marked with the red dot (its label: "{name}"{m2}). JSON only: {{"points": [[x, y], ...]}}, corners clockwise, coordinates 0-1000 relative to this image.'


def mark(c, box, xy, s):
    """The asked room's label point as a red dot on the crop, so rooms with the same name cannot be confused."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(c); x, y = (xy[0] - box[0]) * s, (xy[1] - box[1]) * s
    d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 255, 255)); d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(230, 0, 0))
    return c
SYS = ("You trace room boundaries on scanned architectural floor plans, the way this office draws them.\n" + QI.RULES +
       "\nYou will first see solved examples: study where their corners sit (on the inner wall line, around columns, along slanted walls). "
       "Then answer the new room in exactly the same way and with the same care for detail.")


def img_part(im):
    b = io.BytesIO(); im.save(b, "PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}}


def crop(floor, idx, k, side):
    im = Image.open(V0.B / floor / "clean.png").convert("RGB"); W, H = im.size
    g = V0.load_rooms(floor)[idx]; box = T.crop_box(g["poly"], W, H, k)
    c = im.crop(box); s = side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    c = mark(c, box, g["xy"], s)
    m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
    return im, g, box, c, m2


def example_msgs(spec, side):
    out = []
    for k, e in enumerate(spec.split(",")):
        fl, idx = e.split(":"); assert fl != "015"
        _, g, box, c, m2 = crop(fl, int(idx), k + 1, side)
        P = g["poly"]; pts = list(P.exterior.coords)[:-1]
        if P.exterior.is_ccw: pts = pts[::-1]          # image frame (y down): this makes it clockwise on screen
        rel = [[int(round((x - box[0]) / (box[2] - box[0]) * 1000)), int(round((y - box[1]) / (box[3] - box[1]) * 1000))] for x, y in pts]
        out.append({"role": "user", "content": [img_part(c), {"type": "text", "text": f"Example {k + 1}. " + Q.format(name=g["name"], m2=m2)}]})
        out.append({"role": "assistant", "content": json.dumps({"points": rel})})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="019"); ap.add_argument("--rooms", default="11,18,7,21,14"); ap.add_argument("--tag", default="o1")
    ap.add_argument("--think", type=int, default=0); ap.add_argument("--examples", default="017:18,013:0,017:15")
    ap.add_argument("--ex-side", type=int, default=640); ap.add_argument("--side", type=int, default=896)
    a = ap.parse_args(); assert a.floor != "015"
    ex = example_msgs(a.examples, a.ex_side) if a.examples else []
    rows = []; tiles = []
    for k, i in enumerate(int(v) for v in a.rooms.split(",")):
        im, g, box, c, m2 = crop(a.floor, i, k, a.side)
        msgs = [{"role": "system", "content": SYS}] + ex + [{"role": "user", "content": [img_part(c), {"type": "text", "text": "Now the new room. " + Q.format(name=g["name"], m2=m2)}]}]
        body = {"messages": msgs, "temperature": 0.2, "max_tokens": 1200 + a.think, "chat_template_kwargs": {"enable_thinking": a.think > 0}}
        if a.think > 0: body["thinking_budget_tokens"] = a.think
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
        sec = time.time() - t; txt = r["choices"][0]["message"].get("content") or ""
        pts = T.parse_points(txt)[:60]
        pred = [[box[0] + x / 1000 * (box[2] - box[0]), box[1] + y / 1000 * (box[3] - box[1])] for x, y in pts]
        sc = T.score(pred, g["poly"]) if len(pred) >= 3 else {"iou": 0.0}
        rows.append({"room": i, "sec": round(sec, 1), "in": r["usage"]["prompt_tokens"], "out": r["usage"]["completion_tokens"], **sc}); print(rows[-1], flush=True)
        tiles.append(T.tile(im, box, g["poly"], pred if len(pred) >= 3 else None, f"r{i} IoU {sc.get('iou')} kose {sc.get('corners_hit6')}/{sc.get('gt_corners')} {sec:.0f}s"))
        json.dump({"row": rows[-1], "raw": txt, "pred": pred}, open(T.OUT / f"ornekli_{a.tag}_r{i}.json", "w", encoding="utf-8"))
    n = (len(tiles) + 2) // 3; cw = max(t.width for t in tiles) + 10; ch = max(t.height for t in tiles) + 10
    sh = Image.new("RGB", (cw * 3, ch * n), "white")
    for j, t in enumerate(tiles): sh.paste(t, ((j % 3) * cw, (j // 3) * ch))
    sh.save(T.OUT / f"sayfa_ornekli_{a.tag}.png")
    print("mean IoU", round(float(np.mean([r["iou"] for r in rows])), 3), "| mean sec", round(float(np.mean([r["sec"] for r in rows])), 1))
