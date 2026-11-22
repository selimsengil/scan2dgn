"""Single-room test of the 'Qwen decides, the tool measures' design on GT-known hard rooms of a dev floor (never 015).
Stage A: Qwen sees one enlarged room crop and returns the inner boundary polygon in 0-1000 crop coordinates.
usage: python tek_oda_test.py --rooms 11,18,7,21,14 --think 800 --tag a1"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, base64, io, json, re, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point

import v0_watershed as V0  # noqa: E402

URL = "http://127.0.0.1:8091/v1/chat/completions"
SCHEMA = True
OUT = config.KT / "tek_oda"

PROMPT_A = """This is an enlarged part of an architectural floor plan. Trace the boundary of ONE room: the room that contains the label "{name}"{m2}.

The boundary is the INNER face of the room's walls: the line where the white floor of this room meets the wall.
Rules:
- Walls are the thick double-line bands (often with hatched columns). Follow the wall line that faces this room.
- Columns and wall jogs that stick into the room change the boundary: go around them, each one adds corners.
- Slanted walls stay slanted: give their two real end points, do not straighten them into horizontal or vertical.
- Across a door or an open passage, continue the wall line straight to the other side of the opening.
- Ignore everything that is not a wall: furniture, kitchen counters, sinks, toilets, floor tiles, door swings, dimension lines and their numbers, level marks, stamps, text.
- Windows are part of the wall: the boundary runs along the inner wall line, not the outer one.

Answer with JSON only: {{"points": [[x, y], ...]}} with the corners in clockwise order, coordinates 0-1000 relative to this image (x to the right, y down). One point per corner, no repeated closing point."""


def ask(img, prompt, think):
    b = io.BytesIO(); img.save(b, "PNG")
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}},
        {"type": "text", "text": prompt}]}],
        "temperature": 0.2, "max_tokens": 1500 + think,
        "chat_template_kwargs": {"enable_thinking": think > 0}}
    if think > 0:
        body["thinking_budget_tokens"] = think
    if SCHEMA:   # caps the corner count, so a repeating answer cannot run away
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "room", "strict": True, "schema": {
            "type": "object", "properties": {"points": {"type": "array", "minItems": 3, "maxItems": 40, "items": {
                "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}}}, "required": ["points"]}}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    m = r["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", time.time() - t, r.get("usage", {})


def parse_points(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return [[float(p[0]), float(p[1])] for p in json.loads(m.group(0))["points"]]
    except Exception:
        return [[float(a), float(b)] for a, b in re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", txt)]


def crop_box(poly, W, H, k):
    """A coarse box as a floor-level pass would give it: the room box widened unevenly."""
    x0, y0, x1, y1 = poly.bounds; w, h = x1 - x0, y1 - y0
    f = [(0.22, 0.14, 0.16, 0.24), (0.15, 0.25, 0.23, 0.13)][k % 2]
    return (max(0, int(x0 - f[0] * w - 25)), max(0, int(y0 - f[1] * h - 25)), min(W, int(x1 + f[2] * w + 25)), min(H, int(y1 + f[3] * h + 25)))


def score(pred, gt):
    """IoU, GT corners matched within 6 px, mean/max distance of the GT boundary to the predicted boundary (floor px)."""
    try:
        P = Polygon(pred).buffer(0)
        if P.geom_type != "Polygon": P = max((q for q in P.geoms if q.geom_type == "Polygon"), key=lambda q: q.area)
        iou = P.intersection(gt).area / P.union(gt).area
    except Exception:
        return {"iou": 0.0}
    gc = list(gt.exterior.coords)[:-1]
    d = [min(np.hypot(x - a, y - b) for a, b in pred) for x, y in gc]
    L = gt.exterior.length
    bd = [P.exterior.distance(gt.exterior.interpolate(s)) for s in np.arange(0, L, 4)]
    return {"iou": round(iou, 3), "gt_corners": len(gc), "pred_corners": len(pred), "corners_hit6": int(sum(v <= 6 for v in d)),
            "bnd_mean": round(float(np.mean(bd)), 1), "bnd_p90": round(float(np.percentile(bd, 90)), 1), "bnd_max": round(float(np.max(bd)), 1)}


def tile(im, box, gt, pred, title, size=640):
    c = im.crop(box); s = size / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(c)
    f = lambda pts: [((x - box[0]) * s, (y - box[1]) * s) for x, y in pts]
    d.line(f(gt.exterior.coords), fill=(0, 170, 0), width=3)
    if pred:
        q = f(pred + pred[:1]); d.line(q, fill=(30, 60, 255), width=2)
        for x, y in q[:-1]: d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 0, 0), width=2)
    d.rectangle([0, 0, c.width, 14], fill="white"); d.text((3, 1), title, fill=(200, 0, 0))
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="019"); ap.add_argument("--rooms", default="11,18,7,21,14")
    ap.add_argument("--think", type=int, default=800); ap.add_argument("--tag", default="a1"); ap.add_argument("--side", type=int, default=896)
    a = ap.parse_args()
    assert a.floor != "015"
    OUT.mkdir(parents=True, exist_ok=True)
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); W, H = im.size
    gts = V0.load_rooms(a.floor); res = []; tiles = []
    for k, i in enumerate(int(v) for v in a.rooms.split(",")):
        g = gts[i]; box = crop_box(g["poly"], W, H, k)
        c = im.crop(box); s = a.side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
        m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
        txt, think, sec, use = ask(c, PROMPT_A.format(name=g["name"], m2=m2), a.think)
        pts = parse_points(txt)
        pred = [[box[0] + x / 1000 * (box[2] - box[0]), box[1] + y / 1000 * (box[3] - box[1])] for x, y in pts]
        sc = score(pred, g["poly"]) if len(pred) >= 3 else {"iou": 0.0}
        r = {"room": i, "name": g["name"], "sec": round(sec, 1), "out_tokens": use.get("completion_tokens"), "in_tokens": use.get("prompt_tokens"),
             "scale": round(s, 2), **sc, "box": box, "pred": pred, "raw": txt, "think": think}
        res.append(r); print({k2: v for k2, v in r.items() if k2 not in ("pred", "raw", "think", "box")}, flush=True)
        tiles.append(tile(im, box, g["poly"], pred, f"r{i} {g['name']} IoU {sc.get('iou')} corners {sc.get('corners_hit6')}/{sc.get('gt_corners')} {sec:.0f}s"))
    json.dump(res, open(OUT / f"sonuc_{a.tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sh = Image.new("RGB", (sum(t.width for t in tiles) + 10 * len(tiles), max(t.height for t in tiles)), "white"); x = 0
    for t in tiles: sh.paste(t, (x, 0)); x += t.width + 10
    sh.save(OUT / f"sayfa_{a.tag}.png")
    print("mean IoU", round(float(np.mean([r["iou"] for r in res])), 3), "| mean sec", round(float(np.mean([r["sec"] for r in res])), 1))
