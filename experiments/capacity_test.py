"""Capacity test: can the model SEE the right room boundary, apart from drawing it?

For every DGN room the same crop is shown several times, each with one candidate boundary in
magenta: the DGN room and deliberate mistakes made from it:
  hull    the notches gone (convex hull: columns and recesses not followed)
  outer   through the walls' outer face (the room grown by about a wall)
  inner   inside the room (shrunk by about a wall)
  spill   one side pushed into the neighbouring space
The model answers which one is right. Chance: 1/4 (quad) or 1/2 (pair: DGN vs hull only, the
rooms with notches). A model that picks the DGN room reliably but draws badly has an OUTPUT
problem (tools, fine-tuning); one that cannot pick it has a PERCEPTION problem.

usage: python capacity_test.py --mode quad|pair --tag NAME [--url http://127.0.0.1:8091] [--limit N] [--think]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import base64
import io
import json
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

import bench132 as B

LETTERS = "ABCD"
TILE = 520


def iou(a, b):
    u = a.union(b).area
    return a.intersection(b).area / u if u else 0.0


def as_poly(g):
    g = make_valid(g)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    return g if g.geom_type == "Polygon" and not g.is_empty else None


def distractors(P: Polygon, W: int, H: int, rng: random.Random) -> dict:
    t = max(6.0, 0.07 * P.area ** 0.5)
    frame = box(0, 0, W, H)
    out = {"hull": P.convex_hull, "outer": P.buffer(t, join_style=2), "inner": P.buffer(-t, join_style=2)}
    x0, y0, x1, y1 = P.bounds
    side = rng.choice("LRTB")
    dw, dh = 0.3 * (x1 - x0), 0.3 * (y1 - y0)
    strip = {"L": box(x0 - dw, y0, x0 + 1, y1), "R": box(x1 - 1, y0, x1 + dw, y1),
             "T": box(x0, y0 - dh, x1, y0 + 1), "B": box(x0, y1 - 1, x1, y1 + dh)}[side]
    out["spill"] = P.union(strip)
    good = {}
    for k, g in out.items():
        g = as_poly(g.intersection(frame))
        if g is None:
            continue
        g = g.simplify(1.0)
        s = iou(g, P)
        if 0.55 <= s <= 0.97:          # visibly different, not absurd
            good[k] = g
    return good


def draw_tile(crop: Image.Image, poly, letter: str) -> Image.Image:
    s = TILE / max(crop.size)
    im = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s)))).convert("RGB")
    d = ImageDraw.Draw(im)
    pts = [(x * s, y * s) for x, y in list(poly.exterior.coords)]
    d.line(pts, fill=(210, 0, 210), width=4)
    try:
        f = ImageFont.truetype("arialbd.ttf", 44)
    except OSError:
        f = ImageFont.load_default()
    d.rectangle((0, 0, 58, 56), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    d.text((12, 4), letter, fill=(0, 0, 0), font=f)
    return im


def panel(tiles):
    w = max(t.width for t in tiles); h = max(t.height for t in tiles)
    cols = 2
    rows = (len(tiles) + 1) // 2
    P = Image.new("RGB", (cols * w + 12, rows * h + 12 * (rows - 1)), (255, 255, 255))
    for i, t in enumerate(tiles):
        P.paste(t, ((i % cols) * (w + 12), (i // cols) * (h + 12)))
    return P


Q = ("Görüntüde aynı kat planı kesiti {n} kez var ({letters}). Her kopyada '{target}' odası için MOR renkli bir sınır çizilmiş; "
     "kopyalar yalnız bu mor sınırda farklı. Kurallara göre odanın iç sınırını DOĞRU çizen hangisi? Duvar iç yüzü, kolon "
     "girintileri ve kapı/pencere açıklıklarının düz kapatılması önemlidir. Yalnız JSON yaz: {{\"secim\": \"harf\"}}")


def ask(url, img, text, think):
    b = io.BytesIO(); img.save(b, "PNG")
    msgs = [{"role": "system", "content": B.SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}},
                                         {"type": "text", "text": text}]}]
    body = {"messages": msgs, "max_tokens": 4000 if think else 30, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": bool(think)}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quad", "pair"], default="quad")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", action="store_true")
    a = ap.parse_args()
    out = Path(__file__).with_name("runs") / ("cap_" + a.tag); out.mkdir(parents=True, exist_ok=True)
    resf = out / "RESULT.json"
    res = json.load(open(resf, encoding="utf-8")) if resf.exists() else {}
    n = 0
    for key, f, name, crop, ref, label in B.rooms(1400):
        if key in res:
            continue
        P = as_poly(Polygon(ref))
        if P is None:
            continue
        rng = random.Random(key)
        ds = distractors(P, crop.width, crop.height, rng)
        if a.mode == "pair":
            if "hull" not in ds:
                continue
            options = [("dgn", P), ("hull", ds["hull"])]
        else:
            kinds = (["hull"] if "hull" in ds else []) + [k for k in ("outer", "spill", "inner") if k in ds]
            if len(kinds) < 3:
                continue
            options = [("dgn", P)] + [(k, ds[k]) for k in kinds[:3]]
        rng.shuffle(options)
        if a.limit and n >= a.limit:
            break
        n += 1
        letters = LETTERS[:len(options)]
        img = panel([draw_tile(crop, g, letters[i]) for i, (_, g) in enumerate(options)])
        img.save(out / f"{key}.png")
        t0 = time.time()
        try:
            ans = ask(a.url, img, Q.format(n=len(options), letters=", ".join(letters), target=name), a.think)
        except Exception as e:
            ans = f"ERROR {e}"
        m = re.search(r'"secim"\s*:\s*"?([A-D])', ans) or re.search(r"\b([A-D])\b", ans.split("</think>")[-1])
        pick = m.group(1) if m else None
        kinds_by_letter = {letters[i]: k for i, (k, _) in enumerate(options)}
        res[key] = {"floor": f, "name": name, "options": kinds_by_letter, "pick": pick,
                    "picked": kinds_by_letter.get(pick), "correct": kinds_by_letter.get(pick) == "dgn",
                    "seconds": round(time.time() - t0, 1), "raw": ans[-300:]}
        print(key, name[:20], "->", pick, res[key]["picked"], "OK" if res[key]["correct"] else "", flush=True)
        json.dump(res, open(resf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    summarize(res, out, a.mode)


def summarize(res, out, mode):
    v = list(res.values())
    acc = sum(r["correct"] for r in v) / len(v) if v else 0
    chance = 0.5 if mode == "pair" else 0.25
    picked = Counter(r["picked"] for r in v if not r["correct"])
    by_type = defaultdict(list)
    for r in v:
        by_type[B.QT.__name__ and r["name"].split()[0]].append(r["correct"])
    s = {"mode": mode, "rooms": len(v), "accuracy": round(acc, 3), "chance": chance, "wrong_picks": dict(picked),
         "by_room_word": {k: [sum(x), len(x)] for k, x in sorted(by_type.items(), key=lambda kv: -len(kv[1]))}}
    json.dump(s, open(out / "SUMMARY.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("SUMMARY", json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
