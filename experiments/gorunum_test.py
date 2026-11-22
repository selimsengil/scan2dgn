"""How should the encoded plan be SHOWN to a model? Same job in every variant (dev floors, never 015): Qwen sees one
room, gives a rough polygon (taught by solved examples in the same view, no thinking), and the pieces lying inside it
become the room. Only the picture changes:
  scan     the scanned plan as it is (the current way)
  vektor   the plan redrawn from the encoding: only its lines, clean black on white, no text, stamps or hatching noise
  ikili    both pictures, scan first
  vurgulu  the scan faded, with the encoded lines drawn over it in dark blue
Scored with the boundary score (sinir_skor). Also reports the load on the model: prompt tokens and seconds per room.
usage: python gorunum_test.py --view vektor --tag v1 [--floor 019] [--rooms 11,18] [--samples 3]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point

import v0_watershed as V0  # noqa: E402
import tek_oda_test as T  # noqa: E402
import qwen_ornekli as QO  # noqa: E402
import oda_parca as OP  # noqa: E402
import cizgi_kod as CK  # noqa: E402
import cizgi_hat as CH  # noqa: E402
import cizgi_yuz as CY  # noqa: E402
import sinir_skor as SS  # noqa: E402

OUT = config.KT / "gorunum"
_FLOOR = {}


def floor(fl):
    if fl not in _FLOOR:
        im = Image.open(V0.B / fl / "clean.png").convert("RGB"); _FLOOR[fl] = (im, CK.encode(np.array(im.convert("L"))), V0.load_rooms(fl))
    return _FLOOR[fl]


def vector_crop(enc, box, s, size, fade=None):
    """The crop redrawn from the stroke records alone (letters and digits left out)."""
    c = Image.new("RGB", size, "white") if fade is None else fade
    d = ImageDraw.Draw(c); seg = enc["seg"]; ok = ~enc["glyph"]
    m = ok & (np.maximum(seg[:, 0], seg[:, 2]) >= box[0]) & (np.minimum(seg[:, 0], seg[:, 2]) <= box[2]) & (np.maximum(seg[:, 1], seg[:, 3]) >= box[1]) & (np.minimum(seg[:, 1], seg[:, 3]) <= box[3])
    col = (0, 0, 0) if fade is None else (0, 30, 170)
    for (x1, y1, x2, y2), t in zip(seg[m], enc["thick"][m]):
        d.line([((x1 - box[0]) * s, (y1 - box[1]) * s), ((x2 - box[0]) * s, (y2 - box[1]) * s)], fill=col, width=max(1, int(round(min(t, 4.0) * s * 0.8))))
    return c


def views(fl, box, xy, view, side):
    im, enc, _ = floor(fl)
    c = im.crop(box); s = side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    out = []
    if view in ("scan", "ikili"): out.append(QO.mark(c.copy(), box, xy, s))
    if view in ("vektor", "ikili"): out.append(QO.mark(vector_crop(enc, box, s, c.size), box, xy, s))
    if view == "vurgulu": out.append(QO.mark(vector_crop(enc, box, s, c.size, fade=Image.blend(c, Image.new("RGB", c.size, "white"), 0.6)), box, xy, s))
    return out


NOTE = {"scan": "", "vektor": " The picture is the plan redrawn from its lines only: no text, every drawn line is shown in black.",
        "ikili": " You get two pictures of the same part: first the scanned plan, then the same plan redrawn from its lines only (no text). Coordinates are the same in both.",
        "vurgulu": " The scanned plan is faded and every drawn line is repeated in dark blue."}


def example_msgs(spec, view, side):
    out = []
    for k, e in enumerate(spec.split(",")):
        fl, idx = e.split(":"); assert fl != "015"
        im, _, gts = floor(fl); g = gts[int(idx)]; box = T.crop_box(g["poly"], im.size[0], im.size[1], k + 1)
        P = g["poly"]; pts = list(P.exterior.coords)[:-1]
        if P.exterior.is_ccw: pts = pts[::-1]
        rel = [[int(round((x - box[0]) / (box[2] - box[0]) * 1000)), int(round((y - box[1]) / (box[3] - box[1]) * 1000))] for x, y in pts]
        m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
        out.append({"role": "user", "content": [QO.img_part(v) for v in views(fl, box, g["xy"], view, side)] + [{"type": "text", "text": f"Example {k + 1}. " + QO.Q.format(name=g["name"], m2=m2)}]})
        out.append({"role": "assistant", "content": json.dumps({"points": rel})})
    return out


BACK = {"name": "qwen", "url": QO.URL, "effort": "none", "max_tokens": 900}


def rough(msgs, temp):
    if BACK["name"] == "strata":      # the Strata server (DeepSeek V4.1 Flash 2-bit, OpenAI format); reasoning_effort "none" = no thinking
        body = {"model": BACK.get("model", "deepseek-v4.1-flash"), "messages": msgs, "temperature": temp, "max_tokens": BACK["max_tokens"], "reasoning_effort": BACK["effort"]}
    else:
        body = {"messages": msgs, "temperature": temp, "max_tokens": 900, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BACK["url"], json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    u = r.get("usage", {}); BACK.setdefault("out", []).append(u.get("completion_tokens", 0))
    return T.parse_points(r["choices"][0]["message"].get("content") or "")[:60], u


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="scan", choices=list(NOTE)); ap.add_argument("--tag", default="v1"); ap.add_argument("--floor", default="019"); ap.add_argument("--rooms", default="")
    ap.add_argument("--samples", type=int, default=3); ap.add_argument("--temp", type=float, default=0.4); ap.add_argument("--examples", default="017:18,013:0,017:15")
    ap.add_argument("--side", type=int, default=896); ap.add_argument("--reach", type=float, default=230.0)
    ap.add_argument("--backend", default="qwen", choices=["qwen", "strata"]); ap.add_argument("--effort", default="none"); ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--ex-side", type=int, default=640); ap.add_argument("--port", type=int, default=8093); ap.add_argument("--sys-file", default=""); ap.add_argument("--model-name", default="deepseek-v4.1-flash")
    a = ap.parse_args(); assert a.floor != "015"
    if a.backend == "strata":         # any OpenAI-format server with reasoning_effort (Strata 8093, Maya 8080)
        BACK.update({"name": "strata", "url": f"http://127.0.0.1:{a.port}/v1/chat/completions", "effort": a.effort, "max_tokens": a.max_tokens, "model": a.model_name})
    out = OUT / f"{a.floor}_{a.view}_{a.tag}"
    t_all = time.time(); out.mkdir(parents=True, exist_ok=True)
    im, enc, gts = floor(a.floor); W, H = im.size
    ids = [int(v) for v in a.rooms.split(",")] if a.rooms else [i for i, g in enumerate(gts) if g["poly"].geom_type == "Polygon"]
    ex = example_msgs(a.examples, a.view, a.ex_side if a.view != "ikili" else 512)
    sysmsg = (open(a.sys_file, encoding="utf-8").read() if a.sys_file else QO.SYS) + NOTE[a.view]; rows = []; t_model = 0.0; tok_in = []; calls = 0
    for k, i in enumerate(ids):
        g = gts[i]; box = T.crop_box(g["poly"], W, H, k); bw, bh = box[2] - box[0], box[3] - box[1]
        m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
        msgs = [{"role": "system", "content": sysmsg}] + ex + [{"role": "user", "content": [QO.img_part(v) for v in views(a.floor, box, g["xy"], a.view, a.side if a.view != "ikili" else 704)] + [{"type": "text", "text": "Now the new room. " + QO.Q.format(name=g["name"], m2=m2)}]}]
        raws = []; tries = 0; t = time.time()
        while len(raws) < a.samples and tries < a.samples + 3:
            tries += 1; pts, use = rough(msgs, a.temp if tries == 1 else max(a.temp, 0.6)); calls += 1; tok_in.append(use.get("prompt_tokens", 0))
            fl = [[box[0] + x / 1000 * bw, box[1] + y / 1000 * bh] for x, y in pts] if len(pts) >= 3 else None
            P = OP.clean(fl) if fl else None
            if P is not None and P.buffer(6).contains(Point(*g["xy"])): raws.append(fl)
        t_model += time.time() - t
        fs = CY.faces(CH.lines_in(enc, box, min_len=8), box, min_area=1.0, long_reach=a.reach, max_gap=a.reach)
        U, sel = OP.vote([OP.clean(r) for r in raws], fs, seed_xy=g["xy"])
        final = [list(p) for p in U.simplify(0.4).exterior.coords[:-1]] if U is not None else (raws[0] if raws else [])
        rows.append({"room": i, "name": g["name"], "boundary": final, "rough": raws[0] if raws else []})
        print(f"r{i} {g['name']}: {len(raws)} drawings, {len(sel)} pieces", flush=True)
    json.dump({"rooms": rows, "load": {"view": a.view, "model_seconds": round(t_model, 1), "calls": calls, "prompt_tokens_per_call": int(np.mean(tok_in)) if tok_in else 0, "seconds_per_room": round(t_model / max(1, len(rows)), 1)}},
              open(out / "SONUC.json", "w", encoding="utf-8"), ensure_ascii=False)
    S, _ = SS.score_run(out, a.floor, 3.0, quiet=bool(not a.rooms))
    print("OUT TOKENS per call", int(np.mean(BACK.get("out", [0]))), "| wall", round(time.time() - t_all, 1), "s")
    print("LOAD", {"view": a.view, "seconds_per_room": round(t_model / max(1, len(rows)), 1), "prompt_tokens_per_call": int(np.mean(tok_in)) if tok_in else 0, "calls": calls})
