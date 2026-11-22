"""DeepSeek-V4.1 (the original mixed-precision weights, Colibri) draws a room with thinking and tools.

The model decides the shape; the tools only help it see and measure:
  0. example   one solved teaching room (Tekke 06, green = the DGN answer) -> the model writes its rules
  1. overview  the room's crop with rulers (one pixel frame for every view) and a red dot on the label;
               the model may zoom first, then gives the inner boundary's corners, clockwise
  2. corners   for each corner a native-resolution window with the rulers of the same frame, the current
               polygon in magenta and this corner ringed: keep / move / delete / insert corners
  3. check     the polygon drawn on the overview: accept or give corrected corners
  4. snap      each edge settles on the nearest ink line (a few pixels; it never changes the shape)
One image per request (the engine's protocol): the model's notes travel between steps as text.

usage: python ds_agent.py --rooms 015_02,019_18,001_17 --tag ds_agent_a
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import bench132 as B
import qwen_test as QT
import snap

sys.path.insert(0, r"C:\Users\Selim\Documents\cizim_projeler\asistan_cizim\20261107_tekke_dort_oda")
sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io")

EXE = r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io\deepseek_v41_vl.exe"
# the fastest settings measured for the 500 GB checkpoint: one prefill pass, GPU vision attention, all cold
# experts streamed to the GPU, a reserve that holds ~2300 prompt rows
ENV = {"COLI_VK_CHAIN_ROWS": "2304", "COLI_VK_TIER_RESERVE_GB": "5.0", "COLI_VK_TIER_STREAM_ROWS": "1", "V41_STATS": "1"}

try:
    FONT = ImageFont.truetype("arial.ttf", 15)
    FONT_B = ImageFont.truetype("arialbd.ttf", 18)
except OSError:
    FONT = FONT_B = ImageFont.load_default()

SYSTEM = (
    "You are a careful architectural draughtsman reading a scanned floor plan (Turkish labels). Your job is to trace ONE room's "
    "boundary exactly as the office's CAD drawings do:\n"
    "- The boundary runs along the wall face that looks INTO the room (the inner face), never along the wall's outer face or "
    "its centre.\n"
    "- Door and window openings are closed with a straight line continuing that inner face between the two jambs; door leaves, "
    "swing arcs and frames are not boundary.\n"
    "- Columns or wall returns that project into the room are followed: the boundary steps around them (notches). Recesses "
    "that belong to the room are included.\n"
    "- Furniture, sanitary fixtures, kitchen counters, hatching/tiles, text, dimension lines (thin lines with ticks/arrows and "
    "numbers), axis lines, stamps and signatures are NOT boundary.\n"
    "- Slanted walls are followed as they are.\n"
    "Coordinates: every image has rulers on its top and left edges that give PIXEL coordinates of one fixed frame (x to the "
    "right, y downward). Read positions from the rulers; the faint grid lines continue the major ticks. All views use the same "
    "frame, so a zoomed view shows larger numbers on the same scale.\n"
    "Think step by step, then end with exactly one JSON object as your final answer.")

EXAMPLE_Q = (
    "SOLVED EXAMPLE from the same office. The GREEN line is the correct boundary of the room '{title}', as drawn in the CAD file. "
    "Teacher's note: {focus}\n"
    "Study where the green line runs: which wall face, how it passes openings, how it treats columns, recesses, fixtures and "
    "hatching. Write the 4-6 rules you will apply to the next room. "
    'Final answer JSON: {{"rules": ["...", "..."]}}')

OVERVIEW_Q = (
    "TASK: trace the room '{target}'. The RED dot sits on its printed label; trace the room that contains the red dot only "
    "(neighbouring rooms are visible too).\n"
    "Your rules from the solved example: {rules}\n"
    "{history}"
    "If you need a closer look first, ask for a zoom: {{\"zoom\": [x0, y0, x1, y1]}} (frame pixels; at most {zooms_left} more). "
    "Otherwise give every corner of the room's inner boundary, clockwise, notches included, read from the rulers: "
    '{{"points": [[x, y], ...], "note": "short"}}')

ZOOM_Q = (
    "ZOOM of the frame region {box} for the room '{target}' (rulers in the same frame). {history}"
    "Ask for another zoom ({{\"zoom\": [x0, y0, x1, y1]}}, {zooms_left} left) or give the corners now: "
    '{{"points": [[x, y], ...], "note": "short"}}')

CORNER_Q = (
    "CORNER CHECK {i}/{n} for the room '{target}'. This is a close native-resolution view; rulers in the same frame. "
    "The MAGENTA line is the current boundary, the CYAN ring marks corner {i} at ({x}, {y}). "
    "Is the ring exactly on the corner of the room's INNER wall face? Answer with ONE of:\n"
    '{{"keep": true}}  |  {{"move": [x, y]}}  |  {{"delete": true}}  (not a real corner)  |  '
    '{{"move": [x, y], "insert_after": [[x, y], ...]}}  (a notch the boundary missed, clockwise after this corner)')

CHECK_Q = (
    "FINAL CHECK for the room '{target}'. The MAGENTA polygon (numbered corners) is the boundary after the corner checks. "
    "Compare it with the plan: inner wall faces, openings closed straight, columns/notches followed, nothing outside the room. "
    'Answer {{"accept": true}} or the corrected corners {{"points": [[x, y], ...], "note": "what changed"}}')


def nice_step(span_px: float, target: int = 12) -> int:
    raw = span_px / target
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500):
        if s >= raw:
            return s
    return 1000


def ruled(img: Image.Image, box, max_side: int, poly=None, ring=None, dot=None, numbered=False) -> Image.Image:
    """The frame region box=(x0, y0, x1, y1) scaled to max_side, with rulers in frame pixels."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    reg = img.crop((x0, y0, x1, y1)).convert("RGB")
    s = min(max_side / max(reg.size), 6.0)
    reg = reg.resize((max(1, int(reg.width * s)), max(1, int(reg.height * s))), Image.Resampling.LANCZOS)
    L, T = 62, 34
    out = Image.new("RGB", (reg.width + L + 8, reg.height + T + 8), (255, 255, 255))
    out.paste(reg, (L, T))
    d = ImageDraw.Draw(out, "RGBA")
    fx = lambda x: L + (x - x0) * s
    fy = lambda y: T + (y - y0) * s
    major = nice_step(x1 - x0)
    minor = major / 5 if major >= 5 else major
    v = (x0 // minor) * minor
    while v <= x1:
        if v >= x0:
            X = fx(v); big = v % major == 0
            d.line((X, T - (10 if big else 4), X, T), fill=(0, 0, 0), width=1)
            if big:
                d.line((X, T, X, T + reg.height), fill=(0, 140, 255, 55), width=1)
                d.text((X - 12, 2), str(int(v)), fill=(0, 0, 0), font=FONT)
        v += minor
    v = (y0 // minor) * minor
    while v <= y1:
        if v >= y0:
            Y = fy(v); big = v % major == 0
            d.line((L - (10 if big else 4), Y, L, Y), fill=(0, 0, 0), width=1)
            if big:
                d.line((L, Y, L + reg.width, Y), fill=(0, 140, 255, 55), width=1)
                d.text((2, Y - 8), str(int(v)), fill=(0, 0, 0), font=FONT)
        v += minor
    if dot:
        r = 7
        d.ellipse((fx(dot[0]) - r, fy(dot[1]) - r, fx(dot[0]) + r, fy(dot[1]) + r), fill=(230, 0, 0))
    if poly and len(poly) >= 2:
        q = [(fx(x), fy(y)) for x, y in poly]
        d.line(q + q[:1], fill=(210, 0, 210, 230), width=3)
        if numbered:
            for k, (X, Y) in enumerate(q):
                d.rectangle((X + 4, Y + 4, X + 26, Y + 22), fill=(255, 255, 255, 220))
                d.text((X + 6, Y + 4), str(k + 1), fill=(160, 0, 160), font=FONT_B)
    if ring:
        X, Y = fx(ring[0]), fy(ring[1])
        d.ellipse((X - 14, Y - 14, X + 14, Y + 14), outline=(0, 200, 230), width=3)
    return out


def last_json(raw: str) -> dict:
    tail = raw.split("</think>")[-1]
    for txt in (tail, raw):
        for m in reversed(list(re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", txt, re.S))):
            try:
                return json.loads(m.group(0))
            except Exception:
                continue
    return {}


class Agent:
    def __init__(self, out: Path):
        import deepseek_loop as L1
        L1.EXE = Path(EXE)
        self.S = L1.S
        env = {"COLI_VK_SHADERS": str(Path(EXE).parent / "shaders"), **ENV}
        self.sess = L1.start_session(out, 1500, env)
        self.n = 0
        self.out = out
        self.log = []

    def ask(self, img: Image.Image, text: str, effort: str, budget: int, tag: str) -> dict:
        self.n += 1
        p = self.out / f"step_{self.n:03d}_{tag}.png"; img.save(p)
        t0 = time.time()
        prepared = self.S.prepare_multimodal_payload(p, SYSTEM, text, enable_thinking=True, reasoning_effort=effort)
        r = self.sess.submit_multimodal(self.out / f"req_{self.n:03d}", prepared, generation_budget=budget, timeout_s=7200)
        js = last_json(r["raw"])
        if not js:   # the budget ran out inside the thinking: once more without it, the notes so far kept short
            prepared = self.S.prepare_multimodal_payload(p, SYSTEM, text + "\nAnswer with the JSON only.", enable_thinking=False)
            r2 = self.sess.submit_multimodal(self.out / f"req_{self.n:03d}b", prepared, generation_budget=300, timeout_s=3600)
            js = last_json(r2["raw"])
        e = {"step": self.n, "tag": tag, "effort": effort, "seconds": round(time.time() - t0, 1), "answer": js,
             "tokens": r.get("generated_tokens"), "prompt_tokens": r.get("prompt_tokens"),
             "prefill_s": round(r.get("prefill_only_seconds") or 0, 1), "decode_tps": round(r.get("decode_tps") or 0, 2),
             "thinking_tail": r["raw"].split("</think>")[0][-600:]}
        self.log.append(e)
        print(json.dumps({k: e[k] for k in ("step", "tag", "seconds", "tokens", "prefill_s", "decode_tps")}),
              "->", json.dumps(js, ensure_ascii=False)[:200], flush=True)
        return js

    def close(self):
        self.sess.close()


def pts_ok(p):
    try:
        q = [[float(x), float(y)] for x, y in p]
        return q if len(q) >= 3 else None
    except Exception:
        return None


def example_image():
    import run_pilot as RP
    teach = json.loads((RP.SOURCE_DIR / "OGRETIM_ORNEKLERI.json").read_text(encoding="utf8"))["examples"]
    items = {r["id"]: r for r in json.loads((RP.SOURCE_DIR / "DGN_TARGETS.json").read_text(encoding="utf8"))["elements"]}
    ex = next(e for e in teach if e["number"] == "06")
    x0, y0, x1, y1 = ex["crop_box"]
    crop = RP.source().crop((x0, y0, x1, y1)).convert("RGB")
    pts = items[ex["room_id"]]["points"]
    pts = [[x - x0, y - y0] for x, y in (pts[:-1] if pts[0] == pts[-1] else pts)]
    im = ruled(crop, (0, 0, crop.width, crop.height), 1000)
    d = ImageDraw.Draw(im)
    s = min(1000 / max(crop.size), 6.0)
    q = [(62 + x * s, 34 + y * s) for x, y in pts]
    d.line(q + q[:1], fill=(0, 170, 60), width=4)
    return im, ex


def draw_room(ag: Agent, key, name, crop, ref, label, rules: str, max_zoom=2):
    W, H = crop.size
    full = (0, 0, W, H)
    history, zooms = "", 0
    js = ag.ask(ruled(crop, full, 1280, dot=label), OVERVIEW_Q.format(target=name, rules=rules, history=history,
                                                                       zooms_left=max_zoom), "high", 2600, key + "_overview")
    while "zoom" in js and zooms < max_zoom:
        zooms += 1
        box = [float(v) for v in js["zoom"]][:4]
        history += f"(You already looked at the zoom {[int(v) for v in box]}.) "
        js = ag.ask(ruled(crop, box, 1100, dot=label), ZOOM_Q.format(box=[int(v) for v in box], target=name, history=history,
                                                                     zooms_left=max_zoom - zooms), "high", 2600, key + f"_zoom{zooms}")
    first = pts_ok(js.get("points") or [])
    if not first:
        js = ag.ask(ruled(crop, full, 1280, dot=label), OVERVIEW_Q.format(target=name, rules=rules, history=history + "Now give the corners. ",
                                                                           zooms_left=0), "high", 2600, key + "_overview2")
        first = pts_ok(js.get("points") or [])
    if not first:
        return {"error": "no polygon"}
    poly = [list(p) for p in first]
    # 2. corner by corner, native resolution
    win = max(90.0, 0.18 * min(W, H))
    i = 0
    while i < len(poly) and i < 40:
        x, y = poly[i]
        box = (x - win / 2, y - win / 2, x + win / 2, y + win / 2)
        a = ag.ask(ruled(crop, box, 760, poly=poly, ring=(x, y)),
                   CORNER_Q.format(i=i + 1, n=len(poly), target=name, x=int(x), y=int(y)), "low", 900, key + f"_c{i + 1}")
        if a.get("delete") and len(poly) > 3:
            poly.pop(i)
            continue
        if isinstance(a.get("move"), list) and len(a["move"]) == 2:
            poly[i] = [float(a["move"][0]), float(a["move"][1])]
        extra = pts_ok(a.get("insert_after") or []) if isinstance(a.get("insert_after"), list) and len(a.get("insert_after") or []) else None
        if extra is None and isinstance(a.get("insert_after"), list) and len(a["insert_after"]) in (1, 2):
            try:
                extra = [[float(x), float(y)] for x, y in a["insert_after"]]
            except Exception:
                extra = None
        if extra:
            poly[i + 1:i + 1] = extra
            i += 1 + len(extra)
            continue
        i += 1
    refined = [p[:] for p in poly]
    # 3. final check on the overview
    c = ag.ask(ruled(crop, full, 1280, poly=poly, numbered=True, dot=label), CHECK_Q.format(target=name), "high", 2600, key + "_check")
    final = pts_ok(c.get("points") or []) if not c.get("accept") else None
    final = final or refined
    sn = snap.snap_polygon(crop, final) if len(final) >= 3 else []
    sc = lambda p: QT.brief(QT.score(p, ref, type("H", (), {"inside": None})())) if len(p) >= 3 else {"valid": False}
    return {"first": first, "refined": refined, "final": final, "snapped": sn,
            "score": {"first": sc(first), "refined": sc(refined), "final": sc(final), "snapped": sc(sn)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="015_02,019_18,001_17")
    ap.add_argument("--tag", default="ds_agent_a")
    a = ap.parse_args()
    out = Path(__file__).with_name("runs") / a.tag; out.mkdir(parents=True, exist_ok=True)
    want = a.rooms.split(",")
    todo = [r for r in B.rooms(1400) if r[0] in want]
    ag = Agent(out)
    res = {}
    try:
        t0 = time.time()
        img, ex = example_image()
        rj = ag.ask(img, EXAMPLE_Q.format(title=ex["title"], focus=ex["focus"]), "high", 2600, "example")
        rules = "; ".join(rj.get("rules") or []) or "(none written)"
        print("RULES:", rules, flush=True)
        res["_example"] = {"rules": rules, "seconds": round(time.time() - t0, 1)}
        for key, f, name, crop, ref, label in todo:
            t1 = time.time()
            r = draw_room(ag, key, name, crop, ref, label, rules)
            r.update({"name": name, "floor": f, "reference": ref, "size": list(crop.size), "seconds": round(time.time() - t1, 1)})
            res[key] = r
            crop.save(out / f"{key}.png")
            print("ROOM", key, name, json.dumps(r.get("score"), ensure_ascii=False), r["seconds"], "s", flush=True)
            json.dump({"results": res, "log": ag.log}, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        ag.close()
        json.dump({"results": res, "log": ag.log}, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
