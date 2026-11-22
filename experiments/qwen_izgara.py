"""Single-room test with light tools, model-led (never 015 for development): the model sees the room enlarged with a
coordinate grid in FLOOR pixels, writes the corners by reading the grid, then gets its own drawing back together with
dark-run scans across every edge and corrects it. No shape extraction by the tool.
usage: python qwen_izgara.py --rooms 11,18,7,21,14 --tag g1 [--rounds 1] [--backend local|deepseek]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, base64, io, json, re, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import v0_watershed as V0  # noqa: E402
import tek_oda_test as T  # noqa: E402

OUT = T.OUT
BACK = {"name": "local", "url": "http://127.0.0.1:8091/v1/chat/completions", "model": None, "think": 1000}
USAGE = {"in": 0, "out": 0}

RULES = """The boundary of a room is the INNER face of its walls: the line where the floor of this room meets the wall.
- Walls are the thick double-line bands; hatched or grey filled rectangles in them are columns.
- Columns and wall jogs that stick into the room change the boundary: go around them (each adds corners).
- Slanted walls stay slanted: give their two real end points.
- Across a door or an open passage, continue the wall line straight to the other side of the opening.
- The room may be L-shaped or have extensions and niches: include every part of its floor that no wall separates from it.
- Not walls, ignore them: furniture, kitchen counters, sinks, toilets, floor tiles, door swings, dimension lines and
  their numbers, level marks, stamps, text.
- At a window the boundary runs along the inner wall line."""

P1 = """This is an enlarged part of an architectural floor plan with a red coordinate grid. The red numbers along the top (x) and
the left side (y) are FLOOR PIXEL coordinates; grid lines are {g} px apart and every labelled line shows its value.

Task: give the boundary of ONE room, the room that contains the label "{name}"{m2}.

{rules}

Read every corner position from the grid (interpolate between grid lines). Answer with JSON only:
{{"points": [[x, y], ...]}} : the corners in clockwise order, in floor pixel coordinates."""

P1R = """This is an enlarged part of an architectural floor plan.

Task: give the boundary of ONE room, the room that contains the label "{name}"{m2}.

{rules}

Answer with JSON only: {{"points": [[x, y], ...]}} : the corners in clockwise order, coordinates 0-1000 relative to this
image (x to the right, y down). One point per corner."""

P2 = """Here is your boundary drawn in magenta on the same plan (corner numbers in blue), for the room that contains the label "{name}"{m2}.
Your corners were: {mine}

For every edge I scanned the dark pixels across it at the edge's middle. A wall shows as two dark runs a wall-thickness apart
(its two faces) or one thick run; the inner face is the run nearest to the room's floor. Scans ({frame}):
{scans}

Check the drawing against the plan and fix it:
- an edge that is not on a dark run: move it onto the inner wall face (use the run coordinates);
- a part of the room's floor left outside, or another room's floor inside: change the corners;
- a column or jog you cut through: go around it; a slanted wall drawn straight: make it slanted.

{rules}

Answer with JSON only: {{"points": [[x, y], ...]}} : the corrected corners in clockwise order, {frame}.
If the drawing is already right, repeat the same points."""


def ask(parts, think):
    content = []
    for p in parts:
        if isinstance(p, str): content.append({"type": "text", "text": p})
        else:
            b = io.BytesIO(); p.save(b, "PNG")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}})
    body = {"messages": [{"role": "user", "content": content}]}
    headers = {"Content-Type": "application/json"}
    if BACK["name"] == "local":
        body.update({"temperature": 0.2, "max_tokens": 1500 + think, "chat_template_kwargs": {"enable_thinking": think > 0}})
        if think > 0: body["thinking_budget_tokens"] = think
    else:
        import kat_agent_api as K
        body.update({"model": BACK["model"], "max_tokens": 8000, "thinking": {"type": "enabled"}, "reasoning_effort": BACK.get("effort", "medium")})
        headers["Authorization"] = "Bearer " + K.api_key()
    t = time.time()
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(BACK["url"], json.dumps(body).encode(), headers), timeout=900)); break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:400]
            if attempt == 2: raise SystemExit(f"API error {e.code}: {msg}")
            time.sleep(8)
    u = r.get("usage", {}); USAGE["in"] += u.get("prompt_tokens", 0); USAGE["out"] += u.get("completion_tokens", 0)
    return r["choices"][0]["message"].get("content") or "", time.time() - t


def grid_view(im, box, g=20, side=896, poly=None):
    """Crop enlarged, with a labelled grid in floor pixels; optional polygon overlay with corner numbers."""
    x0, y0, x1, y1 = box; reg = im.crop(box); s = side / max(reg.size)
    reg = reg.resize((int(reg.width * s), int(reg.height * s)), Image.LANCZOS)
    L, Tm = (46, 22) if g else (0, 0); out = Image.new("RGB", (reg.width + L, reg.height + Tm), "white"); out.paste(reg, (L, Tm))
    d = ImageDraw.Draw(out, "RGBA")
    try: font = ImageFont.truetype("arialbd.ttf", 15)
    except Exception: font = ImageFont.load_default()
    major = (g * 5 if (x1 - x0) / g > 24 else g * 2) if g else 1
    for v in (range((x0 // g + 1) * g, x1, g) if g else []):
        X = L + (v - x0) * s; d.line([(X, Tm), (X, out.height)], fill=(255, 0, 0, 130 if v % major == 0 else 50), width=1)
        if v % major == 0: d.text((X - 14, 2), str(v), fill=(200, 0, 0), font=font)
    for v in (range((y0 // g + 1) * g, y1, g) if g else []):
        Y = Tm + (v - y0) * s; d.line([(L, Y), (out.width, Y)], fill=(255, 0, 0, 130 if v % major == 0 else 50), width=1)
        if v % major == 0: d.text((2, Y - 8), str(v), fill=(200, 0, 0), font=font)
    if poly:
        q = [(L + (x - x0) * s, Tm + (y - y0) * s) for x, y in poly]
        d.line(q + q[:1], fill=(220, 0, 220, 255), width=3)
        for k, (x, y) in enumerate(q):
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 0, 255, 255)); d.text((x + 5, y + 3), str(k + 1), fill=(0, 0, 255), font=font)
    return out


def runs_across(gray, a, b, reach=30, dark=150, fx=lambda v: int(round(v)), fy=lambda v: int(round(v))):
    """Dark runs along the line across edge a-b at its middle, as coordinates on the axis that crosses the edge."""
    a = np.array(a, float); b = np.array(b, float); d = b - a; L = np.hypot(*d)
    if L < 6: return None
    n = np.array([-d[1], d[0]]) / L; m = (a + b) / 2; H, W = gray.shape; runs = []; run = None
    for o in range(-reach, reach + 1):
        x, y = int(round(m[0] + o * n[0])), int(round(m[1] + o * n[1]))
        isd = 0 <= x < W and 0 <= y < H and gray[y, x] < dark
        if isd and run is None: run = o
        if (not isd or o == reach) and run is not None: runs.append((run, o - 1 if not isd else o)); run = None
    horiz = abs(d[0]) >= abs(d[1]) * 4; vert = abs(d[1]) >= abs(d[0]) * 4
    if horiz: return "y: " + ", ".join("-".join(str(v) for v in sorted((fy(m[1] + r0 * n[1]), fy(m[1] + r1 * n[1])))) for r0, r1 in runs)
    if vert: return "x: " + ", ".join("-".join(str(v) for v in sorted((fx(m[0] + r0 * n[0]), fx(m[0] + r1 * n[0])))) for r0, r1 in runs)
    return "offsets across the edge (px, 0 = your edge): " + ", ".join(f"{r0}..{r1}" for r0, r1 in runs)


def scan_text(gray, pts, fx=lambda v: int(round(v)), fy=lambda v: int(round(v))):
    out = []
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]; r = runs_across(gray, a, b, fx=fx, fy=fy)
        if r is None: continue
        d = (b[0] - a[0], b[1] - a[1]); kind = "horizontal" if abs(d[0]) >= abs(d[1]) * 4 else "vertical" if abs(d[1]) >= abs(d[0]) * 4 else "slanted"
        own = f"your y={fy((a[1] + b[1]) / 2)}" if kind == "horizontal" else f"your x={fx((a[0] + b[0]) / 2)}" if kind == "vertical" else ""
        out.append(f"edge {k + 1}-{(k + 1) % len(pts) + 1} ({kind}, {own}): dark runs at {r or 'none within 30 px'}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="019"); ap.add_argument("--rooms", default="11,18,7,21,14"); ap.add_argument("--tag", default="g1")
    ap.add_argument("--rounds", type=int, default=1); ap.add_argument("--think", type=int, default=1000); ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--frame", default="rel", choices=["rel", "floor"]); ap.add_argument("--backend", default="local"); ap.add_argument("--model", default="deepseek-flash"); ap.add_argument("--effort", default="medium")
    a = ap.parse_args(); assert a.floor != "015"
    if a.backend == "deepseek":
        BACK.update({"name": "deepseek", "url": "https://api.deepseek.com/chat/completions", "model": a.model, "effort": a.effort})
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); W, H = im.size; gray = np.array(im.convert("L"))
    gts = V0.load_rooms(a.floor); rows = []; tiles = []
    for k, i in enumerate(int(v) for v in a.rooms.split(",")):
        g = gts[i]; box = T.crop_box(g["poly"], W, H, k)
        m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
        t0 = time.time(); hist = []
        rel = a.frame == "rel"; bw, bh = box[2] - box[0], box[3] - box[1]
        if rel:
            to_f = lambda q: [[box[0] + x / 1000 * bw, box[1] + y / 1000 * bh] for x, y in q]
            fx = lambda v: int(round((v - box[0]) / bw * 1000)); fy = lambda v: int(round((v - box[1]) / bh * 1000))
            view = lambda poly=None: grid_view(im, box, 0, poly=poly); frame = "coordinates 0-1000 relative to this image"
            first = P1R.format(name=g["name"], m2=m2, rules=RULES)
        else:
            to_f = lambda q: q; fx = fy = lambda v: int(round(v))
            view = lambda poly=None: grid_view(im, box, a.grid, poly=poly); frame = "floor pixel coordinates"
            first = P1.format(g=a.grid, name=g["name"], m2=m2, rules=RULES)
        txt, _ = ask([view(), first], a.think)
        pts = to_f(T.parse_points(txt)); hist.append(pts)
        for r in range(a.rounds):
            if len(pts) < 3: break
            mine = json.dumps([[fx(x), fy(y)] for x, y in pts])
            txt, _ = ask([view(pts), P2.format(name=g["name"], m2=m2, mine=mine, frame=frame, scans=scan_text(gray, pts, fx, fy), rules=RULES)], a.think)
            new = to_f(T.parse_points(txt))
            if len(new) >= 3: pts = new
            hist.append(pts)
        sec = time.time() - t0
        sc = [T.score(h, g["poly"]) if len(h) >= 3 else {"iou": 0.0} for h in hist]
        rows.append({"room": i, "sec": round(sec, 1), "iou_rounds": [s["iou"] for s in sc], "corners": f'{sc[-1].get("corners_hit6")}/{sc[-1].get("gt_corners")}',
                     "pred_corners": sc[-1].get("pred_corners"), "bnd_mean": sc[-1].get("bnd_mean"), "bnd_p90": sc[-1].get("bnd_p90")})
        print(rows[-1], flush=True)
        json.dump({"hist": hist, "raw": txt}, open(OUT / f"izgara_{a.tag}_r{i}.json", "w", encoding="utf-8"))
        tiles.append(T.tile(im, box, g["poly"], pts if len(pts) >= 3 else None, f"r{i} IoU {' > '.join(str(s['iou']) for s in sc)} kose {sc[-1].get('corners_hit6')}/{sc[-1].get('gt_corners')} {sec:.0f}s"))
    n = (len(tiles) + 2) // 3; cw = max(t.width for t in tiles) + 10; ch = max(t.height for t in tiles) + 10
    sh = Image.new("RGB", (cw * 3, ch * n), "white")
    for j, t in enumerate(tiles): sh.paste(t, ((j % 3) * cw, (j // 3) * ch))
    sh.save(OUT / f"sayfa_izgara_{a.tag}.png")
    print("mean IoU first", round(float(np.mean([r["iou_rounds"][0] for r in rows])), 3), "last", round(float(np.mean([r["iou_rounds"][-1] for r in rows])), 3),
          "| mean sec", round(float(np.mean([r["sec"] for r in rows])), 1), "| tokens", USAGE)
