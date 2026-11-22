"""Qwen's tools (designed by Qwen in the workshop, written by Claude), plugged into kat_agent_api.py --tool-module.

Qwen's design: the model does the semantics (labels, m², which label is which room, door/column decisions, notes);
the tools do every pixel-exact thing (wall faces, intersections, door closing, m², checks).

State lives in ctx["state"]: rooms (id, name, m2, type, flat, points), outside points, barriers, erasures, scale,
manual polygons, last computed cells.
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, Point

import qwen_geo as G

GUIDE = """HOW TO WORK (your own tool design):
You do the semantics; the tools do every pixel-exact thing. Workflow:
1. view the floor in parts (grid 100) and read EVERY room label: name, printed m², flat number, and a point INSIDE
   the room (the label position is fine). Also note balconies / gardens / outside areas.
2. rooms: register all of them in one call (id, name, m2, type, flat, x, y; extra points for L-shaped rooms;
   outside = points in balconies, gardens, shafts you will not draw). You can call rooms again to add/fix entries.
3. calibrate mode=auto FIRST (scale from the printed m² of the registered rooms; robust because it uses many rooms).
   A dimension is only a cross-check: dimension chains and wall-axis vs wall-face make single readings unreliable.
   If the two disagree by >10%, trust auto unless you re-measured two dimensions that agree with each other.
4. walls: computes the wall mask and one cell per room (watershed from your points; door necks split rooms).
   Returns a table: each room's computed m² vs printed m², ratio, and a flag; plus an overview image.
5. For every flagged room, THINK about why it is wrong, look at it with view (overlay=true shows the cells), then fix:
   - ratio > 1 (cell too big): the cell leaked through an opening or a missing wall -> barrier(x1,y1,x2,y2) along the
     inner wall face line / across the door between the jamb ends.
   - ratio < 1 (cell too small): a false wall (furniture, dimension line, hatch) cuts the room -> erase(x0,y0,x1,y1)
     the false piece, or the room is L-shaped / two labels -> add an extra point with rooms.
   - wrong neighbour assignment: your point is in the wrong room -> fix the point with rooms.
   Then call walls again and compare the numbers. A fix is right only if the ratio moves toward 1.00.
   If a room cannot be fixed after 2 attempts, use polygon(id, points) to give its corners by hand (last resort),
   or leave it and write a note.
   If MOST rooms are flagged in the same direction, the scale is wrong, not the walls: fix the scale first.
   Work in small batches: think about 1-3 rooms, call the tool, look at the new numbers. Every reply must end in a
   tool call - long thinking without a call is wasted. Do not abandon the tools for hand tracing.
6. check: overlaps, self-crossings, ratio summary. 7. export: writes the result and submits it.
Ratios between 0.93 and 1.07 are normal (printed m² are rounded, walls plastered). Never 'fix' a good room."""

_I = lambda **p: {"type": "object", "properties": p}
TOOLS = [
    {"type": "function", "function": {"name": "view", "description": "Enlarged crop of the floor with a coordinate grid (floor pixel frame). overlay=true draws the current room cells (id + outline) and barriers.",
     "parameters": {**_I(x0={"type": "integer"}, y0={"type": "integer"}, x1={"type": "integer"}, y1={"type": "integer"},
                         grid={"type": "integer", "description": "grid step in floor px (default 50)"}, overlay={"type": "boolean"}),
                    "required": ["x0", "y0", "x1", "y1"]}}},
    {"type": "function", "function": {"name": "rooms", "description": "Register or update rooms. Each item: id, name, m2 (printed, null if none), type (room|common|shaft), flat, x, y (a point inside the room), extra (optional list of [x,y] more points inside the same room). outside: list of [x,y] points in balconies/gardens/areas not drawn. replace=true replaces the whole list.",
     "parameters": {**_I(items={"type": "array", "items": {"type": "object"}}, outside={"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                         replace={"type": "boolean"}), "required": ["items"]}}},
    {"type": "function", "function": {"name": "calibrate", "description": "Set the scale. mode=auto: from printed m² of registered rooms (needs rooms). mode=dimension: cm value and two approximate end points of a dimension line.",
     "parameters": {**_I(mode={"type": "string", "enum": ["auto", "dimension"]}, cm={"type": "number"},
                         x1={"type": "integer"}, y1={"type": "integer"}, x2={"type": "integer"}, y2={"type": "integer"}), "required": ["mode"]}}},
    {"type": "function", "function": {"name": "walls", "description": "Compute wall mask and room cells from your points, barriers and erasures. Returns per-room computed vs printed m², ratio, wall stickiness, flag, and an overview image.",
     "parameters": _I()}},
    {"type": "function", "function": {"name": "barrier", "description": "Add a wall line (e.g. across a door between its jamb ends, or along a missing wall face) so cells cannot leak through. Takes effect at the next walls call.",
     "parameters": {**_I(x1={"type": "integer"}, y1={"type": "integer"}, x2={"type": "integer"}, y2={"type": "integer"}), "required": ["x1", "y1", "x2", "y2"]}}},
    {"type": "function", "function": {"name": "erase", "description": "Remove false wall pieces (furniture, dimension line, hatch) inside a box so the room is not cut. Takes effect at the next walls call.",
     "parameters": {**_I(x0={"type": "integer"}, y0={"type": "integer"}, x1={"type": "integer"}, y1={"type": "integer"}), "required": ["x0", "y0", "x1", "y1"]}}},
    {"type": "function", "function": {"name": "polygon", "description": "Last resort: set a room's boundary by hand (corners in floor px, clockwise). Overrides its cell.",
     "parameters": {**_I(id={"type": "string"}, points={"type": "array", "items": {"type": "array", "items": {"type": "number"}}}), "required": ["id", "points"]}}},
    {"type": "function", "function": {"name": "check", "description": "Check all current room polygons: overlaps, self-crossings, ratio summary, rooms without a polygon. Returns text and an overview image.",
     "parameters": _I()}},
    {"type": "function", "function": {"name": "export", "description": "Write the final result (all registered rooms with their polygons, names, m², notes) and submit it. Ends the task.",
     "parameters": {**_I(notes={"type": "object", "description": "optional {room_id: note}"})}}},
]


def _st(ctx):
    s = ctx["state"]
    s.setdefault("rooms", []); s.setdefault("outside", []); s.setdefault("barriers", []); s.setdefault("erasures", [])
    s.setdefault("ppc", None); s.setdefault("manual", {}); s.setdefault("polys", {}); s.setdefault("gray", None)
    if s["gray"] is None:
        s["gray"] = G.load_gray(ctx["image"])
    return s


def _m2(r):
    try:
        v = float(r.get("m2"))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _font(n=14):
    try:
        return ImageFont.truetype("arial.ttf", n)
    except Exception:
        return ImageFont.load_default()


def handle(name, args, ctx):
    s = _st(ctx)
    if name == "view":
        return _view(s, ctx, args)
    if name == "rooms":
        items = args.get("items") or []
        if args.get("replace"):
            s["rooms"] = []
        byid = {r["id"]: r for r in s["rooms"]}
        for it in items:
            rid = str(it.get("id") or f"r{len(byid) + 1:02d}")
            r = byid.get(rid, {"id": rid})
            for k in ("name", "m2", "type", "flat", "x", "y", "extra"):
                if k in it:
                    r[k] = it[k]
            byid[rid] = r
        s["rooms"] = list(byid.values())
        if args.get("outside") is not None:
            s["outside"] = [list(map(float, p)) for p in args["outside"]]
        return f"{len(s['rooms'])} rooms registered, {len(s['outside'])} outside points. " + \
            ", ".join(f"{r['id']}={r.get('name')} {r.get('m2')}" for r in s["rooms"]), _mark_rooms(s), None
    if name == "calibrate":
        return _calibrate(s, args)
    if name == "walls":
        return _walls(s)
    if name == "barrier":
        b = [int(args[k]) for k in ("x1", "y1", "x2", "y2")]
        s["barriers"].append(b)
        return f"barrier {b} added ({len(s['barriers'])} total); call walls to recompute", None, None
    if name == "erase":
        b = [int(args[k]) for k in ("x0", "y0", "x1", "y1")]
        s["erasures"].append(b)
        return f"erase box {b} added ({len(s['erasures'])} total); call walls to recompute", None, None
    if name == "polygon":
        rid = str(args["id"]); pts = [[int(round(x)), int(round(y))] for x, y in args["points"]]
        s["manual"][rid] = pts; s["polys"][rid] = pts
        r = next((r for r in s["rooms"] if r["id"] == rid), {})
        msg = f"{rid}: manual polygon with {len(pts)} corners"
        if s["ppc"] and _m2(r):
            a = G.area_m2(pts, s["ppc"]); msg += f", {a:.2f} m2 vs printed {r['m2']} (ratio {a / float(r['m2']):.3f})"
        return msg, None, None
    if name == "check":
        return _check(s)
    if name == "export":
        return _export(s, ctx, args.get("notes") or {})
    return f"unknown tool {name}", None, None


def _view(s, ctx, a):
    img = Image.open(ctx["image"]).convert("RGB")
    x0, y0, x1, y1 = (int(a.get(k, 0)) for k in ("x0", "y0", "x1", "y1"))
    x0, y0 = max(0, min(x0, x1)), max(0, min(y0, y1)); x1, y1 = min(img.width, max(x0 + 10, x1)), min(img.height, max(y0 + 10, y1))
    crop = img.crop((x0, y0, x1, y1))
    sc = min(1400 / crop.width, 1400 / crop.height, 6.0)
    crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(crop, "RGBA"); f = _font(13)
    step = int(a.get("grid") or 50)
    if step > 0:
        for gx in range((x0 // step + 1) * step, x1, step):
            X = (gx - x0) * sc; d.line([(X, 0), (X, crop.height)], fill=(255, 0, 0, 90), width=1); d.text((X + 2, 2), str(gx), fill=(200, 0, 0), font=f)
        for gy in range((y0 // step + 1) * step, y1, step):
            Y = (gy - y0) * sc; d.line([(0, Y), (crop.width, Y)], fill=(255, 0, 0, 90), width=1); d.text((2, Y + 2), str(gy), fill=(200, 0, 0), font=f)
    if a.get("overlay"):
        T = lambda p: ((p[0] - x0) * sc, (p[1] - y0) * sc)
        for rid, pts in s["polys"].items():
            if pts:
                q = [T(p) for p in pts]; d.line(q + q[:1], fill=(210, 0, 210, 230), width=3)
                c = Polygon(pts).representative_point(); d.text(T((c.x, c.y)), rid, fill=(150, 0, 150), font=_font(16))
        for b in s["barriers"]:
            d.line([T(b[:2]), T(b[2:])], fill=(0, 120, 255, 255), width=4)
        for r in s["rooms"]:
            for (px, py) in [(r.get("x"), r.get("y"))] + [tuple(e) for e in (r.get("extra") or [])]:
                if px is not None:
                    X, Y = T((px, py)); d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=(0, 160, 0, 220))
    return f"view x {x0}-{x1}, y {y0}-{y1}, grid {step} px, shown {sc:.2f}x" + (" with cells (magenta), barriers (blue), your points (green)" if a.get("overlay") else ""), crop, None


def _mark_rooms(s):
    gray = s["gray"]; im = Image.fromarray(gray).convert("RGB")
    sc = min(1.0, 1400 / max(im.size)); im = im.resize((int(im.width * sc), int(im.height * sc)))
    d = ImageDraw.Draw(im); f = _font(14)
    for r in s["rooms"]:
        if r.get("x") is None:
            continue
        X, Y = r["x"] * sc, r["y"] * sc
        d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=(0, 170, 0)); d.text((X + 6, Y - 8), r["id"], fill=(0, 120, 0), font=f)
    for (x, y) in s["outside"]:
        d.ellipse([x * sc - 5, y * sc - 5, x * sc + 5, y * sc + 5], fill=(255, 140, 0))
    return im


def _calibrate(s, a):
    if a.get("mode") == "dimension":
        try:
            dist = math.hypot(a["x2"] - a["x1"], a["y2"] - a["y1"]); dim = dist / float(a["cm"])
        except Exception as e:
            return f"calibrate error: {e}", None, None
        msg = f"scale {dim:.4f} px/cm from a {a['cm']} cm dimension ({dist:.1f} px)"
        auto = _auto_ppc(s)
        s["ppc"] = dim
        if auto and abs(dim - auto) / auto > 0.10:
            msg += (f"\nWARNING: the printed room areas give {auto:.4f} px/cm ({(dim / auto - 1):+.0%} off). One of them is wrong: "
                    f"a dimension often spans a different stretch than you think (chain segment, wall axis vs face). "
                    f"Re-check the end ticks, or measure a second dimension. Scale kept at {dim:.4f}.")
        elif auto:
            msg += f" - agrees with the printed areas ({auto:.4f})"
        return msg, None, None
    ppc = _auto_ppc(s)
    if not ppc:
        return "auto calibration needs at least 3 registered rooms with printed m2", None, None
    s["ppc"] = ppc
    return f"scale {ppc:.4f} px/cm (1 px = {1 / ppc:.2f} cm), from the median of the registered rooms (auto)", None, None


def _auto_ppc(s):
    """Scale from printed m2 vs seeded cells. Always starts from a neutral 0.9 so a wrong current scale cannot lock in."""
    rooms = [r for r in s["rooms"] if _m2(r) and r.get("x") is not None and r.get("type", "room") != "shaft"]
    if len(rooms) < 3:
        return None
    ppc = 0.9
    for _ in range(4):   # cells depend on the scale (wall thresholds): iterate
        cells = _cells(s, ppc)
        rat = [math.sqrt(cells[r["id"]].sum() / 255 / (_m2(r) * 1e4)) for r in rooms if cells[r["id"]].any()]
        if not rat:
            return None
        new = float(np.median(rat)); done = abs(new - ppc) / ppc < 0.02; ppc = new
        if done:
            break
    return ppc


def _wall(s, ppc):
    w = G.wall_mask(s["gray"], ppc)
    for b in s["erasures"]:
        w[b[1]:b[3], b[0]:b[2]] = 0
    th = max(3, int(round(3 * ppc)))
    for b in s["barriers"]:
        cv2.line(w, tuple(b[:2]), tuple(b[2:]), 255, th)
    return w


def _cells(s, ppc):
    w = _wall(s, ppc)
    seeds = [(r["id"], [(r["x"], r["y"])] + [tuple(e) for e in (r.get("extra") or [])]) for r in s["rooms"] if r.get("x") is not None]
    s["_wall"] = w
    return G.cells(w, seeds, outside=[tuple(p) for p in s["outside"]], px_per_cm=ppc)


def _walls(s):
    if not s["rooms"]:
        return "register rooms first", None, None
    if not s["ppc"]:
        _calibrate(s, {"mode": "auto"})
        if not s["ppc"]:
            return "set the scale first (calibrate)", None, None
    ppc = s["ppc"]
    cells = _cells(s, ppc)
    lines = [f"scale {ppc:.4f} px/cm. id | name | printed m2 | computed m2 | ratio | on-wall | flag"]
    ratios = []
    for r in s["rooms"]:
        rid = r["id"]
        if rid in s["manual"]:
            pts = s["manual"][rid]
        else:
            m = cells.get(rid)
            pts = G.mask_to_polygon(m) if m is not None and m.any() else None
            s["polys"][rid] = pts
        if not pts:
            lines.append(f"{rid} | {r.get('name')} | {r.get('m2')} | - | - | - | NO CELL (point on a wall or outside?)"); continue
        a = G.area_m2(pts, ppc); stick = G.wall_stickiness(pts, s["_wall"])
        ratio = a / _m2(r) if _m2(r) else None
        flag = "" if ratio is None else ("OK" if 0.93 <= ratio <= 1.07 else ("TOO BIG" if ratio > 1 else "TOO SMALL"))
        if ratio is not None:
            ratios.append(ratio)
        if flag == "TOO SMALL" and a < 0.5:
            flag += " - seed sits in a tiny pocket (furniture/text/door swing): move x,y to open floor"
        elif flag == "TOO BIG" and ratio > 1.6:
            flag += " - leaks into a neighbour: find the door/gap and add a barrier, or a seed is missing next door"
        lines.append(f"{rid} | {r.get('name')} | {r.get('m2')} | {a:.2f} | {'-' if ratio is None else f'{ratio:.3f}'} | {stick:.0%} | {flag}"
                     + (" (manual)" if rid in s["manual"] else ""))
    nok = sum("| OK" in l for l in lines); nbad = sum(("TOO" in l) or ("NO CELL" in l) for l in lines)
    lines.append(f"summary: {nok} OK, {nbad} flagged")
    mid = [x for x in ratios if 0.2 < x < 5]
    if len(mid) >= 5:
        med = float(np.median(mid))
        if med < 0.85 or med > 1.18:
            lines.append(f"WARNING: the median ratio is {med:.2f} - most rooms are off in the same direction, so the SCALE is "
                         f"probably wrong (not the walls). Run calibrate mode=auto and compare before fixing rooms one by one.")
    return "\n".join(lines), _overview(s), None


def _overview(s):
    gray = s["gray"]; im = Image.fromarray(gray).convert("RGB")
    w = s.get("_wall")
    if w is not None:
        a = np.array(im); a[w > 0] = (a[w > 0] * 0.5 + np.array([120, 120, 255]) * 0.5).astype(np.uint8); im = Image.fromarray(a)
    sc = min(1.0, 1400 / max(im.size)); im = im.resize((int(im.width * sc), int(im.height * sc)))
    d = ImageDraw.Draw(im, "RGBA"); f = _font(13)
    for r in s["rooms"]:
        pts = s["polys"].get(r["id"])
        if pts:
            q = [(x * sc, y * sc) for x, y in pts]
            ok = _m2(r) and 0.93 <= G.area_m2(pts, s["ppc"]) / float(r["m2"]) <= 1.07
            d.polygon(q, fill=(0, 200, 0, 50) if ok else (255, 0, 0, 60), outline=(0, 140, 0) if ok else (220, 0, 0))
            c = Polygon(pts).representative_point(); d.text((c.x * sc, c.y * sc), r["id"], fill=(0, 0, 0), font=f)
    for b in s["barriers"]:
        d.line([(b[0] * sc, b[1] * sc), (b[2] * sc, b[3] * sc)], fill=(0, 120, 255), width=3)
    return im


def _check(s):
    polys = {rid: p for rid, p in s["polys"].items() if p}
    out = []
    ids = list(polys)
    for i in range(len(ids)):
        Pi = Polygon(polys[ids[i]])
        if not Pi.is_valid:
            out.append(f"{ids[i]}: self-crossing")
        for j in range(i + 1, len(ids)):
            inter = Pi.buffer(0).intersection(Polygon(polys[ids[j]]).buffer(0)).area
            if inter > 0.02 * min(Pi.area, Polygon(polys[ids[j]]).area):
                out.append(f"{ids[i]} and {ids[j]} overlap ({inter:.0f} px2)")
    missing = [r["id"] for r in s["rooms"] if not polys.get(r["id"])]
    if missing:
        out.append("no polygon: " + ", ".join(missing))
    txt, im, _ = _walls(s) if s["rooms"] else ("", None, None)
    return ("\n".join(out) or "no overlaps or self-crossings") + "\n" + txt.split("\n")[-1], im, None


def _export(s, ctx, notes):
    rooms = []
    for r in s["rooms"]:
        pts = s["polys"].get(r["id"])
        st = "uncertain"
        if pts and _m2(r) and s["ppc"]:
            ratio = G.area_m2(pts, s["ppc"]) / float(r["m2"]); st = "final" if 0.93 <= ratio <= 1.07 else "uncertain"
        rooms.append({"id": r["id"], "name": r.get("name"), "printed_area_m2": r.get("m2"), "type": r.get("type", "room"),
                      "flat": r.get("flat"), "source_labels": [f"{r.get('name')} {r.get('m2')}"], "boundary": pts,
                      "boundary_status": st, "note": (notes or {}).get(r["id"], "")})
    floor = ctx["image"].stem.replace("kat_", "")
    res = {"floor": floor, "image": ctx["image"].name, "rooms": rooms}
    (ctx["out"] / "TOOL_STATE.json").write_text(json.dumps({k: v for k, v in s.items() if not k.startswith("_") and k != "gray" and k != "polys"},
                                                          ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return f"exported {len(rooms)} rooms", None, res
