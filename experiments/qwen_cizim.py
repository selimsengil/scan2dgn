"""Qwen's drawing platform: Qwen draws every room boundary itself; the platform remembers everything for it.

- Phases are enforced: UNDERSTAND (plan) -> INVENTORY (rooms) -> SCALE -> SKETCH (rough outline of every room)
  -> REFINE (room by room, precise) -> REVIEW (check) -> EXPORT. Tools that belong to a later phase refuse early.
- Every tool answer ends with the BOARD: phase and what to do now, Qwen's own plan with ticks, progress counts,
  rooms not drawn / flagged, recent changes (who did what: YOU or TOOL), and the suggested next action.
  Qwen never has to remember where it is.
- Every draw is a new version. history(id) shows all versions of a room, text and picture. Pictures separate what
  Qwen drew (blue) from what the tool changed (orange = magnet moves) and from earlier versions (grey).
- Wall lines Qwen recognised can be recorded (wall); each draw says which recorded walls and which neighbour edges its
  edges sit on, so Qwen follows walls from room to room instead of re-reading them.
- Coordinates: floor pixels (the grid labels), or Qwen's native grounding frame: 0-1000 relative to a view image
  (frame = that view's id). The platform converts.
- The magnet (snap) is optional and off by default.
Plug into kat_agent_api.py --tool-module qwen_cizim.
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point, Polygon

import qwen_geo as G

PHASES = {
    "UNDERSTAND": "Look at the whole floor (view), understand it: how many flats, common areas, where the labels are, how walls "
                  "are drawn here. Then call plan with what you understood and your step list.",
    "INVENTORY": "Read every room label in parts (view, grid 100) and register them ALL in one rooms call (id, name, m2, type, "
                 "flat, label_xy).",
    "SKETCH": "Give EVERY room a rough outline, many rooms per draw call (draw rooms=[{id, points}, ...]); corners within "
              "~10 px are fine. This lays out the whole floor and the shared walls before the detailed work.",
    "WALK": "Walk every room along its inner wall faces (trace, then walk) - this is where columns, notches, pilasters and "
            "door closings are caught. Start each room where the previous one ended when they share a wall.",
    "REVIEW": "Call check. Fix overlaps and weak edges (re-walk that part), or write a note why a room cannot be better.",
    "EXPORT": "Everything is walked and checked: export.",
}

GUIDE = """YOU ARE WORKING ON A DRAWING PLATFORM. You draw every room boundary yourself; the platform remembers for you.
Goal: for every room on the floor, the polygon of its INNER wall faces (where the wall ink meets the room floor; doors
closed straight across on the wall's inner face line), with its name and printed m2 read from its label.
Not walls: furniture, text, dimension lines, door swings, hatch. Columns and pilasters ARE part of the wall: the
boundary goes around them (notches).
The printed m2 is recorded for the result only. It is NOT a check of your drawing - printed areas are often wrong or
rounded. The walls are the only truth.

The platform enforces the order: UNDERSTAND -> INVENTORY -> SKETCH -> WALK -> REVIEW -> EXPORT.
Time budget: about 45 minutes for the whole floor (the board shows the time). Be quick in routine steps, careful at
corners, doors and columns.
Every tool answer ends with a BOARD: your phase and what to do in it, your plan with ticks, what is drawn / to walk /
has problems, the last changes (YOU = your calls, TOOL = something the platform changed), and a suggested next step.
Trust the board instead of your memory. Use plan(done=[...], note=...) to tick steps and keep notes for later.
Thinking: routine pen steps get a short thinking budget. If a spot is tricky (a column, a step in the wall, an
unclear door), pass hard=true and the next step gets more thinking. After an error or warning you get more too.

Seeing your own work: every draw returns a picture of that room (your outline blue, the previous version grey,
neighbours magenta, recorded walls green, magnet moves orange). history(id) shows all versions of a room.
view(..., overlay=true) shows everything you have drawn in any region.

Follow the walls: when you have read a wall's inner face, record it with wall(id, x1,y1,x2,y2, note). A wall runs on
past one room: neighbours along it share the same line, rooms across it sit one wall thickness further. Each draw
tells you which recorded walls and neighbour edges your edges are on - reuse those exact numbers.

Coordinates: floor pixels (the red grid labels), or - your native grounding format - 0-1000 relative to a view image:
pass frame="<view id>" (every view answer starts with its id, e.g. V7) and give points as [x,y] in 0-1000 of that image.
Zoom freely: view(frame="V7", x0..y1 in 0-1000 of V7) zooms into part of V7; every zoom gives a finer 0-1000 frame.
You cannot get lost: each view comes with a second small picture, the minimap of the whole floor (red box = where you are), and the board says which view you are in,
which of your rooms are there, and your recent views. view(room=id) brings you back to any room.

Walking the walls (WALK phase, every room): trace(id) puts the pen on the top-left corner of your sketch, heading
clockwise (or give x, y, heading yourself). Each answer gives three pictures - DETAIL (close-up ahead of the pen, your frame for
the next points), CONTEXT (4x wider: your trail, the pen, the detail box) and the MINIMAP - plus a measured profile of
the line ahead (where it runs, where it breaks - door-sized breaks are marked - and where a line turns off). Then
walk(points, events, heading): put the next corners / door_start / door_end where you see them - several per call
when you can see them. Zoom in with window=100 at a column or a tricky spot, out with window=400 on long straight walls. When you come back to the start
point the room closes by itself. You never lose your place: the trail stays drawn in every view and the board says
where the pen is. undo removes points; trace(id) resumes an unfinished trace.

Feedback per room: "on a line %" per edge (is the edge on a drawn line?), shared walls with neighbours, overlaps.
Below 35 % = that edge is not on a line: you misread it (a straight door closing still touches the jambs and passes).
A fix is right when the edge lands on the line.
Every reply must end in a tool call. Put several corners in one walk call when you can see them."""

_I = lambda **p: {"type": "object", "properties": p}
_PTS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
TOOLS = [
    {"type": "function", "function": {"name": "plan", "description": "Write or update your plan. First call: understanding (what the floor is like) and steps (your ordered step list). Later: done=[step numbers finished], note='something to remember'.",
     "parameters": _I(understanding={"type": "string"}, steps={"type": "array", "items": {"type": "string"}},
                      done={"type": "array", "items": {"type": "integer"}}, note={"type": "string"})}},
    {"type": "function", "function": {"name": "view", "description": "Look at a region, zoomed to fit. Box x0,y0,x1,y1 in floor px, OR in 0-1000 of an earlier view with frame=<view id> (zoom into part of what you saw), OR room=<id> to look around one of your rooms. overlay=true draws your outlines and recorded walls. Every picture has a minimap of the whole floor. The answer starts with the new view id. No box = whole floor.",
     "parameters": _I(x0={"type": "number"}, y0={"type": "number"}, x1={"type": "number"}, y1={"type": "number"},
                      frame={"type": "string"}, room={"type": "string"},
                      grid={"type": "integer", "description": "grid step in floor px (default 50)"}, overlay={"type": "boolean"})}},
    {"type": "function", "function": {"name": "rooms", "description": "Register or update rooms. Each item: id, name, m2 (printed, null if none), type (room|common|shaft), flat, label_xy (optional [x,y] of its label). replace=true replaces the list.",
     "parameters": {**_I(items={"type": "array", "items": {"type": "object"}}, replace={"type": "boolean"}), "required": ["items"]}}},
    {"type": "function", "function": {"name": "calibrate", "description": "Optional scale from a dimension line (cm and its two end points, frame optional). Only used to express measured gaps and wall thicknesses in cm.",
     "parameters": {**_I(cm={"type": "number"}, x1={"type": "number"}, y1={"type": "number"}, x2={"type": "number"}, y2={"type": "number"},
                         frame={"type": "string"}), "required": ["cm", "x1", "y1", "x2", "y2"]}}},
    {"type": "function", "function": {"name": "wall", "description": "Record a wall inner-face line you recognised (id, two end points, note e.g. 'top wall of flat 9, rooms below it'). Recorded walls are drawn green and matched against your edges.",
     "parameters": {**_I(id={"type": "string"}, x1={"type": "number"}, y1={"type": "number"}, x2={"type": "number"}, y2={"type": "number"},
                         note={"type": "string"}, frame={"type": "string"}), "required": ["id", "x1", "y1", "x2", "y2"]}}},
    {"type": "function", "function": {"name": "draw", "description": "Set room outlines (corners clockwise on the inner wall faces). Batch: rooms=[{id, points}, ...] stores many rough sketches at once and returns an overview picture. Single: id + points returns a per-edge report (on a wall line?), relations to walls and neighbours, and a close-up picture. frame: omit for floor px, or a view id for 0-1000 of that view. snap: optional magnet radius in px (default 0 = off).",
     "parameters": _I(rooms={"type": "array", "items": {"type": "object"}}, id={"type": "string"}, points=_PTS,
                      mode={"type": "string", "enum": ["sketch", "drawn"]}, frame={"type": "string"}, snap={"type": "integer"},
                      why={"type": "string", "description": "what you changed and why (kept in the history)"})}},
    {"type": "function", "function": {"name": "trace", "description": "Start (or resume) walking along the inner wall faces of one room, like tracing with a pen. Give the room id, a start point ON the inner face line (best a corner), and the heading (E, W, N, S) to walk. Each answer shows: DETAIL (close-up ahead of the pen), CONTEXT (4x wider, your trail and the detail box marked), MINIMAP, and a measured profile of the line ahead. action=undo removes the last n points; action=abort drops the trace.",
     "parameters": {**_I(id={"type": "string"}, x={"type": "number"}, y={"type": "number"}, heading={"type": "string", "enum": ["E", "W", "N", "S"]},
                         frame={"type": "string"}, window={"type": "integer", "description": "detail window length in floor px (default 160; smaller = closer)"},
                         action={"type": "string", "enum": ["start", "resume", "undo", "abort"]}, n={"type": "integer"},
                         hard={"type": "boolean"}), "required": ["id"]}}},
    {"type": "function", "function": {"name": "walk", "description": "Continue the trace: give the next point(s) where the inner face line turns, or a door/opening starts or ends, with an event per point (corner, door_start, door_end, column, step, end). frame = the DETAIL view id for 0-1000 coordinates (or omit for floor px). heading = the new walking direction after the last point. Returning to the start point closes the room.",
     "parameters": {**_I(points=_PTS, events={"type": "array", "items": {"type": "string"}}, heading={"type": "string", "enum": ["E", "W", "N", "S"]},
                         frame={"type": "string"}, window={"type": "integer"}, note={"type": "string"},
                         hard={"type": "boolean", "description": "true = this spot is tricky, give me more thinking on the next step"}), "required": ["points"]}}},
    {"type": "function", "function": {"name": "history", "description": "All versions of one room (who changed what, numbers per version) with a picture of every version. Without id: the full change log.",
     "parameters": _I(id={"type": "string"})}},
    {"type": "function", "function": {"name": "check", "description": "Review all rooms: overlaps, self-crossings, ratio table, rooms not drawn, rooms still at sketch level. Returns text and an overview picture.",
     "parameters": _I()}},
    {"type": "function", "function": {"name": "export", "description": "Write the final result and submit it. Allowed after check, when every room is drawn or has a note.",
     "parameters": {**_I(notes={"type": "object", "description": "optional {room_id: note}"})}}},
]

ORDER = ["UNDERSTAND", "INVENTORY", "SKETCH", "WALK", "REVIEW", "EXPORT"]
import os as _os
TIME_BUDGET_MIN = int(_os.environ.get("QWEN_TIME_BUDGET_MIN", "45"))


def _st(ctx):
    s = ctx["state"]
    import time as _t
    for k, v in (("rooms", []), ("ppc", None), ("polys", {}), ("gray", None), ("plan", None), ("versions", {}),
                 ("log", []), ("walls", {}), ("views", {}), ("n", 0), ("checked_at", -1), ("last_change", 0),
                 ("dims", []), ("notes", {}), ("t0", _t.time()), ("hard", False), ("trouble", False)):
        s.setdefault(k, v)
    if s["gray"] is None:
        s["gray"] = G.load_gray(ctx["image"])
        t, _ = cv2.threshold(s["gray"], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        s["_ink"] = s["gray"] < min(max(t, 120), 200)
    return s


def _m2(r):
    try:
        v = float((r or {}).get("m2"))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _font(n=14):
    try:
        return ImageFont.truetype("arial.ttf", n)
    except Exception:
        return ImageFont.load_default()


def _room(s, rid):
    return next((r for r in s["rooms"] if r["id"] == rid), None)


def _log(s, who, what):
    s["log"].append(f"#{s['n']} {who}: {what}")


# ---------------------------------------------------------------- phases and board
def phase(s):
    if not s["plan"]:
        return "UNDERSTAND"
    if not s["rooms"]:
        return "INVENTORY"
    if any(r["id"] not in s["polys"] and r["id"] not in s["notes"] for r in s["rooms"]):
        return "SKETCH"
    if any(_level(s, r["id"]) != "walked" for r in s["rooms"] if r["id"] not in s["notes"]):
        return "WALK"
    if s["checked_at"] < s["last_change"]:
        return "REVIEW"
    return "EXPORT"


def _level(s, rid):
    """sketch | drawn (precise draw, not walked) | walked"""
    v = s["versions"].get(rid)
    return v[-1]["mode"] if v else None


def line_presence(ink, h, c, a, b, tol=1):
    """Share of the edge (inner 80 %) with a dark line within +-tol px. Tuned on dev floors 019/013/010:
    below 0.35 flags 3 % of true DGN edges (door closings included) and 87 % of edges 8-15 px off the wall."""
    lo, hi = sorted((a, b)); m = int(0.1 * (hi - lo)); lo, hi = lo + m, hi - m
    if hi - lo < 3:
        return 1.0
    H, W = ink.shape
    t = np.arange(lo, hi + 1); t = t[(t >= 0) & (t < (W if h else H))]
    v = np.zeros(len(t), bool)
    for r in range(c - tol, c + tol + 1):
        if 0 <= r < (H if h else W):
            v |= ink[r, t] if h else ink[t, r]
    return float(v.mean()) if len(t) else 0.0


def _edge_presence(s, A, B, kind):
    h = kind == "h"
    return line_presence(s["_ink"], h, A[1] if h else A[0], A[0] if h else A[1], B[0] if h else B[1])


def _weak_edges(s, P):
    """h/v edges (12 px or longer) that do not sit on a dark line."""
    weak = []
    for i, A, B, kind, inward in _edges(P):
        if kind == "o":
            continue
        L = abs(B[0] - A[0]) if kind == "h" else abs(B[1] - A[1])
        if L >= 12 and _edge_presence(s, A, B, kind) < 0.35:
            weak.append(i)
    return weak


def _overlaps(s, rid):
    P = s["polys"].get(rid)
    if not P:
        return []
    me, out = Polygon(P).buffer(0), []
    for oid, Q in s["polys"].items():
        if oid != rid and Q:
            a = me.intersection(Polygon(Q).buffer(0)).area
            if a > 0.01 * min(me.area, Polygon(Q).area) and a > 20:
                out.append(oid)
    return out


def _flag(s, rid):
    """Quality from the walls, never from m2: '-' not drawn, 'sketch', 'OVERLAP', 'WEAK' (edges off the wall lines),
    'OK' (walked or drawn, on wall lines, no overlap)."""
    P = s["polys"].get(rid)
    if not P:
        return "-"
    if _level(s, rid) == "sketch":
        return "sketch"
    if _overlaps(s, rid):
        return "OVERLAP"
    if _weak_edges(s, P):
        return "WEAK"
    return "OK"


def _elapsed_min(s):
    import time as _t
    return (_t.time() - s["t0"]) / 60


def think_budget(ctx):
    """Thinking tokens for the NEXT model call, chosen by the platform from the situation (used by kat_agent_api).
    Long where a decision is made or something went wrong, short for routine pen steps."""
    s = ctx.get("state") or {}
    if not s.get("plan"):
        return 2500
    if s.get("hard") or s.get("trouble"):
        return 2000
    ph = phase(_st(ctx))
    t = (s.get("traces") or {}).get(s.get("active")) if s.get("active") else None
    if ph == "WALK":
        if t and len(t["pts"]) > 1:
            return 500       # routine step along a wall
        return 1200          # choosing where to start the next room
    return {"INVENTORY": 1500, "SKETCH": 1500, "REVIEW": 1500, "EXPORT": 800}.get(ph, 1500)


def board(s):
    ph = phase(s)
    el = _elapsed_min(s)
    L = [f"\n===== BOARD (action #{s['n']}, time {el:.0f} of {TIME_BUDGET_MIN} min) =====",
         f"PHASE {ORDER.index(ph) + 1}/{len(ORDER)} {ph}: {PHASES[ph]}"]
    if el > 0.8 * TIME_BUDGET_MIN and ph not in ("REVIEW", "EXPORT"):
        L.append("TIME IS SHORT: finish the rooms still open with fewer, longer walk steps (window=300), then check and export.")
    if s["plan"]:
        L.append("YOUR PLAN: " + s["plan"]["understanding"][:400])
        for i, st in enumerate(s["plan"]["steps"], 1):
            L.append(f"  [{'x' if i in s['plan']['done'] else ' '}] {i}. {st}")
        if s["plan"]["notes"]:
            L.append("YOUR NOTES: " + " | ".join(s["plan"]["notes"][-6:]))
    if s["rooms"]:
        ids = [r["id"] for r in s["rooms"]]
        missing = [i for i in ids if i not in s["polys"] and i not in s["notes"]]
        walked = [i for i in ids if _level(s, i) == "walked"]
        todo = [i for i in ids if i in s["polys"] and _level(s, i) != "walked" and i not in s["notes"]]
        fl = {i: _flag(s, i) for i in ids if i in s["polys"]}
        bad = [f"{i}({f})" for i, f in fl.items() if f in ("OVERLAP", "WEAK")]
        L.append(f"ROOMS {len(ids)}: walked {len(walked)}, to walk {len(todo)}, not drawn {len(missing)}"
                 + (f", noted {len(s['notes'])}" if s["notes"] else "") + f", problems {len(bad)}")
        if missing:
            L.append("  not drawn: " + ", ".join(missing[:25]) + (" ..." if len(missing) > 25 else ""))
        if todo and ph == "WALK":
            L.append("  to walk: " + ", ".join(todo[:25]) + (" ..." if len(todo) > 25 else ""))
        if bad:
            L.append("  problems: " + ", ".join(bad[:20]))
    if s["ppc"]:
        L.append(f"SCALE: {s['ppc']:.4f} px/cm (optional, used for cm in the measurements)")
    if s["walls"]:
        L.append(f"RECORDED WALLS: {len(s['walls'])} (" + ", ".join(list(s["walls"])[-8:]) + ")")
    t = (s.get("traces") or {}).get(s.get("active")) if s.get("active") else None
    if t:
        L.append(f"TRACING {s['active']}: {len(t['pts'])} points, pen at {t['pts'][-1]} heading {t['heading']}, "
                 f"started at {t['pts'][0]}; last events: {', '.join(t['events'][-4:])}. Continue with walk, or trace(action=undo).")
    open_t = [r for r, x in (s.get("traces") or {}).items() if not x.get("closed") and r != s.get("active")]
    if open_t:
        L.append("UNFINISHED TRACES: " + ", ".join(open_t) + " (trace(id) resumes)")
    if s.get("trail"):
        vid = s["trail"][-1]; x0, y0, x1, y1 = s["views"][vid]
        inside = _rooms_in(s, (x0, y0, x1, y1))
        L.append(f"YOU ARE LOOKING AT: {vid} = x {x0}-{x1}, y {y0}-{y1}" + (f", your rooms there: {', '.join(inside[:8])}" if inside else "")
                 + f". Recent views: {' > '.join(f'{v}({s['views'][v][0]},{s['views'][v][1]})' for v in s['trail'][-4:])}")
    if s["log"]:
        L.append("LAST CHANGES: " + " | ".join(s["log"][-5:]))
    L.append("NEXT: " + _next(s, ph))
    return "\n".join(L)


def _next(s, ph):
    if ph == "SKETCH":
        m = [r["id"] for r in s["rooms"] if r["id"] not in s["polys"] and r["id"] not in s["notes"]]
        return f"sketch {', '.join(m[:8])}{' ...' if len(m) > 8 else ''} in one draw(rooms=[...]) call"
    if ph == "WALK":
        if s.get("active"):
            return f"continue walking {s['active']} (walk)"
        todo = [r["id"] for r in s["rooms"] if r["id"] in s["polys"] and _level(s, r["id"]) != "walked" and r["id"] not in s["notes"]]
        if not todo:
            return "check"
        last = next((x for x in reversed(s["log"]) if "closed" in x), None)
        ref = None
        for rid, v in reversed(list(s["versions"].items())):
            if v and v[-1]["mode"] == "walked":
                ref = Polygon(v[-1]["final"]).centroid; break
        if ref is not None:   # nearest unwalked room: shared walls continue
            todo.sort(key=lambda r: Polygon(s["polys"][r]).centroid.distance(ref))
        return f"trace({todo[0]}) - starts at a corner of its sketch; then walk" + (f" (next nearest: {', '.join(todo[1:3])})" if len(todo) > 1 else "")
    return {"UNDERSTAND": "view the whole floor, then plan", "INVENTORY": "view the floor in parts and register all rooms",
            "REVIEW": "check", "EXPORT": "export"}[ph]


GATES = {"rooms": "INVENTORY", "draw": "SKETCH", "check": "SKETCH", "export": "REVIEW", "trace": "WALK", "walk": "WALK"}


def handle(name, args, ctx):
    s = _st(ctx)
    s["n"] += 1
    need = GATES.get(name)
    if need and ORDER.index(phase(s)) < ORDER.index(need):
        return f"NOT YET: {name} belongs to phase {need}; you are in {phase(s)}." + board(s), None, None
    pre = ""
    s["hard"] = bool(args.get("hard"))   # Qwen asks for more thinking on the next step
    s["trouble"] = False
    fn = {"plan": _plan, "view": _view, "rooms": _rooms, "calibrate": _calibrate, "wall": _wall, "draw": _draw,
          "history": _history, "check": _check, "export": _export, "trace": _trace, "walk": _walk}.get(name)
    if not fn:
        return f"unknown tool {name}" + board(s), None, None
    if name == "view" and "overlay" not in args and s["polys"]:
        args = dict(args, overlay=True)
    s["_mini"] = None
    txt, im, res = fn(s, ctx, args)
    if res is not None:
        return txt, im, res
    if any(k in txt for k in ("NOT YET", "error", "OVERLAPS", "not on a wall line", "unknown tool")):
        s["trouble"] = True   # something went wrong: the next step gets a longer thinking budget
    if isinstance(im, list):
        pass   # tool assembled its own picture set (trace/walk)
    elif im is not None and s.get("_mini") is not None:
        im = [im, s["_mini"]]
    return pre + txt + board(s), im, None


# ---------------------------------------------------------------- coordinates
def _to_floor(s, pts, frame):
    """frame None -> floor px; 'V<n>' -> 0-1000 relative to that view's image."""
    if not frame:
        return [[int(round(float(x))), int(round(float(y)))] for x, y in pts]
    v = s["views"].get(str(frame))
    if not v:
        raise ValueError(f"unknown view id {frame} (known: {', '.join(list(s['views'])[-5:])})")
    x0, y0, x1, y1 = v
    return [[int(round(x0 + float(x) / 1000 * (x1 - x0))), int(round(y0 + float(y) / 1000 * (y1 - y0)))] for x, y in pts]


# ---------------------------------------------------------------- tools
def _plan(s, ctx, a):
    if not s["plan"]:
        if not a.get("understanding") or not a.get("steps"):
            return "plan needs understanding (what you see on this floor) and steps (your ordered list)", None, None
        s["plan"] = {"understanding": a["understanding"], "steps": list(a["steps"]), "done": [], "notes": []}
        _log(s, "YOU", "wrote the plan"); return f"plan stored ({len(a['steps'])} steps)", None, None
    if a.get("steps"):
        s["plan"]["steps"] = list(a["steps"])
    if a.get("understanding"):
        s["plan"]["understanding"] = a["understanding"]
    for d in a.get("done") or []:
        if d not in s["plan"]["done"]:
            s["plan"]["done"].append(int(d))
    if a.get("note"):
        s["plan"]["notes"].append(a["note"])
    return "plan updated", None, None


def _rooms(s, ctx, a):
    if a.get("replace"):
        s["rooms"] = []
    byid = {r["id"]: r for r in s["rooms"]}
    for it in a.get("items") or []:
        rid = str(it.get("id") or f"r{len(byid) + 1:02d}")
        r = byid.get(rid, {"id": rid})
        for k in ("name", "m2", "type", "flat", "label_xy"):
            if k in it:
                r[k] = it[k]
        byid[rid] = r
    s["rooms"] = list(byid.values())
    _log(s, "YOU", f"registered rooms ({len(s['rooms'])})")
    return f"{len(s['rooms'])} rooms: " + ", ".join(f"{r['id']}={r.get('name')} {r.get('m2')}" for r in s["rooms"]), None, None


def _calibrate(s, ctx, a):
    """Optional: scale from a dimension, only to express measured gaps / wall thicknesses in cm."""
    try:
        (x1, y1), (x2, y2) = _to_floor(s, [[a["x1"], a["y1"]], [a["x2"], a["y2"]]], a.get("frame"))
        dist = math.hypot(x2 - x1, y2 - y1); dim = dist / float(a["cm"])
    except Exception as e:
        return f"calibrate error: {e}", None, None
    s["dims"].append(dim)
    msg = f"scale {dim:.4f} px/cm from {a['cm']} cm = {dist:.1f} px ({x1},{y1})-({x2},{y2})"
    if len(s["dims"]) > 1:
        others = s["dims"][:-1]
        msg += f"\nyour earlier dimension scales: {', '.join(f'{d:.4f}' for d in others)}"
        if abs(dim - np.median(others)) / np.median(others) > 0.05:
            msg += " - they DISAGREE by more than 5%: one reading is wrong (chain segment? axis vs face?)."
        else:
            msg += " - consistent."
    s["ppc"] = float(np.median(s["dims"]))
    _log(s, "YOU", f"scale {s['ppc']:.4f}")
    return msg + f"\nscale in use: {s['ppc']:.4f} px/cm", None, None


def _wall(s, ctx, a):
    try:
        (x1, y1), (x2, y2) = _to_floor(s, [[a["x1"], a["y1"]], [a["x2"], a["y2"]]], a.get("frame"))
    except Exception as e:
        return f"wall error: {e}", None, None
    if abs(y2 - y1) <= 0.07 * abs(x2 - x1):
        y1 = y2 = int(round((y1 + y2) / 2))
    elif abs(x2 - x1) <= 0.07 * abs(y2 - y1):
        x1 = x2 = int(round((x1 + x2) / 2))
    wid = str(a["id"])
    s["walls"][wid] = {"line": [x1, y1, x2, y2], "note": a.get("note", "")}
    _log(s, "YOU", f"recorded wall {wid}")
    return f"wall {wid} recorded: ({x1},{y1})-({x2},{y2}) {a.get('note', '')}", None, None


def face_score(ink, horiz, c, a, b, inward, gap=3):
    """Share of the edge (inner 80 %) lying ON the room-side wall line: ink at c, free floor `gap` px further inside.
    (The DGN boundaries of the dev floors sit on the inner ink line itself, measured with GT edge profiles.)"""
    h, w = ink.shape
    lo, hi = sorted((a, b)); m = int(0.1 * (hi - lo)); lo, hi = lo + m, hi - m
    if hi - lo < 3:
        return 0.0
    t = np.arange(lo, hi + 1)
    lim = h if horiz else w
    t = t[(t >= 0) & (t < (w if horiz else h))]
    if not len(t) or not (0 <= c < lim and 0 <= c + inward * gap < lim):
        return 0.0
    get = (lambda k: ink[k, t]) if horiz else (lambda k: ink[t, k])
    on = get(c) | get(max(0, min(lim - 1, c - inward)))
    free = ~get(c + inward * gap) & ~get(max(0, min(lim - 1, c + inward * (gap + 1))))
    return float(np.mean(on & free))


def _edges(P):
    """Yield (i, A, B, kind, inward) for each edge; kind h/v/o."""
    poly = Polygon(P).buffer(0)
    n = len(P)
    for i in range(n):
        A, B = P[i], P[(i + 1) % n]
        kind = "h" if A[1] == B[1] else ("v" if A[0] == B[0] else "o")
        inward = 0
        if kind != "o":
            mx, my = (A[0] + B[0]) / 2, (A[1] + B[1]) / 2
            inward = 1 if poly.contains(Point(mx, my + 2) if kind == "h" else Point(mx + 2, my)) else -1
        yield i, A, B, kind, inward


def snap_polygon(pts, ink, r=6, min_face=0.35, margin=0.3):
    """Magnet (optional): slide each h/v edge up to r px onto the best wall line. Returns (pts, {edge: shift})."""
    P = [list(map(int, p)) for p in G.orthogonalize([list(map(int, p)) for p in pts])]
    if len(P) < 3 or r <= 0:
        return P, {}
    moves = {}
    for i, A, B, kind, inward in list(_edges(P)):
        if kind == "o":
            continue
        h = kind == "h"; c0 = A[1] if h else A[0]; a, b = (A[0], B[0]) if h else (A[1], B[1])
        best = (face_score(ink, h, c0, a, b, inward), 0)
        for d in range(1, r + 1):
            for dd in (d, -d):
                sc = face_score(ink, h, c0 + dd, a, b, inward)
                if sc > best[0] + margin:
                    best = (sc, dd)
        if best[0] >= min_face and best[1]:
            moves[i] = ("y" if h else "x", c0 + best[1], best[1])
    for i, (ax, v, _) in moves.items():
        for j in (i, (i + 1) % len(P)):
            P[j][0 if ax == "x" else 1] = int(v)
    return P, {i: m[2] for i, m in moves.items()}


def _relations(s, rid, P):
    """For each h/v edge: recorded walls it lies on, neighbour edges on the same line or one wall thickness away."""
    out = {}
    tol_same = 4
    wall_px = (s["ppc"] or 0.8) * np.array([8, 45])   # plausible wall thickness 8-45 cm
    for i, A, B, kind, inward in _edges(P):
        if kind == "o":
            continue
        h = kind == "h"; c = A[1] if h else A[0]; lo, hi = sorted((A[0], B[0]) if h else (A[1], B[1]))
        notes = []
        for wid, w in s["walls"].items():
            x1, y1, x2, y2 = w["line"]
            if h and y1 == y2 and abs(y1 - c) <= tol_same and min(x1, x2) < hi and max(x1, x2) > lo:
                notes.append(f"on wall {wid} (y={y1})")
            if not h and x1 == x2 and abs(x1 - c) <= tol_same and min(y1, y2) < hi and max(y1, y2) > lo:
                notes.append(f"on wall {wid} (x={x1})")
        for oid, Q in s["polys"].items():
            if oid == rid or not Q:
                continue
            for j, C, D, k2, inw2 in _edges(Q):
                if k2 != kind:
                    continue
                c2 = C[1] if h else C[0]; lo2, hi2 = sorted((C[0], D[0]) if h else (C[1], D[1]))
                if min(hi, hi2) - max(lo, lo2) < 0.3 * min(hi - lo, hi2 - lo2):
                    continue
                d = abs(c2 - c)
                if d <= tol_same and inw2 == inward:
                    notes.append(f"in line with {oid} edge {j} ({'y' if h else 'x'}={c2}{', off by ' + str(d) if d else ''})")
                elif inw2 == -inward and wall_px[0] <= d <= wall_px[1] and (c2 - c) * inward < 0:
                    notes.append(f"across the wall from {oid} edge {j} ({d} px = {d / (s['ppc'] or 0.8):.0f} cm wall)")
        out[i] = notes
    me = Polygon(P).buffer(0)   # real overlaps, measured on the areas
    ov = []
    for oid, Q in s["polys"].items():
        if oid != rid and Q:
            a = me.intersection(Polygon(Q).buffer(0)).area
            if a > 0.01 * min(me.area, Polygon(Q).area) and a > 20:
                ov.append(f"{oid} ({a:.0f} px2)")
    if ov:
        out["overlap"] = ov
    return out


def _store(s, rid, given, mode, frame=None, snap=0, why=""):
    P, moves = snap_polygon(given, s["_ink"], int(snap or 0))
    if len(P) < 3 or not Polygon(P).is_valid:
        P = given
    prev = s["polys"].get(rid)
    s["polys"][rid] = P
    vers = s["versions"].setdefault(rid, [])
    vers.append({"v": len(vers) + 1, "at": s["n"], "mode": mode, "given": given, "final": P, "moves": moves, "why": why})
    s["last_change"] = s["n"]
    _log(s, "YOU", f"{rid} v{len(vers)} {mode}")
    if moves:
        _log(s, "TOOL", f"magnet moved {len(moves)} edge(s) of {rid}")
    return P, moves, prev


def _draw(s, ctx, a):
    if a.get("rooms"):   # batch: many rough outlines in one call
        lines, bad = [], []
        for it in a["rooms"]:
            try:
                rid = str(it["id"]); given = _to_floor(s, it["points"], it.get("frame") or a.get("frame"))
                if len(given) < 3:
                    raise ValueError("need 3+ corners")
                _store(s, rid, given, "sketch" if phase(s) == "SKETCH" or a.get("mode") == "sketch" else "drawn")
                lines.append(f"{rid}: {len(s['polys'][rid])} corners" + ("" if _room(s, rid) else " (NOT a registered id)"))
            except Exception as e:
                bad.append(f"{it.get('id')}: error {e}")
        ov = sorted({tuple(sorted((r, o))) for r in s["polys"] for o in _overlaps(s, r)})
        txt = f"stored {len(lines)} outline(s): " + "; ".join(lines) + ("\n" + "\n".join(bad) if bad else "")
        if ov:
            txt += "\nOVERLAPS between sketches: " + ", ".join(f"{x}/{y}" for x, y in ov[:15])
        txt += "\nPicture: all your outlines on the floor (yellow = sketch, green = OK, red = problem)."
        return txt, _overview(s), None
    rid = str(a.get("id"))
    try:
        given = _to_floor(s, a["points"], a.get("frame"))
    except Exception as e:
        return f"draw error: {e}", None, None
    if len(given) < 3:
        return "draw error: need at least 3 corners", None, None
    mode = a.get("mode") or ("sketch" if phase(s) == "SKETCH" else "drawn")
    if mode not in ("sketch", "drawn", "walked"):
        mode = "drawn"
    P, moves, prev = _store(s, rid, given, mode, snap=a.get("snap"), why=a.get("why", ""))
    vers = s["versions"][rid]
    room = _room(s, rid) or {}
    rel = _relations(s, rid, P)
    L = [f"{rid} {room.get('name', '?')} v{len(vers)} ({mode}): {len(P)} corners {P}"]
    if not room:
        L.append("  NOTE: not a registered room id")
    for i, A, B, kind, inward in _edges(P):
        if kind == "o":
            L.append(f"  edge {i} {A}->{B}: oblique"); continue
        h = kind == "h"; c = A[1] if h else A[0]
        f = _edge_presence(s, A, B, kind)
        side = ("top" if inward > 0 else "bottom") if h else ("left" if inward > 0 else "right")
        t = f"  edge {i} {side} {'y' if h else 'x'}={c}: on a line {f:.0%}"
        if i in moves:
            t += f", TOOL moved it {moves[i]:+d} px"
        Lh = abs(B[0] - A[0]) if h else abs(B[1] - A[1])
        if f < 0.35 and mode != "sketch" and Lh >= 12:
            t += "  <- not on a wall line: re-read this edge"
        if rel.get(i):
            t += "; " + "; ".join(rel[i][:3])
        L.append(t)
    if rel.get("overlap"):
        L.append("  OVERLAPS: " + ", ".join(rel["overlap"]) + " - one of the two outlines crosses a wall")
    L.append(f"  status: {mode}, quality {_flag(s, rid)}")
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    m = max(30, int(0.2 * max(max(xs) - min(xs), max(ys) - min(ys))))
    vt, im, _ = _view(s, ctx, {"x0": min(xs) - m, "y0": min(ys) - m, "x1": max(xs) + m, "y1": max(ys) + m,
                               "grid": 25, "overlay": True, "focus": rid, "prev": prev, "_internal": True})
    L.append("  picture: " + vt)
    return "\n".join(L), im, None


def _view(s, ctx, a):
    img = s.get("_img")
    if img is None:
        img = s["_img"] = Image.open(ctx["image"]).convert("RGB")
    if a.get("room") and s["polys"].get(str(a["room"])):   # look around one of your rooms
        P = s["polys"][str(a["room"])]; xs, ys = [p[0] for p in P], [p[1] for p in P]
        m = max(30, int(0.25 * max(max(xs) - min(xs), max(ys) - min(ys))))
        a = dict(a, x0=min(xs) - m, y0=min(ys) - m, x1=max(xs) + m, y1=max(ys) + m, frame=None, overlay=True, focus=str(a["room"]))
    try:
        (x0, y0), (x1, y1) = _to_floor(s, [[a.get("x0", 0), a.get("y0", 0)], [a.get("x1", img.width), a.get("y1", img.height)]],
                                       a.get("frame"))
    except Exception as e:
        return f"view error: {e}", None, None
    x0, y0 = max(0, min(x0, x1)), max(0, min(y0, y1)); x1, y1 = min(img.width, max(x0 + 10, x1)), min(img.height, max(y0 + 10, y1))
    vid = f"V{len(s['views']) + 1}"
    s["views"][vid] = (x0, y0, x1, y1)
    if not a.get("_internal"):
        s.setdefault("trail", []).append(vid)
    crop = img.crop((x0, y0, x1, y1))
    sc = min(1400 / crop.width, 1400 / crop.height, 6.0)
    crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(crop, "RGBA"); f = _font(13)
    T = lambda p: ((p[0] - x0) * sc, (p[1] - y0) * sc)
    step = int(a.get("grid") or 50)
    if step > 0:
        for gx in range((x0 // step + 1) * step, x1, step):
            X = (gx - x0) * sc; d.line([(X, 0), (X, crop.height)], fill=(255, 0, 0, 70), width=1); d.text((X + 2, 2), str(gx), fill=(200, 0, 0), font=f)
        for gy in range((y0 // step + 1) * step, y1, step):
            Y = (gy - y0) * sc; d.line([(0, Y), (crop.width, Y)], fill=(255, 0, 0, 70), width=1); d.text((2, Y + 2), str(gy), fill=(200, 0, 0), font=f)
    if a.get("overlay"):
        for wid, w in s["walls"].items():
            x_1, y_1, x_2, y_2 = w["line"]
            d.line([T((x_1, y_1)), T((x_2, y_2))], fill=(0, 170, 0, 200), width=3); d.text(T((x_1, y_1)), wid, fill=(0, 120, 0), font=f)
        if a.get("prev"):
            q = [T(p) for p in a["prev"]]; d.line(q + q[:1], fill=(120, 120, 120, 200), width=2)
        for rid, pts in s["polys"].items():
            if not pts:
                continue
            q = [T(p) for p in pts]
            col = (0, 90, 255, 240) if rid == a.get("focus") else (210, 0, 210, 190)
            d.line(q + q[:1], fill=col, width=3 if rid == a.get("focus") else 2)
            for p in q:
                d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=col, width=2)
            c = Polygon(pts).representative_point(); d.text(T((c.x, c.y)), rid, fill=col[:3], font=_font(16))
        if a.get("focus") and s["versions"].get(a["focus"]):
            v = s["versions"][a["focus"]][-1]
            for i in v["moves"]:   # tool changes in orange: the edge where YOU put it
                n = len(v["final"]); A, B = v["final"][i], v["final"][(i + 1) % n]
                d.line([T(A), T(B)], fill=(255, 140, 0, 255), width=5)
    if a.get("overlay"):
        _draw_traces(s, d, T, sc, f)
    if a.get("mark_box"):
        bx = a["mark_box"]; d.rectangle([T(bx[:2]), T(bx[2:])], outline=(255, 120, 0, 255), width=3)
    s["_mini"] = _minimap(s, (x0, y0, x1, y1))
    inside = _rooms_in(s, (x0, y0, x1, y1))
    legend = " Legend: blue = room just drawn (yours), grey = its previous version, magenta = other rooms, green = recorded walls, orange = edges the TOOL moved." if a.get("overlay") else ""
    return (f"{vid}: x {x0}-{x1}, y {y0}-{y1}, grid {step} px, zoom {sc:.2f}x. To zoom further give a box in 0-1000 of "
            f"this image with frame={vid}. The second picture (minimap) shows where this is on the floor."
            + (f" Your rooms here: {', '.join(inside)}." if inside else " None of your rooms are here yet.") + legend), crop, None


def _rooms_in(s, box):
    x0, y0, x1, y1 = box
    out = []
    for rid, P in s["polys"].items():
        if P:
            c = Polygon(P).representative_point()
            if x0 <= c.x <= x1 and y0 <= c.y <= y1:
                out.append(f"{rid}({_level(s, rid)}, {_flag(s, rid)})")
    return out


def _minimap(s, box, mw=420):
    """Separate small picture of the whole floor: drawn rooms filled (green OK, yellow sketch, red flagged), the
    current view as a red box. Sent as a second image so it never hides part of the plan."""
    g = s["gray"]; H, W = g.shape
    k = mw / W; mh = int(H * k)
    mini = Image.fromarray(cv2.resize(g, (mw, mh), interpolation=cv2.INTER_AREA)).convert("RGB")
    d = ImageDraw.Draw(mini, "RGBA")
    for rid, P in s["polys"].items():
        if P:
            col = (0, 200, 0, 110) if _flag(s, rid) == "OK" else ((255, 200, 0, 110) if _level(s, rid) == "sketch" else (255, 0, 0, 110))
            d.polygon([(x * k, y * k) for x, y in P], fill=col)
    x0, y0, x1, y1 = box
    d.rectangle([x0 * k, y0 * k, max(x0 * k + 3, x1 * k), max(y0 * k + 3, y1 * k)], outline=(255, 0, 0, 255), width=3)
    for rid, t in (s.get("traces") or {}).items():
        if not t.get("closed") and t["pts"]:
            q = [(x * k, y * k) for x, y in t["pts"]]
            if len(q) > 1:
                d.line(q, fill=(0, 190, 220, 255), width=2)
            if rid == s.get("active"):
                X, Y = q[-1]; d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=(230, 0, 0, 255))
    mini.caption = (f"minimap (red dot = your pen, cyan = open trails): whole floor, red box = this view (x {x0}-{x1}, y {y0}-{y1}); "
                    "green = your OK rooms, yellow = sketches, red = flagged")
    return mini


def _history(s, ctx, a):
    rid = a.get("id")
    if not rid:
        return "change log:\n" + "\n".join(s["log"][-40:]), None, None
    vers = s["versions"].get(str(rid))
    if not vers:
        return f"{rid}: no versions yet", None, None
    L = [f"{rid} {(_room(s, rid) or {}).get('name', '')}: {len(vers)} version(s)"]
    for v in vers:
        L.append(f"  v{v['v']} at action #{v['at']} {v['mode']}: weak edges {_weak_edges(s, v['final']) or 'none'}, "
                 f"{len(v['final'])} corners {v['final']}" + (f", TOOL moved edges {list(v['moves'])}" if v["moves"] else "")
                 + (f" - why: {v['why']}" if v["why"] else ""))
    allp = [p for v in vers for p in v["final"]]
    xs, ys = [p[0] for p in allp], [p[1] for p in allp]
    m = max(30, int(0.2 * max(max(xs) - min(xs), max(ys) - min(ys))))
    x0, y0, x1, y1 = min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m
    _, im, _ = _view(s, ctx, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "grid": 25, "_internal": True})
    d = ImageDraw.Draw(im, "RGBA"); sc = im.width / (min(Image.open(ctx["image"]).width, x1) - max(0, x0))
    T = lambda p: ((p[0] - max(0, x0)) * sc, (p[1] - max(0, y0)) * sc)
    cols = [(150, 150, 150), (255, 160, 0), (0, 170, 170), (170, 0, 170), (0, 90, 255)]
    for k, v in enumerate(vers):
        col = (0, 90, 255) if k == len(vers) - 1 else cols[k % 4]
        q = [T(p) for p in v["final"]]; d.line(q + q[:1], fill=col + (230,), width=3 if k == len(vers) - 1 else 2)
        d.text((q[0][0] + 4, q[0][1] + 4 + 14 * k), f"v{v['v']}", fill=col, font=_font(15))
    L.append("  picture: every version, the latest in blue")
    return "\n".join(L), im, None


def _overview(s):
    im = Image.fromarray(s["gray"]).convert("RGB")
    sc = min(1.0, 1400 / max(im.size)); im = im.resize((int(im.width * sc), int(im.height * sc)))
    d = ImageDraw.Draw(im, "RGBA"); f = _font(13)
    for r in s["rooms"]:
        pts = s["polys"].get(r["id"])
        if pts:
            q = [(x * sc, y * sc) for x, y in pts]
            fl = _flag(s, r["id"])
            fill = (0, 200, 0, 50) if fl == "OK" else ((255, 200, 0, 60) if _level(s, r["id"]) == "sketch" else (255, 0, 0, 60))
            d.polygon(q, fill=fill, outline=(0, 0, 0))
            c = Polygon(pts).representative_point(); d.text((c.x * sc, c.y * sc), r["id"], fill=(0, 0, 0), font=f)
    return im


def _check(s, ctx, a):
    polys = {rid: p for rid, p in s["polys"].items() if p}
    out, ids = [], list(polys)
    for i in range(len(ids)):
        Pi = Polygon(polys[ids[i]])
        if not Pi.is_valid:
            out.append(f"{ids[i]}: self-crossing")
        for j in range(i + 1, len(ids)):
            Pj = Polygon(polys[ids[j]])
            inter = Pi.buffer(0).intersection(Pj.buffer(0)).area
            if inter > 0.02 * min(Pi.area, Pj.area):
                out.append(f"{ids[i]} and {ids[j]} OVERLAP ({inter:.0f} px2)")
    for r in s["rooms"]:
        rid = r["id"]
        if rid not in polys:
            out.append(f"{rid} | {r.get('name')} | NOT DRAWN" + (f" (note: {s['notes'][rid]})" if rid in s["notes"] else ""))
        else:
            wk = _weak_edges(s, polys[rid])
            out.append(f"{rid} | {r.get('name')} | {_level(s, rid)} | v{len(s['versions'].get(rid, []))} | {_flag(s, rid)}"
                       + (f" (edges off the wall lines: {wk})" if wk else ""))
    s["checked_at"] = s["n"]
    _log(s, "YOU", "check")
    return "\n".join(out) + "\nPicture: green = OK, yellow = still sketch, red = problem.", _overview(s), None


def _export(s, ctx, notes):
    notes = (notes or {}).get("notes") if isinstance(notes, dict) and "notes" in notes else notes
    s["notes"].update({str(k): v for k, v in (notes or {}).items()})
    missing = [r["id"] for r in s["rooms"] if r["id"] not in s["polys"] and r["id"] not in s["notes"]]
    if missing:
        return f"NOT YET: rooms without outline or note: {', '.join(missing)}" + board(s), None, None
    if s["checked_at"] < s["last_change"]:
        return "NOT YET: call check after your last change, look at it, then export." + board(s), None, None
    rooms = []
    for r in s["rooms"]:
        pts = s["polys"].get(r["id"])
        st = "final" if _flag(s, r["id"]) == "OK" and _level(s, r["id"]) == "walked" else "uncertain"
        rooms.append({"id": r["id"], "name": r.get("name"), "printed_area_m2": r.get("m2"), "type": r.get("type", "room"),
                      "flat": r.get("flat"), "source_labels": [f"{r.get('name')} {r.get('m2')}"], "boundary": pts,
                      "boundary_status": st, "note": s["notes"].get(r["id"], "")})
    floor = ctx["image"].stem.replace("kat_", "")
    res = {"floor": floor, "image": ctx["image"].name, "rooms": rooms}
    (ctx["out"] / "TOOL_STATE.json").write_text(json.dumps(
        {k: s[k] for k in ("plan", "rooms", "ppc", "polys", "versions", "walls", "log", "notes", "dims")},
        ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return f"exported {len(rooms)} rooms", None, res


# ---------------------------------------------------------------- wall walking (trace / walk)
HEAD = {"E": (1, 0), "W": (-1, 0), "S": (0, 1), "N": (0, -1)}


def _tr(s):
    s.setdefault("traces", {}); s.setdefault("active", None)
    return s["traces"].get(s["active"]) if s["active"] else None


def _trace(s, ctx, a):
    _tr(s)
    rid = str(a.get("id")); act = a.get("action") or ("resume" if rid in s["traces"] and a.get("x") is None else "start")
    if act == "abort":
        s["traces"].pop(rid, None); s["active"] = None if s["active"] == rid else s["active"]
        _log(s, "YOU", f"aborted trace {rid}"); return f"trace of {rid} dropped", None, None
    if act == "undo":
        t = s["traces"].get(rid)
        if not t or len(t["pts"]) <= 1:
            return f"{rid}: nothing to undo", None, None
        k = max(1, int(a.get("n") or 1)); k = min(k, len(t["pts"]) - 1)
        del t["pts"][-k:]; del t["events"][-k:]
        s["active"] = rid; _log(s, "YOU", f"undo {k} point(s) of trace {rid}")
        return _look(s, ctx, f"{rid}: removed the last {k} point(s); pen back at {t['pts'][-1]}")
    if act == "resume":
        if rid not in s["traces"]:
            return f"{rid}: no trace to resume - start one with x, y, heading", None, None
        s["active"] = rid
        if a.get("heading"):
            s["traces"][rid]["heading"] = a["heading"]
        if a.get("window"):
            s["traces"][rid]["window"] = int(a["window"])
        return _look(s, ctx, f"resumed trace of {rid}: {len(s['traces'][rid]['pts'])} points so far")
    head = a.get("heading")
    if a.get("x") is None and s["polys"].get(rid):   # start at the top-left corner of its sketch, walking clockwise
        P = s["polys"][rid]
        i = min(range(len(P)), key=lambda k: P[k][0] + P[k][1])
        nxt = P[(i + 1) % len(P)]; prv = P[i - 1]
        # screen-clockwise from the top-left corner goes east: pick the neighbour that is more to the east
        tgt = nxt if (nxt[0] - P[i][0]) >= (prv[0] - P[i][0]) else prv
        x, y = P[i]
        if not head:
            dx, dy = tgt[0] - x, tgt[1] - y
            head = ("E" if dx > 0 else "W") if abs(dx) >= abs(dy) else ("S" if dy > 0 else "N")
        start_note = f" (start = top-left corner of your sketch, ({x},{y}); check it is on the inner face and move it with undo/trace(x,y) if not)"
    else:
        start_note = ""
        try:
            (x, y), = _to_floor(s, [[a["x"], a["y"]]], a.get("frame"))
        except Exception as e:
            return f"trace error: give x, y (and heading) to start ({e})", None, None
    s["traces"][rid] = {"pts": [[x, y]], "events": ["start"], "heading": head or "E",
                        "window": int(a.get("window") or 260), "closed": False, "notes": []}
    s["active"] = rid
    _log(s, "YOU", f"started trace {rid} at ({x},{y}) heading {s['traces'][rid]['heading']}")
    return _look(s, ctx, f"trace of {rid} started at ({x},{y}), heading {s['traces'][rid]['heading']}{start_note}")


def _walk(s, ctx, a):
    t = _tr(s)
    if not t:
        return "no active trace: call trace(id, x, y, heading) first", None, None
    rid = s["active"]
    try:
        pts = _to_floor(s, a["points"], a.get("frame"))
    except Exception as e:
        return f"walk error: {e}", None, None
    ev = list(a.get("events") or [])
    for i, p in enumerate(pts):
        t["pts"].append(p); t["events"].append(ev[i] if i < len(ev) else "corner")
    if a.get("note"):
        t["notes"].append(a["note"])
    if a.get("window"):
        t["window"] = int(a["window"])
    if a.get("heading"):
        t["heading"] = a["heading"]
    elif len(t["pts"]) >= 2:
        (x0, y0), (x1, y1) = t["pts"][-2], t["pts"][-1]
        t["heading"] = ("E" if x1 > x0 else "W") if abs(x1 - x0) >= abs(y1 - y0) else ("S" if y1 > y0 else "N")
    _log(s, "YOU", f"walk {rid}: +{len(pts)} point(s) -> {t['pts'][-1]} ({t['events'][-1]})")
    P = t["pts"]
    close_tol = max(6, int(0.05 * t["window"]))
    if len(P) >= 4 and math.hypot(P[-1][0] - P[0][0], P[-1][1] - P[0][1]) <= close_tol:
        poly = P[:-1]
        t["closed"] = True; s["active"] = None
        txt, im, _ = _draw(s, ctx, {"id": rid, "points": poly, "mode": "walked", "why": "traced along the walls"})
        ims = [im] + ([s["_mini"]] if s.get("_mini") is not None else [])
        return f"ROOM CLOSED: back at the start, {rid} becomes a polygon of your {len(poly)} points.\n" + txt, ims, None
    return _look(s, ctx, f"{rid}: pen at {P[-1]} after '{t['events'][-1]}', heading {t['heading']}, {len(P)} points")


def _boxes(t):
    (cx, cy), W = t["pts"][-1], t["window"]
    hx, hy = HEAD[t["heading"]]
    back, ahead, half = 0.2 * W, 0.8 * W, 0.35 * W
    if hx:
        xs = sorted((cx - hx * back, cx + hx * ahead)); det = (xs[0], cy - half, xs[1], cy + half)
    else:
        ys = sorted((cy - hy * back, cy + hy * ahead)); det = (cx - half, ys[0], cx + half, ys[1])
    ctxb = (cx - 2 * W, cy - 2 * W, cx + 2 * W, cy + 2 * W)
    return [int(round(v)) for v in det], [int(round(v)) for v in ctxb]


def _grid_for(span):
    for g in (5, 10, 20, 25, 50, 100, 200):
        if span / g <= 12:
            return g
    return 200


def _look(s, ctx, head):
    t = _tr(s)
    det, cb = _boxes(t)
    vt, dim, _ = _view(s, ctx, {"x0": det[0], "y0": det[1], "x1": det[2], "y1": det[3], "grid": _grid_for(det[2] - det[0]),
                                "overlay": True, "_internal": True, "_detail": True})
    vid = vt.split(":")[0]
    s.setdefault("trail", []).append(vid)
    ct, cim, _ = _view(s, ctx, {"x0": cb[0], "y0": cb[1], "x1": cb[2], "y1": cb[3], "grid": _grid_for(cb[2] - cb[0]),
                                "overlay": True, "_internal": True, "mark_box": det})
    if cim.width > 800:
        cim = cim.resize((800, int(cim.height * 800 / cim.width)), Image.Resampling.LANCZOS)
    mini = s.get("_mini")
    cim.caption = f"CONTEXT (4x wider): your trail (cyan, numbered), pen (red arrow), the DETAIL box (orange rectangle)."
    L = [head, f"DETAIL = {vid} (x {det[0]}-{det[2]}, y {det[1]}-{det[3]}): give the next points in 0-1000 of {vid} with "
               f"frame={vid}, or in floor px.", _probe(s, t)]
    if t["notes"]:
        L.append("your notes on this trace: " + " | ".join(t["notes"][-4:]))
    L.append(f"trail so far: " + " -> ".join(f"{e}{tuple(p)}" for p, e in zip(t["pts"][-6:], t["events"][-6:])))
    return "\n".join(L), [dim, cim] + ([mini] if mini is not None else []), None


def _probe(s, t):
    """Measure the dark line under/near the pen along the heading: where it runs, where it breaks, where it turns."""
    ink = s["_ink"]; h, w = ink.shape
    (cx, cy), W = t["pts"][-1], t["window"]
    hx, hy = HEAD[t["heading"]]; px, py = -hy, hx   # perpendicular
    L = int(1.3 * W)

    def at(tt, o):
        x, y = int(round(cx + hx * tt + px * o)), int(round(cy + hy * tt + py * o))
        return 0 <= x < w and 0 <= y < h and bool(ink[y, x])

    best = max(range(-6, 7), key=lambda o: (sum(at(tt, o) for tt in range(0, 25)), -abs(o)))
    if sum(at(tt, best) for tt in range(0, 25)) < 10:
        return "AHEAD (measured): no dark line under the pen within 6 px - you may be off the wall face, or in an opening."
    on = [any(at(tt, best + d) for d in (-1, 0, 1)) for tt in range(L)]
    raw, st = [], 0
    for i in range(1, L + 1):
        if i == L or on[i] != on[st]:
            raw.append([on[st], st, i]); st = i
    runs = []
    for r in raw:   # breaks shorter than 4 px are scan noise: merge them into the line
        if runs and (not r[0] and r[2] - r[1] < 4 and r[2] < L) :
            runs[-1][2] = r[2]; continue
        if runs and r[0] and runs[-1][0]:
            runs[-1][2] = r[2]; continue
        runs.append(r)
    ppc = s["ppc"] or 0.8
    axis = "x" if hx else "y"; base = cx if hx else cy; sgn = hx if hx else hy

    def coord(tt):
        return int(round(base + sgn * tt))
    side_a = sum(not at(tt, best + 4) for tt in range(0, 25)); side_b = sum(not at(tt, best - 4) for tt in range(0, 25))
    names = {(1, 0): ("S", "N"), (-1, 0): ("N", "S"), (0, 1): ("W", "E"), (0, -1): ("E", "W")}[(hx, hy)]
    free = names[0] if side_a > side_b else names[1]
    lineval = int(round((cy if hx else cx) + (py if hx else px) * best))
    parts = [f"AHEAD (measured along the line at {'y' if hx else 'x'}={lineval}, free floor on its {free} side):"]
    for is_ink, a0, a1 in runs:
        if is_ink:
            parts.append(f"line {axis} {coord(a0)}..{coord(a1)}")
        else:
            gap = a1 - a0
            parts.append(f"BREAK {axis} {coord(a0)}..{coord(a1)} ({gap} px = {gap / ppc:.0f} cm" + (", door-sized" if 60 <= gap / ppc <= 130 else "") + ")")
    # does a line turn off at the end of the first run?
    if runs and runs[0][0]:
        e = runs[0][2]
        for side, nm in ((1, names[0]), (-1, names[1])):
            if sum(at(e - 2, best + side * k) for k in range(3, 18)) >= 10:
                parts.append(f"at {axis} {coord(e)} a line turns {nm}")
    return " ".join(parts[:1]) + " " + "; ".join(parts[1:8])


def _draw_traces(s, d, T, sc, f):
    """Trails of open traces (cyan, numbered points), the pen of the active trace as a red arrow."""
    for rid, t in (s.get("traces") or {}).items():
        if t.get("closed") or not t["pts"]:
            continue
        q = [T(p) for p in t["pts"]]
        if len(q) > 1:
            d.line(q, fill=(0, 190, 220, 255), width=3)
        for k, p in enumerate(q):
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(0, 190, 220, 255))
            d.text((p[0] + 5, p[1] + 3), str(k), fill=(0, 120, 150), font=f)
        if rid == s.get("active"):
            hx, hy = HEAD[t["heading"]]; X, Y = q[-1]; Lh = 28
            d.line([(X, Y), (X + hx * Lh, Y + hy * Lh)], fill=(230, 0, 0, 255), width=4)
            d.polygon([(X + hx * (Lh + 10), Y + hy * (Lh + 10)), (X + hx * Lh - hy * 7, Y + hy * Lh + hx * 7),
                       (X + hx * Lh + hy * 7, Y + hy * Lh - hx * 7)], fill=(230, 0, 0, 255))
            d.ellipse([X - 7, Y - 7, X + 7, Y + 7], outline=(230, 0, 0, 255), width=3)
