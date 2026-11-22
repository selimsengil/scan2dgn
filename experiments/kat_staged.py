"""Staged floor drawing for a model that cannot plan a long tool session by itself (Qwen3.8-27B looped on
`scan` for 78 steps). The harness owns the plan, as 6.1 Sol recommended (codex_cevap_03.md, item 7):

  1. inventory  one request: floor overview + four quadrant views with a coordinate grid -> list of every
                space (name, printed m², type, label point, rough box) in floor pixels.
  2. rooms      one short conversation per space: an enlarged grid view of its box, the tools view/scan, and at
                most --room-steps tool calls before it must call `room` with the polygon.
  3. result     all spaces written to SONUC_<floor>.json (same format as the other runs) -> kat_score.py.

Same tools and teaching text as kat_agent_api.py. Backends: local llama-server (default) or the DeepSeek API.

usage: python kat_staged.py [--backend local|deepseek] [--floor 015] [--room-steps 8] [--think-inventory]
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import json
import time
from pathlib import Path

from PIL import Image

import kat_agent_api as K

ROOM_TOOLS = [t for t in K.TOOLS if t["function"]["name"] in ("view", "scan")] + [
    {"type": "function", "function": {"name": "room", "description": "Submit this space's final boundary (clockwise corners, floor pixels).",
     "parameters": {"type": "object", "properties": {"boundary": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                                     "note": {"type": "string"}}, "required": ["boundary"]}}}]

INV_Q = ("STEP 1 — INVENTORY. List EVERY space of this floor: rooms of each flat, common areas (N.O.A / stair hall), "
         "shafts and balconies. Apply the merge rules (one entry for spaces that form one room). Answer ONLY with JSON: "
         '{"spaces": [{"id": "s01", "name": "SALON", "printed_area_m2": 23.41, "type": "room|common|balcony|shaft", '
         '"flat": "3", "source_labels": ["SALON A=23.41 m2"], "label_xy": [x, y], "box": [x0, y0, x1, y1]}]}. '
         "label_xy = where its name is printed; box = a rough rectangle that contains the whole space including its walls "
         "(a little generous is fine). All in the FULL floor pixel frame; read positions from the grid labels.")

ROOM_Q = ("STEP 2 — DRAW ONE SPACE: {name} ({area} m², {typ}, flat {flat}), label at {lxy}, rough box {box}. "
          "Neighbouring spaces: {nb}. The view below shows its box enlarged with a 10-px grid in floor pixels. "
          "Trace its boundary on the inner wall faces following the teaching rules (columns, door/window closures, "
          "merges). Use view/scan only where a corner is unclear (you have at most {k} tool calls), then call `room` "
          "with the clockwise corners in FULL floor pixels.")


def inventory(fl, task, think):
    sysmsg = K.system_text(task, "single").replace("HOW TO ANSWER: you have no tools.", "HOW TO ANSWER (inventory step): no tools.")
    W, H = fl.img.size
    parts = K.first_user(fl.img)["content"]
    for name, (x0, y0, x1, y1) in {"top-left": (0, 0, W // 2 + 60, H // 2 + 60), "top-right": (W // 2 - 60, 0, W, H // 2 + 60),
                                    "bottom-left": (0, H // 2 - 60, W // 2 + 60, H), "bottom-right": (W // 2 - 60, H // 2 - 60, W, H)}.items():
        im, text = fl.view(x0, y0, x1, y1, 50)
        parts += [{"type": "text", "text": f"Quadrant {name}: {text}"}, K.img_part(im)]
    parts.append({"type": "text", "text": INV_Q})
    old = K.BACKEND["think"]; K.BACKEND["think"] = think
    r = K.call([{"role": "system", "content": sysmsg}, {"role": "user", "content": parts}], "high", None, 24000 if think else 8000)
    K.BACKEND["think"] = old
    txt = r["choices"][0]["message"].get("content") or ""
    js = K.parse_json(txt) or {}
    return js.get("spaces", []), txt


def draw_space(fl, sp, spaces, task_rules, k):
    W, H = fl.img.size
    x0, y0, x1, y1 = [int(v) for v in sp.get("box", [0, 0, W, H])]
    m = 25
    view_box = (max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))
    im, text = fl.view(*view_box, 10)
    nb = [f"{o.get('name')} box {o.get('box')}" for o in spaces if o is not sp and _near(o.get("box"), sp.get("box"))][:8]
    q = ROOM_Q.format(name=sp.get("name"), area=sp.get("printed_area_m2"), typ=sp.get("type"), flat=sp.get("flat"),
                      lxy=sp.get("label_xy"), box=sp.get("box"), nb="; ".join(nb) or "none", k=k)
    msgs = [{"role": "system", "content": task_rules},
            {"role": "user", "content": [{"type": "text", "text": text}, K.img_part(im), {"type": "text", "text": q}]}]
    calls = 0
    for step in range(k + 2):
        force = calls >= k
        r = K.call(msgs, "high", ROOM_TOOLS if not force else [t for t in ROOM_TOOLS if t["function"]["name"] == "room"], 4000)
        msg = r["choices"][0]["message"]
        a = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            a["tool_calls"] = msg["tool_calls"]
        msgs.append(a)
        if not msg.get("tool_calls"):
            js = K.parse_json(msg.get("content") or "")
            if js and js.get("boundary"):
                return js["boundary"], calls, js.get("note", "")
            msgs.append({"role": "user", "content": "Call `room` with the boundary now."})
            continue
        images = []
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            if name == "room":
                return args.get("boundary"), calls, args.get("note", "")
            calls += 1
            if name == "view":
                vim, t = fl.view(args.get("x0", 0), args.get("y0", 0), args.get("x1", 0), args.get("y1", 0), args.get("grid", 10))
                if vim is not None:
                    images.append((t, vim))
            else:
                t = fl.scan(args.get("axis", "row"), args.get("at", 0), args.get("start", 0), args.get("end", 0))
            left = k - calls
            msgs.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": t + (f" ({left} tool calls left)" if left > 0 else " (no tool calls left: call `room` now)")})
        if images:
            parts = []
            for t, vim in images[-2:]:
                parts += [{"type": "text", "text": t}, K.img_part(vim)]
            msgs.append({"role": "user", "content": parts})
    return None, calls, "no boundary returned"


def _near(a, b, pad=40):
    if not a or not b:
        return False
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["local", "deepseek"], default="local")
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    ap.add_argument("--floor", default="015")
    ap.add_argument("--room-steps", dest="room_steps", type=int, default=8)
    ap.add_argument("--think-inventory", dest="think_inv", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.backend == "local":
        K.BACKEND.update({"name": "local", "url": a.url, "think": False})
    out = K.TEST / f"{a.floor}_{a.tag or ('qwen' if a.backend == 'local' else 'deepseek')}_staged"
    out.mkdir(parents=True, exist_ok=True)
    src = K.TEST / f"{a.floor}_sol_uzman" / f"kat_{a.floor}.png"
    (out / f"kat_{a.floor}.png").write_bytes(src.read_bytes())
    fl = K.Floor(out / f"kat_{a.floor}.png")
    task = (K.TEST / f"GOREV_{a.floor}.md").read_text(encoding="utf-8")
    t0 = time.time()
    spaces, inv_text = inventory(fl, task, a.think_inv)
    (out / "inventory.json").write_text(json.dumps({"spaces": spaces, "raw": inv_text}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"inventory: {len(spaces)} spaces, {time.time() - t0:.0f} s", flush=True)
    intro, ra, rb, rc = K.teaching()
    rules = "\n\n".join([intro, "Rules:\n" + ra, "Not boundary:\n" + rb, rc])
    rooms = []
    for sp in spaces:
        t1 = time.time()
        try:
            b, calls, note = draw_space(fl, sp, spaces, rules, a.room_steps)
        except SystemExit as e:
            b, calls, note = None, 0, f"error: {e}"
        rooms.append({"id": sp.get("id"), "name": sp.get("name"), "printed_area_m2": sp.get("printed_area_m2"), "type": sp.get("type"),
                      "flat": sp.get("flat"), "source_labels": sp.get("source_labels"), "boundary": b,
                      "boundary_status": "final" if b else "uncertain", "note": note})
        print(f"  {sp.get('id')} {sp.get('name')}: {len(b) if b else 0} corners, {calls} tool calls, {time.time() - t1:.0f} s", flush=True)
    res = {"floor": a.floor, "image": f"kat_{a.floor}.png", "rooms": rooms}
    (out / f"SONUC_{a.floor}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    json.dump({"seconds": round(time.time() - t0), "spaces": len(spaces)}, open(out / "RUN.json", "w"))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
