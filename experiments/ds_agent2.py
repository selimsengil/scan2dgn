"""ds_agent v2: the same loop as ds_agent.py, taught with the cached teaching prefix.

Every request = [teaching prefix: intro, SHEET_A, text, SHEET_B, text, SHEET_C, text + method]
              + [this step's view image + question].
The prefix never changes, so the engine (deepseek_v41_mi.exe: several images a request, prefix reuse with
images) reads it once and reuses it; each step costs only its own view and question.
Views follow 6.1 Sol's practice: clean image, rulers on the top and left edge only (no grid over the plan),
small regions enlarged with nearest-neighbour so pixels stay sharp; every corner named by the two faces
that meet there.

usage: python ds_agent2.py --rooms 015_02,019_18,001_17 --tag ds_agent_b
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

from PIL import Image, ImageDraw

import bench132 as B
import ds_agent as A1
import qwen_test as QT
import snap

TEACH = Path(__file__).with_name("teach")
EXE = r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io\deepseek_v41_mi.exe"
# 3.9 tok/s measured with a 4 GB reserve (5 GB left fewer experts on the card: 3.1 tok/s); 1536 rows a prefill step
# (the cached teaching prefix makes later prompts short; the first one is read in a few steps once)
ENV = {"COLI_VK_CHAIN_ROWS": "1536", "COLI_VK_TIER_RESERVE_GB": "4.0", "COLI_VK_TIER_STREAM_ROWS": "1",
       "V41_STATS": "1", "COLI_KV_PREFIX": "1", "CTX": "8192"}   # teaching ~3.5k + view ~1k + thinking + answer
THINK = {"overview": 700, "montage": 600, "check": 500}   # thinking budget (tokens) before the answer is forced
ANSWER_TOKENS = 260

SYSTEM = ("You are a careful architectural draughtsman. You trace room boundaries on scanned Turkish floor plans exactly as "
          "this office's CAD operators do. Think as much as a decision needs and no more; end every answer with exactly "
          "one JSON object.")


def sections():
    t = (TEACH / "TEACH.md").read_text(encoding="utf-8")
    parts = dict(re.findall(r"## ([^\n]+)\n(.*?)(?=\n## |\Z)", t, re.S))
    get = lambda k: next(v.strip() for kk, v in parts.items() if kk.startswith(k))
    return get("INTRO"), get("AFTER SHEET A"), get("AFTER SHEET B"), get("AFTER SHEET C"), get("HOW TO WORK")


def prefix_parts():
    intro, a, b, c, how = sections()
    return [("text", intro), ("text", "SHEET A:"), ("image", str(TEACH / "SHEET_A.png")), ("text", a),
            ("text", "SHEET B:"), ("image", str(TEACH / "SHEET_B.png")), ("text", b),
            ("text", "SHEET C:"), ("image", str(TEACH / "SHEET_C.png")), ("text", c), ("text", how),
            ("text", "--- END OF TEACHING. The task follows. ---")]


def ruled(img, box, max_side, poly=None, ring=None, dot=None, numbered=False):
    """Sol's view: rulers only (no grid), nearest-neighbour when enlarged >= 2x."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    reg = img.crop((x0, y0, x1, y1)).convert("RGB")
    s = min(max_side / max(reg.size), 6.0)
    rs = Image.Resampling.NEAREST if s >= 2 else Image.Resampling.LANCZOS
    reg = reg.resize((max(1, int(reg.width * s)), max(1, int(reg.height * s))), rs)
    L, T = 62, 34
    out = Image.new("RGB", (reg.width + L + 8, reg.height + T + 8), (255, 255, 255))
    out.paste(reg, (L, T))
    d = ImageDraw.Draw(out, "RGBA")
    fx = lambda x: L + (x - x0) * s
    fy = lambda y: T + (y - y0) * s
    major = A1.nice_step(x1 - x0)
    minor = major / 5 if major >= 5 else major
    for lo, hi, f, horiz in ((x0, x1, fx, True), (y0, y1, fy, False)):
        v = (lo // minor) * minor
        while v <= hi:
            if v >= lo:
                P, big = f(v), v % major == 0
                if horiz:
                    d.line((P, T - (10 if big else 4), P, T), fill=(0, 0, 0))
                    if big: d.text((P - 12, 2), str(int(v)), fill=(0, 0, 0), font=A1.FONT)
                else:
                    d.line((L - (10 if big else 4), P, L, P), fill=(0, 0, 0))
                    if big: d.text((2, P - 8), str(int(v)), fill=(0, 0, 0), font=A1.FONT)
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
                d.text((X + 6, Y + 4), str(k + 1), fill=(160, 0, 160), font=A1.FONT_B)
    if ring:
        X, Y = fx(ring[0]), fy(ring[1])
        d.ellipse((X - 14, Y - 14, X + 14, Y + 14), outline=(0, 200, 230), width=3)
    return out


FORMAT_POINTS = ('Answer with one JSON object with three keys: "points" = the list of all corners as [x, y] pairs in frame '
                 'pixels, clockwise, starting anywhere; "faces" = a list with one short string per corner naming the two '
                 'faces that meet there; "note" = one sentence.')
OVERVIEW_Q = (
    "TASK: trace the room '{target}'. The RED dot sits on its printed label; the room is the whole space enclosed by the "
    "walls around that dot (rooms are much larger than their labels). Neighbouring rooms are visible too. {history}"
    "If you need a closer look first, answer only {{\"zoom\": [x0, y0, x1, y1]}} with a frame-pixel box ({zooms_left} zooms "
    "left). Otherwise trace the inner boundary. " + FORMAT_POINTS)
ZOOM_Q = ("ZOOM of the frame box {box} (same frame, finer ruler). {history}Answer another {{\"zoom\": [x0, y0, x1, y1]}} "
          "({zooms_left} left) or trace the room now. " + FORMAT_POINTS)
MONTAGE_Q = (
    "CORNER CHECK for '{target}'. Each numbered panel is a close, sharp view around one corner of your boundary (rulers in "
    "the same frame; MAGENTA = your boundary, CYAN ring = that corner). Faces you named: {faces}. For every panel decide: "
    "the ring sits where the two inner faces meet (keep), or it must move, or it is not a real corner (delete), or a corner "
    "is missing right after it (insert, e.g. a column shoulder). Answer one JSON object with the key \"corners\": a list with "
    "one entry per panel, each an object with \"i\" (panel number) and exactly one of \"keep\": true, \"move\": [x, y], "
    "\"delete\": true, or \"move\" plus \"insert_after\" (a list of [x, y]).")
CHECK_Q = ("FINAL CHECK of '{target}': the MAGENTA polygon with numbered corners on the whole view. Inner faces, straight "
           "closures at openings, every column shoulder, nothing outside the room, no part of the room left out? Answer "
           "{{\"accept\": true}} if it is right; otherwise give the corrected boundary. " + FORMAT_POINTS)


def montage(crop, poly, win, idx):
    """Numbered panels (one per corner index in idx), 4 a row."""
    from PIL import ImageFont
    try:
        big = ImageFont.truetype("arialbd.ttf", 30)
    except OSError:
        big = A1.FONT_B
    tiles = []
    for i in idx:
        x, y = poly[i]
        t = ruled(crop, (x - win / 2, y - win / 2, x + win / 2, y + win / 2), 300, poly=poly, ring=(x, y))
        d = ImageDraw.Draw(t)
        d.rectangle((t.width - 44, t.height - 40, t.width - 2, t.height - 2), fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((t.width - 38, t.height - 38), str(i + 1), fill=(0, 0, 0), font=big)
        tiles.append(t)
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    cols = min(4, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    M = Image.new("RGB", (cols * w + (cols - 1) * 8, rows * h + (rows - 1) * 8), (255, 255, 255))
    for k, t in enumerate(tiles):
        M.paste(t, ((k % cols) * (w + 8), (k // cols) * (h + 8)))
    return M


class Agent2:
    def __init__(self, out: Path):
        import deepseek_loop as L1
        L1.EXE = Path(EXE)
        self.S = L1.S
        self.sess = L1.start_session(out, 1500, {"COLI_VK_SHADERS": str(Path(EXE).parent / "shaders"), **ENV})
        self.prefix = prefix_parts()
        self.n, self.out, self.log = 0, out, []

    def ask(self, img, text, effort, budget, tag):
        """Thinking up to `budget` tokens; if the answer is not out by then, the thinking is closed where it
        stopped ("</think>") and the answer is generated from there (the engine reuses the held state)."""
        self.n += 1
        p = self.out / f"step_{self.n:03d}_{tag}.png"; img.save(p)
        t0 = time.time()
        parts = self.prefix + [("image", str(p)), ("text", text)]
        r = self.sess.submit_multimodal_multi(self.out / f"req_{self.n:03d}", SYSTEM, parts, enable_thinking=True,
                                              reasoning_effort=effort, generation_budget=budget + ANSWER_TOKENS,
                                              timeout_s=7200, stop_text=None)
        raw, forced = r["raw"], False
        js = A1.last_json(raw) if "</think>" in raw else {}
        if not js:   # cut inside the thinking: close it and let the answer out
            forced = True
            thought = raw.split("</think>")[0]
            r2 = self.sess.submit_multimodal_multi(self.out / f"req_{self.n:03d}f", SYSTEM, parts, enable_thinking=True,
                                                   reasoning_effort=effort, generation_budget=ANSWER_TOKENS, timeout_s=3600,
                                                   assistant_prefix=thought + "\n\nTime is up; answering now with the best "
                                                                              "estimate.\n</think>\n\n")
            js = A1.last_json(r2["raw"])
            r = {**r, "forced_answer": r2["raw"][-400:], "forced_prefill_s": r2.get("prefill_only_seconds")}
        e = {"step": self.n, "tag": tag, "seconds": round(time.time() - t0, 1), "answer": js, "forced": forced,
             "tokens": r.get("generated_tokens"), "prompt_tokens": r.get("prompt_tokens"),
             "prefill_s": round(r.get("prefill_only_seconds") or 0, 1), "decode_tps": round(r.get("decode_tps") or 0, 2),
             "reuse": r.get("mi_reuse_tokens"), "thinking_tail": raw.split("</think>")[0][-600:]}
        self.log.append(e)
        print(json.dumps({k: e[k] for k in ("step", "tag", "seconds", "tokens", "prompt_tokens", "prefill_s", "reuse", "forced")}),
              "->", json.dumps(js, ensure_ascii=False)[:200], flush=True)
        return js

    def close(self):
        self.sess.close()


def draw_room(ag, key, name, crop, ref, label, max_zoom=3):
    W, H = crop.size
    full = (0, 0, W, H)
    history, zooms = "", 0
    js = ag.ask(ruled(crop, full, 1200, dot=label), OVERVIEW_Q.format(target=name, history=history, zooms_left=max_zoom),
                "high", THINK["overview"], key + "_overview")
    while "zoom" in js and zooms < max_zoom:
        zooms += 1
        box = [float(v) for v in js["zoom"]][:4]
        history += f"(Already seen zoom {[int(v) for v in box]}.) "
        js = ag.ask(ruled(crop, box, 1000, dot=label), ZOOM_Q.format(box=[int(v) for v in box], history=history,
                                                                     zooms_left=max_zoom - zooms), "high", THINK["overview"], key + f"_zoom{zooms}")
    first = A1.pts_ok(js.get("points") or [])
    if not first:
        js = ag.ask(ruled(crop, full, 1200, dot=label), OVERVIEW_Q.format(target=name, history=history + "Trace it now. ",
                                                                           zooms_left=0), "high", THINK["overview"], key + "_overview2")
        first = A1.pts_ok(js.get("points") or [])
    if not first:
        return {"error": "no polygon"}
    faces = list(js.get("faces") or [])
    poly = [list(p) for p in first]
    win = max(90.0, 0.18 * min(W, H))
    # corner check: all corners at once, 8 panels per image
    for start in range(0, min(len(poly), 24), 8):
        idx = list(range(start, min(start + 8, len(poly))))
        fl = "; ".join(f"{i + 1}: {faces[i] if i < len(faces) else 'not named'}" for i in idx)
        a = ag.ask(montage(crop, poly, win, idx), MONTAGE_Q.format(target=name, faces=fl), "low", THINK["montage"], key + f"_corners{start // 8 + 1}")
        ins = {}
        for c in a.get("corners") or []:
            try:
                i = int(c.get("i")) - 1
            except Exception:
                continue
            if i not in idx:
                continue
            if isinstance(c.get("move"), list) and len(c["move"]) == 2:
                poly[i] = [float(c["move"][0]), float(c["move"][1])]
            if c.get("delete"):
                poly[i] = None
            if isinstance(c.get("insert_after"), list):
                try:
                    ins[i] = [[float(u), float(v)] for u, v in c["insert_after"]][:3]
                except Exception:
                    pass
        new = []
        for i, p in enumerate(poly):
            if p is not None:
                new.append(p)
            new.extend(ins.get(i, []))
        poly = new if len(new) >= 3 else [p for p in poly if p is not None]
    refined = [p[:] for p in poly]
    c = ag.ask(ruled(crop, full, 1200, poly=poly, numbered=True, dot=label), CHECK_Q.format(target=name), "high", THINK["check"], key + "_check")
    final = (A1.pts_ok(c.get("points") or []) if not c.get("accept") else None) or refined
    sn = snap.snap_polygon(crop, final) if len(final) >= 3 else []
    sc = lambda p: QT.brief(QT.score(p, ref, type("H", (), {"inside": None})())) if len(p) >= 3 else {"valid": False}
    return {"first": first, "refined": refined, "final": final, "snapped": sn,
            "score": {"first": sc(first), "refined": sc(refined), "final": sc(final), "snapped": sc(sn)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="015_02,019_18,001_17")
    ap.add_argument("--tag", default="ds_agent_b")
    a = ap.parse_args()
    out = Path(__file__).with_name("runs") / a.tag; out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in B.rooms(1400) if r[0] in a.rooms.split(",")]
    ag = Agent2(out)
    res = {}
    try:
        for key, f, name, crop, ref, label in todo:
            t1 = time.time()
            r = draw_room(ag, key, name, crop, ref, label)
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
