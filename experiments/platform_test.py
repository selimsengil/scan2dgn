"""Scripted dry run of the qwen_cizim platform on a dev floor (never 015), standing in for Qwen with GT rooms:
plan -> rooms -> one batch sketch (noisy) -> walk every room along its GT corners (2 corners per step, one undo)
-> check -> export. Checks gates, board, thinking budgets, pictures, export. Pictures go to kat_test/platform_test/.
usage: python platform_test.py --floor 019"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import qwen_cizim as C  # noqa: E402
import qwen_geo as G  # noqa: E402
import v0_watershed as V0  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--floor", default="019")
a = ap.parse_args()
out = config.KT / "platform_test"; out.mkdir(parents=True, exist_ok=True)
for p in out.glob("*_*.png"):
    p.unlink()
ctx = {"image": V0.B / a.floor / "clean.png", "out": out, "state": {}}
gts = [g for g in V0.load_rooms(a.floor) if g["poly"].geom_type == "Polygon"]
k = [0]; budgets = []; t_tool = [0.0]


rec = []


def call(name, args, show=False):
    rec.append({"step": len(rec), "tool_calls": [{"id": f"c{len(rec)}", "function": {"name": name, "arguments": json.dumps(args)}}]})
    budgets.append((name, C.think_budget(ctx)))
    t = time.time()
    txt, im, res = C.handle(name, args, ctx)
    t_tool[0] += time.time() - t
    k[0] += 1
    for j, one in enumerate(im if isinstance(im, list) else [im]):
        if one is not None and (show or name != "walk"):
            one.save(out / f"{k[0]:02d}_{name}{f'_{j}' if j else ''}.png")
    if show:
        print(txt)
    else:
        print(f"[{k[0]:02d}] {name}: {txt.split(chr(10))[0][:140]}")
    return txt, res


call("draw", {"id": "x", "points": [[0, 0], [10, 0], [10, 10]]})          # gate: no plan yet
call("plan", {"understanding": "test floor", "steps": ["read labels", "sketch all", "walk all", "check", "export"]})
call("rooms", {"items": [{"id": f"r{i}", "name": g.get("name"), "m2": g.get("area_m2")} for i, g in enumerate(gts)]})
call("trace", {"id": "r0"})                                                # gate: sketch first
rng = np.random.default_rng(0)
polys = []
for g in gts:
    P = G.orthogonalize([[int(round(x)), int(round(y))] for x, y in list(g["poly"].exterior.coords)[:-1]])
    polys.append(P)
x0s = [min(p[0] for p in P) for P in polys]
sk = [{"id": f"r{i}", "points": [[min(p[0] for p in P) + int(rng.integers(-8, 9)), min(p[1] for p in P) + int(rng.integers(-8, 9))],
                                 [max(p[0] for p in P), min(p[1] for p in P)], [max(p[0] for p in P), max(p[1] for p in P)],
                                 [min(p[0] for p in P), max(p[1] for p in P)]]} for i, P in enumerate(polys)]
call("draw", {"rooms": sk[:20]}, show=True)
call("draw", {"rooms": sk[20:]})
for i, P in enumerate(polys):
    show = i == 1
    h = P[1]
    head = ("E" if h[0] > P[0][0] else "W") if abs(h[0] - P[0][0]) >= abs(h[1] - P[0][1]) else ("S" if h[1] > P[0][1] else "N")
    call("trace", {"id": f"r{i}", "x": P[0][0], "y": P[0][1], "heading": head}, show=show)
    for j in range(1, len(P), 2):
        call("walk", {"points": P[j:j + 2], "events": ["corner"] * 2, "hard": (j == 1 and i == 1)}, show=show and j == 1)
        if i == 1 and j == 1:
            call("trace", {"id": "r1", "action": "undo", "n": 1})
            call("walk", {"points": [P[2]], "events": ["corner"]})
    call("walk", {"points": [P[0]], "events": ["corner"]}, show=show)
call("export", {})                                                         # gate: check first
call("check", {}, show=True)
t, res = call("export", {})
if res:
    (out / f"SONUC_{a.floor}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print("exported", len(res["rooms"]))
json.dump(rec, open(out / "transcript.json", "w", encoding="utf-8"))
json.dump({"seconds": 60, "steps": len(rec), "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "submitted": bool(res)},
          open(out / "RUN.json", "w"))
from collections import Counter
print("think budgets by tool:", dict(Counter(f"{n}:{b}" for n, b in budgets)))
print(f"platform time {t_tool[0]:.1f} s for {k[0]} calls ({t_tool[0] / k[0]:.2f} s/call)")
