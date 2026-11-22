"""Loop v2: one opening at a time, a yes/no question, and an area guard.

Modes
  greedy : no model. Each candidate opening is toggled; kept if the printed-area error of
           the rooms around it drops.
  model  : DeepSeek sees a close-up of ONE candidate closing line (red) and answers whether
           it closes a door / window / wall gap (true) or crosses a room (false). The answer
           sets the opening; the area guard reverts it if the rooms around get worse.
Only openings next to rooms whose area disagrees with the printed one are considered.

usage: python deepseek_loop2.py --floor 015 --mode greedy --out runs/015_greedy
       python deepseek_loop2.py --floor 015 --mode model --thinking --budget 2500 --out runs/015_m_think
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import araclar as A
import deepseek_loop as L1

SYSTEM = L1.SYSTEM
Q = """Görüntü bir kat planından yakın bir kesit. KIRMIZI kalın çizgi, bir aracın "burada iki alanı ayıran bir açıklık var,
buradan kapatayım" dediği yer.

Doğru bir kapatma çizgisi: bir kapı, pencere ya da duvardaki geçit boşluğunu, iki yanındaki duvar uçlarını birleştirerek
duvar hizasında kapatır; çizginin iki tarafında farklı alanlar (iki oda, oda ve koridor, oda ve balkon) vardır.
Yanlış bir kapatma çizgisi: bir odanın, koridorun ya da balkonun içinden geçer; mobilya, yatak, tezgah, merdiven ya da
aks çizgisi ile duvar arasında kalır; iki tarafı aynı alandır.

Kırmızı çizgi doğru bir kapatma çizgisi mi? Yalnız şu JSON ile cevap ver:
{{"kapat": true veya false, "neden": "kısa açıklama"}}"""


def opening_view(k: A.Kat, o: A.Opening, path: Path, win_m: float = 3.5, max_px: int = 900):
    r = int(win_m / 2 * k.s)
    x0, y0 = max(0, o.p[0] - r), max(0, o.p[1] - r)
    x1, y1 = min(k.W, o.p[0] + r), min(k.H, o.p[1] + r)
    vis = cv2.cvtColor(k.img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    f = min(2.0, max_px / max(vis.shape[:2]))
    vis = cv2.resize(vis, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
    a = (int((o.a[0] - x0) * f), int((o.a[1] - y0) * f)); b = (int((o.b[0] - x0) * f), int((o.b[1] - y0) * f))
    cv2.line(vis, a, b, (0, 0, 255), max(3, int(0.05 * k.s * f)))
    cv2.imwrite(str(path), vis)


def around(k: A.Kat, o: A.Opening) -> set[int]:
    """Seeds whose region touches the opening (within 0.5 m of its line)."""
    r = int(0.5 * k.s)
    x0, y0 = max(0, min(o.a[0], o.b[0]) - r), max(0, min(o.a[1], o.b[1]) - r)
    x1, y1 = min(k.W, max(o.a[0], o.b[0]) + r), min(k.H, max(o.a[1], o.b[1]) + r)
    vals = set(np.unique(k.lab[y0:y1, x0:x1]).tolist()) - {0, 1}
    return {i for i in range(len(k.labels)) if k.root[i] + 2 in vals}


def err(k: A.Kat, rooms: list[dict], seeds: set[int]) -> float:
    e, seen = 0.0, set()
    for i in seeds:
        r = rooms[i]
        if k.root[i] in seen or not r["area_text"] or not (0.8 < r["area_text"] < 200):
            continue
        seen.add(k.root[i])
        e += min(2.0, abs(r["area_m2"] / r["area_text"] - 1))
    return e


def relevant(k: A.Kat, rooms: list[dict], tol: float) -> list[A.Opening]:
    bad = {i for i, r in enumerate(rooms) if not (r["area_text"] and abs(r["area_m2"] / r["area_text"] - 1) < tol)}
    return [o for o in k.openings if around(k, o) & bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="015")
    ap.add_argument("--mode", choices=["greedy", "model"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--budget", type=int, default=2500)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--no_guard", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    k = A.Kat(a.floor)
    rooms = k.segment()
    before = k.score()
    print("tools only:", {x: before[x] for x in ("ge95", "ge90", "ge80", "mean")}, flush=True)
    cand = relevant(k, rooms, a.tol)
    print("openings to decide:", [o.id for o in cand], flush=True)
    log = {"args": vars(a), "before": before, "steps": []}
    sess = L1.start_session(out, a.budget) if a.mode == "model" else None
    try:
        for n, o in enumerate(cand, 1):
            seeds = around(k, o)
            e0 = err(k, rooms, seeds)
            was = o.id in k.closed
            t0 = time.time()
            if a.mode == "greedy":
                want, ans = (not was), None
            else:
                img = out / f"op_{o.id:02d}.png"
                opening_view(k, o, img)
                prepared = L1.S.prepare_multimodal_payload(img, SYSTEM, Q, enable_thinking=a.thinking, reasoning_effort=a.effort)
                res = sess.submit_multimodal(out / f"req_{n:03d}", prepared, generation_budget=a.budget, timeout_s=7200)
                ans = L1.parse(res["raw"].split("</think>")[-1])
                if "kapat" not in ans or not isinstance(ans.get("kapat"), (bool, list)):
                    ans = L1.parse(res["raw"])
                want = bool(ans.get("kapat")) if isinstance(ans.get("kapat"), bool) else was
            if want != was:
                (k.closed.add if want else k.closed.discard)(o.id)
                new = k.segment()
                e1 = err(k, new, seeds | around(k, o))
                e0b = e0
                keep = a.no_guard or e1 <= e0b - (0.005 if a.mode == "greedy" else -0.005)
                if not keep:
                    (k.closed.add if was else k.closed.discard)(o.id)
                    new = k.segment()
                rooms = new
            else:
                e1, keep = e0, True
            step = {"opening": o.id, "width_m": o.width_m, "was_closed": was, "model_says_close": want if a.mode == "model" else None,
                    "final_closed": o.id in k.closed, "err_before": round(e0, 3), "err_after": round(e1, 3), "kept": keep,
                    "answer": ans, "seconds": round(time.time() - t0, 1)}
            log["steps"].append(step)
            print(json.dumps(step, ensure_ascii=False), flush=True)
            json.dump(log, open(out / "LOG.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    finally:
        if sess:
            sess.close()
    k.segment()
    after = k.score()
    log["after"] = after
    print("tools only:", {x: before[x] for x in ("ge95", "ge90", "ge80", "mean")})
    print("after     :", {x: after[x] for x in ("ge95", "ge90", "ge80", "mean")})
    k.render(out / "final.png", gt=True)
    json.dump(log, open(out / "LOG.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
