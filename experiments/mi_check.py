"""Check deepseek_v41_mi.exe's cached teaching prefix: the same request answered with the prefix reused and
without it (V41_MI_PREFIX=0) must give the same text; and say how long each prefill took.

usage: python mi_check.py
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
import time
from pathlib import Path

import bench132 as B
import ds_agent2 as A2

OUT = Path(__file__).with_name("runs") / "mi_check"
Q = "In one sentence: what room is under the red dot, and how many corners does its boundary have? Answer as JSON {\"room\": text, \"corners\": number}."


def session(tag, extra):
    import deepseek_loop as L1
    L1.EXE = Path(A2.EXE)
    env = {"COLI_VK_SHADERS": str(Path(A2.EXE).parent / "shaders"), **A2.ENV, **extra}
    (OUT / tag).mkdir(parents=True, exist_ok=True)
    return L1.start_session(OUT / tag, 600, env)


def ask(sess, d, img_path):
    parts = A2.prefix_parts() + [("image", str(img_path)), ("text", Q)]
    t0 = time.time()
    r = sess.submit_multimodal_multi(d, A2.SYSTEM, parts, enable_thinking=False, generation_budget=40, timeout_s=3600)
    return {"raw": r["raw"], "s": round(time.time() - t0, 1), "prompt_tokens": r.get("prompt_tokens"),
            "prefill_s": r.get("prefill_only_seconds"), "reuse": r.get("mi_reuse_tokens"), "kind": r.get("mi_reuse_kind")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = []
    for key, f, name, crop, ref, label in B.rooms(1400):
        if key in ("013_04", "016_03"):
            p = OUT / f"{key}.png"
            A2.ruled(crop, (0, 0, crop.width, crop.height), 1200, dot=label).save(p)
            imgs.append(p)
    res = {}
    s = session("cached", {})
    try:
        res["A_first"] = ask(s, OUT / "cached" / "a", imgs[0])
        print("A", res["A_first"], flush=True)
        res["B_cached"] = ask(s, OUT / "cached" / "b", imgs[1])
        print("B", res["B_cached"], flush=True)
    finally:
        s.close()
    s = session("plain", {"V41_MI_PREFIX": "0"})
    try:
        res["B_plain"] = ask(s, OUT / "plain" / "b", imgs[1])
        print("B plain", res["B_plain"], flush=True)
    finally:
        s.close()
    res["same_text"] = res["B_cached"]["raw"] == res["B_plain"]["raw"]
    print("SAME:", res["same_text"])
    json.dump(res, open(OUT / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
