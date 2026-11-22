"""Realistic speed test: thinking on, a floor-plan reasoning prompt (draft acceptance is not inflated by repetitive text).
usage: python qwen_hiz2.py [--n 600]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8091)
ap.add_argument("--n", type=int, default=600)
a = ap.parse_args()
q = ("A room cell computed by a watershed tool has area 22.78 m2 but the printed label says ODA A:13.14 m2 (ratio 1.733). "
     "Its neighbour ANTRE 7.54 m2 came out at 4.78 m2. The room is at x 1150-1450, y 1250-1500 in a 2523x2071 floor plan, "
     "the door K3-90/210 is on its west wall near y 1330. Think step by step about what probably went wrong and which "
     "barrier line (x1,y1,x2,y2) you would add, then answer with the tool call you would make.")
body = {"messages": [{"role": "user", "content": q}], "max_tokens": a.n, "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": True}}
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                                            json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
t = r.get("timings", {})
print(f"thinking gen {t.get('predicted_n')} tok @ {t.get('predicted_per_second', 0):.1f} tok/s"
      + (f" | draft accepted {t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get('draft_n') else ""))
