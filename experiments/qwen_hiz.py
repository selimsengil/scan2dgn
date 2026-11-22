"""Quick speed test of the local llama-server: prompt processing and generation tok/s (from llama-server timings).
usage: python qwen_hiz.py [--port 8091] [--n 300]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8091)
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()
text = ("Kat planlarında oda sınırı bitmiş iç duvar yüzünden çizilir. " * 120)
body = {"messages": [{"role": "user", "content": text + "\nBunu bir paragrafta özetle ve sonra 1'den 200'e kadar say."}],
        "max_tokens": a.n, "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                                            json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
t = r.get("timings", {})
print(f"prompt {t.get('prompt_n')} tok @ {t.get('prompt_per_second', 0):.0f} tok/s | gen {t.get('predicted_n')} tok @ {t.get('predicted_per_second', 0):.1f} tok/s")
