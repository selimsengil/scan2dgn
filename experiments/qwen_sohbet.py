"""Multi-turn chat with the local Qwen3.8-27B (llama-server) for the tool co-design workshop.

usage: python qwen_sohbet.py --dir WORKDIR --msg MESSAGE_FILE [--img PATH ...] [--new]
- History in WORKDIR/sohbet_gecmis.json; readable transcript appended to WORKDIR/SOHBET.md.
- Images are attached to this user message (downscaled to 1400 px).
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import json
import time
from pathlib import Path

from PIL import Image

import kat_agent_api as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--msg", required=True)
    ap.add_argument("--img", nargs="*", default=[])
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=24000)
    a = ap.parse_args()
    d = Path(a.dir)
    hist_f = d / "sohbet_gecmis.json"
    hist = [] if a.new or not hist_f.exists() else json.load(open(hist_f, encoding="utf-8"))
    text = Path(a.msg).read_text(encoding="utf-8")
    parts = []
    for p in a.img:
        im = Image.open(p).convert("RGB"); im.thumbnail((1400, 1400))
        parts += [{"type": "text", "text": f"[görüntü: {Path(p).name}]"}, K.img_part(im)]
    parts.append({"type": "text", "text": text})
    hist.append({"role": "user", "content": parts})
    md = open(d / "SOHBET.md", "a", encoding="utf-8")
    turn = sum(1 for m in hist if m["role"] == "user")
    md.write(f"\n\n---\n## Tur {turn}\n\n### 🟣 Claude\n\n" + (f"*(ekli görüntüler: {', '.join(Path(p).name for p in a.img)})*\n\n" if a.img else "") + text.strip() + "\n\n### 🟠 Qwen\n\n")
    md.flush()
    K.BACKEND.update({"name": "local", "url": "http://127.0.0.1:8091/v1/chat/completions", "think": True})
    t0 = time.time()
    r = K.call(hist, "high", None, a.max_tokens)
    m = r["choices"][0]["message"]
    reasoning = m.get("reasoning_content") or ""
    if reasoning:
        md.write("> 💭 " + reasoning[-1500:].replace("\n", " ") + "\n\n")
    md.write((m.get("content") or "") + f"\n\n*({time.time() - t0:.0f} s, {r.get('usage')})*\n")
    md.close()
    # keep only text of old images in history to save context: replace image parts of earlier turns
    hist.append({"role": "assistant", "content": m.get("content") or ""})
    json.dump(hist, open(hist_f, "w", encoding="utf-8"), ensure_ascii=False)
    print(m.get("content") or "")
    print(f"\n[{time.time() - t0:.0f} s] usage={r.get('usage')} finish={r['choices'][0].get('finish_reason')}")


if __name__ == "__main__":
    main()
