"""Minimal client for the local Qwen3.8-27B llama-server (127.0.0.1:8091).

ask(text, images=[np.ndarray BGR], think=False) -> (answer, timings)
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import base64
import json
import sys
import time
import urllib.request

import cv2

URL = "http://127.0.0.1:8091/v1/chat/completions"


def ask(text: str, images=(), think: bool = False, max_tokens: int = 800):
    content = []
    for im in images:
        ok, buf = cv2.imencode(".png", im)
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf).decode()}})
    content.append({"type": "text", "text": text})
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    t = r.get("timings", {})
    t["wall_s"] = round(time.time() - t0, 1)
    return r["choices"][0]["message"]["content"], t


if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\Selim\Documents\cizim_projeler\egitim_verisi\sam_etiket_hassas_v3_20261022\floors\015\clean.png")[100:760, 130:740]
    q = sys.argv[1] if len(sys.argv) > 1 else "Bu kat planı kesitinde hangi odalar var, yazılı alanlarıyla listele. Her odanın kapısı hangi duvarda? Kısa cevap ver."
    a, t = ask(q, [img])
    print(t.get("wall_s"), "s | prompt", t.get("prompt_n"), "tok | gen", t.get("predicted_n"), "tok at",
          round(t.get("predicted_per_second", 0), 1), "tok/s")
    print(a)
