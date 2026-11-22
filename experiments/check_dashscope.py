"""Check the saved DASHSCOPE_API_KEY without sending any image: list the models the key can use (Qwen Max ones).

usage: python check_dashscope.py
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import urllib.request

import kat_agent_api as K

key = K.api_key("DASHSCOPE_API_KEY")
print("key saved:", bool(key), "| looks like sk-...:", bool(key and key.startswith("sk-")), "| ascii:", bool(key and key.isascii()))
if key:
    req = urllib.request.Request("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
                                 headers={"Authorization": "Bearer " + key})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=30))
        ids = sorted(m.get("id", "") for m in d.get("data", []))
        print(len(ids), "models;", "max / vl:", [i for i in ids if "max" in i or "vl" in i][:40])
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode(errors="replace")[:400])
