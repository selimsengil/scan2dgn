"""Whole-sheet tag finding on Malkoc K2 without knowing the openings: PaddleOCR-VL spotting on tiles (no upscale).
usage: python tam_pafta.py TILE STEP   -> A_vl_T<TILE>.json (+ reads).  5070 Ti only."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, re, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from ortak import *
T, ST = int(sys.argv[1]), int(sys.argv[2])
MP = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\PaddleOCR-VL-1.6"
model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP)
g = sheet(); reads = []; t0 = time.time(); X0, Y0, X1, Y1 = 400, 400, 5300, 10500
for y0 in range(Y0, Y1 - 100, ST):
    for x0 in range(X0, X1 - 100, ST):
        tile = g[y0:y0 + T, x0:x0 + T]; h, w = tile.shape
        msg = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(tile).convert("RGB")}, {"type": "text", "text": "Spotting:"}]}]
        inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                       images_kwargs={"size": {"shortest_edge": 112896, "longest_edge": 2048 * 28 * 28}}).to("cuda")
        with torch.no_grad(): out = model.generate(**inp, max_new_tokens=1200, do_sample=False, repetition_penalty=1.05)
        txt = proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=False)
        for line in txt.split("\n"):
            locs = [int(v) for v in re.findall(r"<\|LOC_(\d+)\|>", line)]; t = re.sub(r"<\|[^|]*\|>", "", line).replace("</s>", "").strip()
            if len(locs) >= 8 and t: reads.append({"box": [[x0 + locs[2 * i] / 1000 * w, y0 + locs[2 * i + 1] / 1000 * h] for i in range(4)], "text": t})
    print(y0, len(reads), round(time.time() - t0), flush=True)
json.dump(reads, open(H / f"A_vl_T{T}_okumalar.json", "w", encoding="utf-8"), ensure_ascii=False)
r = score_spot(reads, f"VL spotting tum pafta T{T}"); r["sure_s"] = round(time.time() - t0); json.dump(r, open(H / f"A_vl_T{T}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: r[k] for k in ("door", "window", "okuma_sayisi", "sure_s")})
