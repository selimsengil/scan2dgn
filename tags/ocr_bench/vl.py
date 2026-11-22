"""PaddleOCR-VL-1.6 (transformers, bf16, GPU).  usage: python vl.py dene|B|A"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys, time
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from ortak import *
MP = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\PaddleOCR-VL-1.6"
model = AutoModelForImageTextToText.from_pretrained(MP, torch_dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP)
def run(img, task):
    im = Image.fromarray(img).convert("RGB")
    if task == "spotting" and max(im.size) < 1500: im = im.resize((im.size[0] * 2, im.size[1] * 2), Image.LANCZOS)
    mx = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
    msg = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": {"ocr": "OCR:", "spotting": "Spotting:"}[task]}]}]
    inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                   images_kwargs={"size": {"shortest_edge": 112896, "longest_edge": mx}}).to(model.device)
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=1024, do_sample=False)
    return proc.decode(out[0][inp["input_ids"].shape[-1]:-1]), im.size
mode = sys.argv[1]
if mode == "dene":
    g = sheet(); t = g[1500:2700, 1500:2700]
    for task in ("spotting", "ocr"):
        t0 = time.time(); txt, sz = run(t, task); print(task, sz, round(time.time() - t0, 1), "s\n", txt[:1500])
if mode == "B":
    res = {}; t0 = time.time()
    for oid, kind, tag, ims in crops():
        res[oid] = []
        for im in ims:
            if max(im.shape) < 700: im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            res[oid].append(run(im, "ocr")[0])
    r = score_crops(res, "PaddleOCR-VL-1.6 kirpim", {"sure_s": round(time.time() - t0, 1)}); json.dump(r, open(H / "B_vl.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: r[k] for k in ("door", "window", "sure_s")})
if mode == "A":
    import re as _re
    g = sheet(); T, ST = 1200, 1000; reads = []; t0 = time.time(); X0, Y0, X1, Y1 = 400, 400, 5300, 10500
    for y0 in range(Y0, Y1 - 200, ST):
        for x0 in range(X0, X1 - 200, ST):
            tile = g[y0:y0 + T, x0:x0 + T]; txt, sz = run(tile, "spotting"); h, w = tile.shape
            for line in txt.split("\n"):
                locs = [int(v) for v in _re.findall(r"<\|LOC_(\d+)\|>", line)]; text = _re.sub(r"<\|LOC_\d+\|>", "", line).strip()
                if len(locs) >= 8 and text:
                    pts = [[x0 + locs[2 * i] / 1000 * w, y0 + locs[2 * i + 1] / 1000 * h] for i in range(4)]; reads.append({"box": pts, "text": text})
            print(x0, y0, len(reads), round(time.time() - t0), flush=True)
    json.dump(reads, open(H / "A_vl_okumalar.json", "w", encoding="utf-8"), ensure_ascii=False)
    r = score_spot(reads, "PaddleOCR-VL-1.6 spotting"); r["sure_s"] = round(time.time() - t0); json.dump(r, open(H / "A_vl.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: r[k] for k in ("door", "window", "okuma_sayisi", "sure_s")})
