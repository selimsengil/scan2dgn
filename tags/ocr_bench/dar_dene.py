"""Tight level crops of openings (box + margin), OCR mode, PaddleOCR-VL vs GLM-OCR, a few floors.  5070 Ti.
usage: python dar_dene.py FLOOR N margin_m scale"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, sys, time, re
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import cv2, numpy as np, torch
from PIL import Image
from shapely.geometry import shape
from transformers import AutoProcessor, AutoModelForImageTextToText
K = str(config.KT); D = str(config.MODELS)
fid, N, mg, sc = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
d = json.load(open(K + rf"\aciklik_ag\veri\{fid}.json", encoding="utf-8")); s = d["px_per_m"]
g = cv2.imdecode(np.fromfile(K + rf"\dgn_katlar\{fid}\clean.png", np.uint8), 0)
def level(o):
    c = np.array(shape(o["poly"]).minimum_rotated_rectangle.exterior.coords)[:4]; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e))
    dd = c[(k + 1) % 4] - c[k]; ang = np.degrees(np.arctan2(dd[1], dd[0])); m = c.mean(0); L = (max(e) / 2 + mg * s) * sc; Hh = (min(e) / 2 + mg * s) * sc
    M = cv2.getRotationMatrix2D(tuple(m), ang, sc); M[:, 2] += [L - m[0], Hh - m[1]]; v = cv2.warpAffine(g, M, (int(2 * L), int(2 * Hh)), borderValue=255, flags=cv2.INTER_CUBIC)
    return [v, cv2.rotate(v, cv2.ROTATE_180)], max(e) / s * 100
ops = d["openings"][:N]
for name, MP, prompt in (("VL", D + r"\PaddleOCR-VL-1.6", "OCR:"), ("GLM", D + r"\GLM-OCR", "Text Recognition:")):
    model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP); t0 = time.time(); outs = []
    for o in ops:
        ims, w = level(o); r = []
        for im in ims:
            msg = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(im).convert("RGB")}, {"type": "text", "text": prompt}]}]
            inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda"); inp.pop("token_type_ids", None)
            with torch.no_grad(): out = model.generate(**inp, max_new_tokens=48, do_sample=False, repetition_penalty=1.05)
            r.append(proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).replace("\n", " ")[:40])
        outs.append((o["kind"][0], round(w), r))
    print(name, round(time.time() - t0, 1), "s", flush=True)
    for x in outs: print("  ", x)
    del model; torch.cuda.empty_cache()
