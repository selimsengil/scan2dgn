"""B (reading level crops) with GLM-OCR or MinerU2.5, transformers bf16 GPU, prompt 'Text Recognition:'.  usage: python b_vlm.py glm|mineru [dene]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys, time
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from ortak import *
D = str(config.MODELS); which = sys.argv[1]
MP = {"glm": D + r"\GLM-OCR", "mineru": D + r"\MinerU2.5-2509-1.2B"}[which]
model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP)
PROMPT = "\nText Recognition:" if which == "mineru" else "Text Recognition:"
def run(img):
    im = Image.fromarray(img).convert("RGB")
    msg = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
    inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    inp.pop("token_type_ids", None)
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=128, do_sample=False, repetition_penalty=1.05 if which == "mineru" else 1.0)
    return proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
if len(sys.argv) > 2:
    c = crops(); print(c[0][2], repr(run(c[0][3][0]))); print(c[30][2], repr(run(cv2.resize(c[30][3][0], None, fx=2, fy=2)))); sys.exit()
res = {}; t0 = time.time()
for oid, kind, tag, ims in crops():
    res[oid] = [run(cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if max(im.shape) < 700 else im) for im in ims]
name = {"glm": "GLM-OCR kirpim", "mineru": "MinerU2.5 kirpim"}[which]
r = score_crops(res, name, {"sure_s": round(time.time() - t0, 1)}); json.dump(r, open(H / f"B_{which}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: r[k] for k in ("door", "window", "sure_s")})
