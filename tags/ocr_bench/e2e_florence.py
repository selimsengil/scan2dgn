"""Florence-2-large <OCR_WITH_REGION> on 800 px tiles, sheet turned 4 ways.  usage: python e2e_florence.py KEY  (.venv-ocr3, 5070 Ti)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import test2, tile_util as TU
MP = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\Florence-2-large"; key = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(MP, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / test2.FLOORS[key] / "clean.png"), np.uint8), 0); reads = []; t0 = time.time(); n = 0
for r, x0, y0, t, HW in TU.tiles(g, 800, 700, (0, 90, 180, 270)):
    im = Image.fromarray(t).convert("RGB"); inp = proc(text="<OCR_WITH_REGION>", images=im, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad(): out = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=1024, num_beams=1, do_sample=False)
    txt = proc.batch_decode(out, skip_special_tokens=False)[0]; p = proc.post_process_generation(txt, task="<OCR_WITH_REGION>", image_size=im.size)["<OCR_WITH_REGION>"]
    for q, lab in zip(p["quad_boxes"], p["labels"]):
        lab = lab.replace("</s>", "").replace("<s>", "").strip()
        if lab: reads.append({"box": TU.back(np.asarray(q).reshape(4, 2), r, x0, y0, HW), "text": lab})
    n += 1
test2.save("florence2_large", key, reads, time.time() - t0, {"parca": n})
