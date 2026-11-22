"""DeepSeek-OCR-2 grounding OCR on 1024 px tiles (sheet as is; the model reads rotated text itself or not).
usage: python e2e_deepseek.py KEY  (.venv-ocr3, 5070 Ti)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, re, sys, time, tempfile
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import cv2, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
import test2, tile_util as TU
MP = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\DeepSeek-OCR-2"; key = sys.argv[1]
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
model = AutoModel.from_pretrained(MP, _attn_implementation="eager", trust_remote_code=True, use_safetensors=True).eval().cuda().to(torch.bfloat16)
g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / test2.FLOORS[key] / "clean.png"), np.uint8), 0); reads = []; t0 = time.time(); n = 0
tmp = tempfile.mkdtemp(); RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(.*?)\]\]<\|/det\|>", re.S)
for r, x0, y0, t, HW in TU.tiles(g, 1024, 900, (0,)):
    f = os.path.join(tmp, "t.png"); cv2.imwrite(f, t)
    res = model.infer(tok, prompt="<image>\n<|grounding|>OCR this image. ", image_file=f, output_path=tmp, base_size=1024, image_size=768, crop_mode=False, save_results=False, eval_mode=True)
    h, w = t.shape
    for lab, nums in RE.findall(res or ""):
        for bb in re.findall(r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+", "[" + nums + "]"):
            a, b, c, d = [int(v) for v in bb.split(",")]; X1, Y1, X2, Y2 = a / 999 * w, b / 999 * h, c / 999 * w, d / 999 * h
            reads.append({"box": TU.back([[X1, Y1], [X2, Y1], [X2, Y2], [X1, Y2]], r, x0, y0, HW), "text": lab.strip()})
    n += 1
    if n == 1: print("ilk parca", (res or "")[:300], flush=True)
test2.save("deepseek_ocr2", key, reads, time.time() - t0, {"parca": n})
