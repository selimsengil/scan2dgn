"""Second harvest pass: tight level crops (box + 8 cm, x2 cubic, 0 and 180 deg) read in OCR mode by GLM-OCR and PaddleOCR-VL.
All readings stored; a reading is a tag candidate when it has number+number (PAT).  Grading against DGN width/height is done in
atama2.py.   Output dar/<floor>.json {id: {kind, dgn_cm, okumalar: {glm: [t0, t180], vl: [t0, t180]}}}.   5070 Ti only."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from shapely.geometry import shape
from transformers import AutoProcessor, AutoModelForImageTextToText
H = config.KT / "ocr_test"; K = config.KT; D = str(config.MODELS); OUT = H / "dar"; OUT.mkdir(exist_ok=True)
SKIP = ("kisla", "malkoc_361_79__k2")
assert "5070" in torch.cuda.get_device_name(0)
floors = [p.stem for p in sorted((K / "aciklik_ag" / "veri").glob("*.json")) if not any(k in p.stem.lower() for k in SKIP)]
def level(g, o, s, mg=0.08, sc=2.0):
    c = np.array(shape(o["poly"]).minimum_rotated_rectangle.exterior.coords)[:4]; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e))
    dd = c[(k + 1) % 4] - c[k]; ang = np.degrees(np.arctan2(dd[1], dd[0])); m = c.mean(0); L = (max(e) / 2 + mg * s) * sc; Hh = (min(e) / 2 + mg * s) * sc
    M = cv2.getRotationMatrix2D(tuple(m), ang, sc); M[:, 2] += [L - m[0], Hh - m[1]]; v = cv2.warpAffine(g, M, (int(2 * L), int(2 * Hh)), borderValue=255, flags=cv2.INTER_CUBIC)
    return [v, cv2.rotate(v, cv2.ROTATE_180)], max(e) / s * 100
res = {}
for name, MP, prompt in (("glm", D + r"\GLM-OCR", "Text Recognition:"), ("vl", D + r"\PaddleOCR-VL-1.6", "OCR:")):
    model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP); T0 = time.time()
    for fid in floors:
        f = OUT / f"{fid}.json"; R = res.setdefault(fid, json.load(open(f, encoding="utf-8")) if f.exists() else {})
        if R and all(name in v["okumalar"] for v in R.values()): continue
        d = json.load(open(K / "aciklik_ag" / "veri" / f"{fid}.json", encoding="utf-8")); s = d["px_per_m"]
        g = cv2.imdecode(np.fromfile(str(K / "dgn_katlar" / fid / "clean.png"), np.uint8), 0)
        for o in d["openings"]:
            ims, w = level(g, o, s); rr = []
            for im in ims:
                msg = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(im).convert("RGB")}, {"type": "text", "text": prompt}]}]
                inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda"); inp.pop("token_type_ids", None)
                with torch.no_grad(): out = model.generate(**inp, max_new_tokens=40, do_sample=False, repetition_penalty=1.05)
                rr.append(proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).replace("\n", " ")[:60])
            R.setdefault(o["id"], {"kind": o["kind"], "dgn_cm": round(w, 1), "okumalar": {}})["okumalar"][name] = rr
        json.dump(R, open(f, "w", encoding="utf-8"), ensure_ascii=False)
        print(name, fid, len(R), round(time.time() - T0), flush=True)
    del model; torch.cuda.empty_cache()
print("bitti")
