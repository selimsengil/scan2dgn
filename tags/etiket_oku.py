"""Read the tag regions found on a sheet: level crop of each region (+ margin, x2 cubic, 0/180 deg), GLM-OCR and
PaddleOCR-VL in OCR mode, keep the readings that contain number+number.  No door/window decision here.
usage: python etiket_oku.py FLOOR REGIONS.json OUT.json [--mg 0.10]      (.venv-spec, 5070 Ti only)
OUT: [{box, text, okuyucu, sayilar:[a,b], onek, oy}] one per region with a number pair (majority over the 4 readings)."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, os, re
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from collections import Counter
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
K = config.KT; D = str(config.MODELS)
PAT = re.compile(r"([A-Z]{1,2}\d{0,2})?[\s\-]*(\d{2,3})[/xX×*1Il|]?(\d{3})")


def parse(t):
    t = t.upper().replace("O", "0").replace(" ", ""); out = []
    for m in PAT.finditer(t):
        a, b = int(m.group(2)), int(m.group(3))
        if 30 <= a <= 400 and 100 <= b <= 300: out.append((m.group(1) or "", a, b))
    return out


def level(g, box, s, mg, sc=2.0):
    c = np.asarray(box, float)[:4]; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e))
    dd = c[(k + 1) % 4] - c[k]; ang = np.degrees(np.arctan2(dd[1], dd[0])); m = c.mean(0); L = (max(e) / 2 + mg * s) * sc; Hh = (min(e) / 2 + mg * s) * sc
    M = cv2.getRotationMatrix2D(tuple(m), ang, sc); M[:, 2] += [L - m[0], Hh - m[1]]
    v = cv2.warpAffine(g, M, (max(8, int(2 * L)), max(8, int(2 * Hh))), borderValue=255, flags=cv2.INTER_CUBIC)
    return [v, cv2.rotate(v, cv2.ROTATE_180)]


def oku(fid, regs, mg=0.10, readers=("glm", "vl")):
    L = json.load(open(K / "dgn_katlar" / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
    g = cv2.imdecode(np.fromfile(str(K / "dgn_katlar" / fid / "clean.png"), np.uint8), 0); texts = [[] for _ in regs]
    assert "5070" in torch.cuda.get_device_name(0)
    for name, MP, prompt in [x for x in (("glm", D + r"\GLM-OCR", "Text Recognition:"), ("vl", D + r"\PaddleOCR-VL-1.6", "OCR:")) if x[0] in readers]:
        model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP)
        for i, rg in enumerate(regs):
            for im in level(g, rg["box"], s, mg):
                msg = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(im).convert("RGB")}, {"type": "text", "text": prompt}]}]
                inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda"); inp.pop("token_type_ids", None)
                with torch.no_grad(): out = model.generate(**inp, max_new_tokens=40, do_sample=False, repetition_penalty=1.05)
                texts[i].append((name, proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).replace("\n", " ")[:60]))
        del model; torch.cuda.empty_cache()
    res = []
    for rg, ts in zip(regs, texts):
        cands = [(c, n, t) for n, t in ts for c in parse(t)]
        if not cands: continue
        votes = Counter((c[1], c[2]) for c, _, _ in cands); (a, b), v = votes.most_common(1)[0]
        pre = Counter(c[0] for c, _, _ in cands if (c[1], c[2]) == (a, b) and c[0]).most_common(1)
        res.append({"box": rg["box"], "text": f"{pre[0][0] + ' ' if pre else ''}{a}/{b}", "sayilar": [a, b], "onek": pre[0][0] if pre else "", "oy": v,
                    "okumalar": ts, "score": rg.get("score")})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("floor"); ap.add_argument("regions"); ap.add_argument("out"); ap.add_argument("--mg", type=float, default=0.10); a = ap.parse_args()
    regs = json.load(open(a.regions, encoding="utf-8")); res = oku(a.floor, regs, a.mg)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0); print(len(regs), "bolge,", len(res), "sayi cifti")
