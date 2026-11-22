"""SAM 2.1 (hiera-large, Apache-2.0) on the exported rooms, zero-shot, prompted with the room's label point.
Runs in .venv-spec.

SAM returns three candidate masks for a point. The label point sits on text, so the smallest mask is often
just the letters: the answer is the largest candidate that contains the point and covers < 60 % of the image.
Mask -> polygon: the outer contour of that mask's connected component at the point, simplified with
Douglas-Peucker at 0.6 % of the perimeter (keeps column notches).

usage: .venv-spec python spec_sam2.py [crop|ctx]  ->  runs/spec_inputs/sam2_<mode>.json
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

IN = Path(__file__).with_name("runs") / "spec_inputs"
MODEL = "facebook/sam2.1-hiera-large"


def mask_polygon(mask, pt):
    m = mask.astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    x, y = int(round(pt[0])), int(round(pt[1]))
    x = min(max(x, 0), m.shape[1] - 1); y = min(max(y, 0), m.shape[0] - 1)
    k = lab[y, x]
    if k == 0:
        sizes = [(lab == i).sum() for i in range(1, n)]
        if not sizes:
            return []
        k = 1 + int(np.argmax(sizes))
    comp = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    eps = 0.006 * cv2.arcLength(c, True)
    a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    return a.astype(float).tolist()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "crop"
    dev = "cuda"
    proc = Sam2Processor.from_pretrained(MODEL)
    model = Sam2Model.from_pretrained(MODEL).to(dev).eval()
    meta = json.load(open(IN / "rooms.json", encoding="utf-8"))
    out = {}
    for key, m in meta.items():
        img = Image.open(IN / f"{key}_{mode}.png").convert("RGB")
        lx, ly = m[mode]["label"]
        inp = proc(images=img, input_points=[[[[lx, ly]]]], input_labels=[[[1]]], return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp, multimask_output=True)
        masks = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy()  # (3, H, W)
        scores = o.iou_scores[0, 0].float().cpu().numpy().tolist()
        H, W = masks.shape[-2:]
        cands = []
        for i in range(masks.shape[0]):
            mk = masks[i] > 0
            area = mk.mean()
            yi, xi = min(max(int(ly), 0), H - 1), min(max(int(lx), 0), W - 1)
            cands.append({"i": i, "area": float(area), "has_pt": bool(mk[yi, xi]), "score": scores[i]})
        ok = [c for c in cands if c["has_pt"] and c["area"] < 0.60]
        best = max(ok, key=lambda c: c["area"]) if ok else max(cands, key=lambda c: c["score"])
        poly = mask_polygon(masks[best["i"]] > 0, (lx, ly))
        out[key] = {"pick": poly, "cands": cands, "chosen": best["i"]}
        print(key, "chosen", best, "corners", len(poly), flush=True)
    json.dump(out, open(IN / f"sam2_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
