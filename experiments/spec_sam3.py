"""SAM 3 (or 3.1) on the exported rooms, zero-shot, with the official code's SAM-1-style interactive predictor.
Runs in .venv-sam3. The checkpoint is loaded by Meta's own loader, which uses torch.load(weights_only=True).

prompts:
  point     the room's label point only (largest of the 3 candidates that holds the point and is < 60 % of the image)
  r2s       label point + the bounding box of Raster2Seq's polygon
  qwen      label point + the bounding box of Qwen's (no-shot) polygon

usage: .venv-sam3 python spec_sam3.py CKPT TAG [point|r2s|qwen ...]  ->  runs/spec_inputs/<TAG>_<prompt>.json
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from pathlib import Path

import types

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision  # noqa: E402,F401  -- before the stub: torch._dynamo probes for a real triton at import
import torchvision.ops  # noqa: E402,F401
import torch._dynamo  # noqa: E402,F401
torch.cuda.init()   # CUDA's lazy init also probes triton; run it before the stub exists

# Triton has no Windows build. SAM 3 imports it at module level only for a Euclidean distance transform
# (sam3/model/edt.py, "meant to mimic cv2.distanceTransform(input, cv2.DIST_L2, 0)") and for training/NMS
# kernels. A stub lets the modules import; the EDT is replaced with OpenCV's.
if "triton" not in sys.modules:
    _tr, _tl = types.ModuleType("triton"), types.ModuleType("triton.language")
    _tr.jit = lambda f=None, **kw: f if f is not None else (lambda g: g)
    _tr.autotune = lambda *a, **k: (lambda g: g)
    _tr.heuristics = lambda *a, **k: (lambda g: g)
    _tr.Config = lambda *a, **k: None
    _tr.cdiv = lambda a, b: -(-a // b)
    _tl.constexpr = int
    _tr.language = _tl
    sys.modules["triton"], sys.modules["triton.language"] = _tr, _tl


def _edt_cv2(data: torch.Tensor):
    """cv2 version of sam3.model.edt.edt_triton: (B, H, W) binary -> L2 distance to the nearest zero pixel."""
    arr = data.detach().to("cpu").numpy().astype(np.uint8)
    out = np.stack([cv2.distanceTransform(a, cv2.DIST_L2, 0) for a in arr])
    return torch.from_numpy(out).to(data.device)


import sam3.model.edt as _edt  # noqa: E402
_edt.edt_triton = _edt_cv2
import sam3.model.sam3_tracker_utils as _tu  # noqa: E402
_tu.edt_triton = _edt_cv2

from sam3 import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor

IN = Path(__file__).with_name("runs") / "spec_inputs"


def mask_polygon(mask, pt):
    m = mask.astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    x = min(max(int(round(pt[0])), 0), m.shape[1] - 1)
    y = min(max(int(round(pt[1])), 0), m.shape[0] - 1)
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
    return cv2.approxPolyDP(c, 0.006 * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float).tolist()


def main():
    ckpt, tag, modes = sys.argv[1], sys.argv[2], sys.argv[3:] or ["point"]
    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=False, enable_inst_interactivity=True)
    proc = Sam3Processor(model)
    meta = json.load(open(IN / "rooms.json", encoding="utf-8"))
    prompts = json.load(open(IN / "prompts.json", encoding="utf-8"))
    res = {m: {} for m in modes}
    for key, m in meta.items():
        img = Image.open(IN / f"{key}_crop.png").convert("RGB")
        W, H = img.size
        lx, ly = m["crop"]["label"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            for mode in modes:
                pt, lb = np.array([[lx, ly]]), np.array([1])
                if mode == "point":
                    masks, scores, _ = model.predict_inst(state, point_coords=pt, point_labels=lb, multimask_output=True)
                    cand = []
                    for i in range(len(masks)):
                        mk = masks[i] > 0
                        has = bool(mk[min(max(int(ly), 0), H - 1), min(max(int(lx), 0), W - 1)])
                        cand.append((i, float(mk.mean()), has, float(scores[i])))
                    ok = [c for c in cand if c[2] and c[1] < 0.60]
                    best = max(ok, key=lambda c: c[1]) if ok else max(cand, key=lambda c: c[3])
                    mk = masks[best[0]] > 0
                    info = {"cands": cand, "chosen": best[0]}
                else:
                    p = prompts.get(key, {}).get(mode)
                    if not p:
                        res[mode][key] = {"pick": []}
                        continue
                    box = np.array(p["box"], float)
                    pad = 0.04 * max(box[2] - box[0], box[3] - box[1])   # a little room for a coarse box
                    box = np.array([max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad)])
                    masks, scores, _ = model.predict_inst(state, point_coords=pt, point_labels=lb, box=box[None, :],
                                                          multimask_output=False)
                    mk = masks[0] > 0
                    info = {"score": float(scores[0])}
                poly = mask_polygon(mk, (lx, ly))
                res[mode][key] = {"pick": poly, **info}
                print(key, mode, "corners", len(poly), info if mode != "point" else info["chosen"], flush=True)
    for mode in modes:
        json.dump(res[mode], open(IN / f"{tag}_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
