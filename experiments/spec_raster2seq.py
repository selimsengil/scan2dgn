"""Raster2Seq (CubiCasa5K checkpoint) on the exported rooms, zero-shot. Runs in .venv-spec.

The deformable-attention CUDA extension is not compiled on Windows: its pure-PyTorch reference
implementation (shipped in the same file) is used instead. For every image the model returns all room
polygons (and door/window lines); the room's answer is the polygon containing the label point, else the
one whose centroid is nearest to it.

usage: .venv-spec python spec_raster2seq.py [crop|ctx]  ->  runs/spec_inputs/r2s_<mode>.json
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from shapely.geometry import Point, Polygon

REPO = Path(r"C:\Users\Selim\Documents\Proje\third_party\specialists\raster2seq_src\Raster2Seq-master")
IN = Path(__file__).with_name("runs") / "spec_inputs"
sys.path.insert(0, str(REPO))
sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")
import matplotlib  # noqa: E402
import matplotlib.cm  # noqa: E402
if not hasattr(matplotlib.cm, "get_cmap"):   # removed in matplotlib 3.9; the repo still imports it
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap

from models.ops.functions import ms_deform_attn_func as F  # noqa: E402
from models.ops.modules import ms_deform_attn as M  # noqa: E402


class _PT:
    @staticmethod
    def apply(value, shapes, level_start, loc, weights, step):
        return F.ms_deform_attn_core_pytorch(value, shapes, loc, weights)


M.MSDeformAttnFunction = _PT

import predict as P  # noqa: E402
from engine import generate  # noqa: E402
from models import build_model  # noqa: E402
from raster2seq_hub import resolve_checkpoint_path  # noqa: E402

FLAGS = ["--dataset_name=cubicasa", "--checkpoint=hf:cubicasa5k", "--semantic_classes=12", "--input_channels", "3",
         "--poly2seq", "--seq_len", "512", "--num_bins", "32", "--disable_poly_refine", "--dec_attn_concat_src",
         "--per_token_sem_loss", "--use_anchor", "--ema4eval"]
SIZE = 256


def load():
    args = P.get_args_parser().parse_args(FLAGS)
    if args.disable_poly_refine:   # as predict.py's __main__ does
        args.with_poly_refine = False
    from datasets.discrete_tokenizer import DiscreteTokenizer
    tok = DiscreteTokenizer(args.num_bins, args.seq_len, add_cls=args.add_cls_token)
    args.vocab_size = len(tok)
    model = build_model(args, train=False, tokenizer=tok).cuda().eval()
    ck = torch.load(resolve_checkpoint_path(args.checkpoint), map_location="cpu", weights_only=False)
    sd = ck["ema"] if args.ema4eval else ck["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("missing", len(miss), "unexpected", len(unexp), flush=True)
    return model, args


def to_input(img):
    """ResizeAndPad as in predict.py: keep aspect, longest side -> 256, pad white (top-left)."""
    w, h = img.size
    s = SIZE / max(w, h)
    im = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    canvas.paste(im, (0, 0))
    x = torch.as_tensor(np.array(canvas)).permute(2, 0, 1).float() / 255.0
    return x, s


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "crop"
    meta = json.load(open(IN / "rooms.json", encoding="utf-8"))
    model, args = load()
    out = {}
    for key, m in meta.items():
        img = Image.open(IN / f"{key}_{mode}.png").convert("RGB")
        x, s = to_input(img)
        with torch.no_grad():
            o = generate(model, x[None].cuda(), semantic_rich=True, use_cache=True, per_token_sem_loss=True, poly2seq=True)
        polys, labels = o["room"][0], o["labels"][0]
        rooms = []
        for p, lab in zip(polys, labels):
            p = np.asarray(p, float).reshape(-1, 2)
            if len(p) >= 3:
                rooms.append({"pts": (p / s).tolist(), "label": int(lab) if lab is not None else -1})
        lx, ly = m[mode]["label"]
        pick = None
        for r in rooms:
            try:
                if Polygon(r["pts"]).buffer(0).contains(Point(lx, ly)):
                    pick = r; break
            except Exception:
                pass
        if pick is None and rooms:
            pick = min(rooms, key=lambda r: Polygon(r["pts"]).buffer(0).centroid.distance(Point(lx, ly)))
        out[key] = {"pick": pick["pts"] if pick else [], "contains": bool(pick) and Polygon(pick["pts"]).buffer(0).contains(Point(lx, ly)),
                    "n_rooms": len(rooms), "all": rooms}
        print(key, "rooms", len(rooms), "picked", len(out[key]["pick"]), "contains", out[key]["contains"], flush=True)
    json.dump(out, open(IN / f"r2s_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
