"""Score saved model polygons before and after aligning them to the encoded lines (no model call).
usage: python hizala_test.py ornekli_o1 [izgara_g2 ...]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import v0_watershed as V0, cizgi_kod as CK, cizgi_hat as CH, cizgi_hizala as HZ, tek_oda_test as T

im = Image.open(V0.B / "019" / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); W, H = im.size
enc = CK.encode(gray); gts = V0.load_rooms("019")
for tag in sys.argv[1:]:
    tiles = []; a0 = []; a1 = []
    for k, i in enumerate([11, 18, 7, 21, 14]):
        f = T.OUT / f"{tag}_r{i}.json"
        if not f.exists(): continue
        d = json.load(open(f, encoding="utf-8")); pred = d.get("pred") or d["hist"][-1]
        if len(pred) < 3: continue
        g = gts[i]; box = T.crop_box(g["poly"], W, H, k); ls = CH.lines_in(enc, box, min_len=8)
        new, rep = HZ.align(pred, ls)
        s0 = T.score(pred, g["poly"]); s1 = T.score(new, g["poly"]); a0.append(s0["iou"]); a1.append(s1["iou"])
        print(f"{tag} r{i}: IoU {s0['iou']} -> {s1['iou']} | corners<=6px {s0['corners_hit6']} -> {s1['corners_hit6']} of {s0['gt_corners']} | boundary mean {s0['bnd_mean']} -> {s1['bnd_mean']} px | edges matched {sum(e['id'] is not None for e in rep)}/{len(rep)}")
        t = T.tile(im, box, g["poly"], new, f"r{i} hizali IoU {s1['iou']} kose {s1['corners_hit6']}/{s1['gt_corners']} (once {s0['iou']}, {s0['corners_hit6']})"); tiles.append(t)
    if tiles:
        print(f"{tag}: mean IoU {np.mean(a0):.3f} -> {np.mean(a1):.3f}")
        n = (len(tiles) + 2) // 3; cw = max(t.width for t in tiles) + 10; ch = max(t.height for t in tiles) + 10
        sh = Image.new("RGB", (cw * 3, ch * n), "white")
        for j, t in enumerate(tiles): sh.paste(t, ((j % 3) * cw, (j // 3) * ch))
        sh.save(T.OUT / f"sayfa_hizali_{tag}.png")
