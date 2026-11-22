"""Picture sheet of a floor-test result (SONUC_<floor>.json + SKOR.json in a run folder): every GT room with the
prediction matched to it (green = GT, blue = model). usage: python sonuc_sayfa.py <run dir> [floor]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import v0_watershed as V0, tek_oda_test as T

run = Path(sys.argv[1]); floor = sys.argv[2] if len(sys.argv) > 2 else "019"; assert floor != "015"
im = Image.open(V0.B / floor / "clean.png").convert("RGB"); W, H = im.size
gts = [g for g in V0.load_rooms(floor) if g["poly"].geom_type == "Polygon"]
res = [r for r in json.load(open(run / f"SONUC_{floor}.json", encoding="utf-8"))["rooms"] if r.get("type") != "balcony" and r.get("boundary") and len(r["boundary"]) >= 3]; sk = json.load(open(run / "SKOR.json", encoding="utf-8"))
tiles = []
for r in sk["rooms"]:
    g = gts[r["dgn"]]; x0, y0, x1, y1 = g["poly"].bounds; m = 45
    box = (max(0, int(x0 - m)), max(0, int(y0 - m)), min(W, int(x1 + m)), min(H, int(y1 + m)))
    pred = res[r["pred"]]["boundary"] if r.get("pred") is not None else None
    tiles.append(T.tile(im, box, g["poly"], [list(p) for p in pred] if pred else None, f"r{r['dgn']} {g['name']} IoU {r['IoU']}", size=400))
cols = 7; n = (len(tiles) + cols - 1) // cols; cw = max(t.width for t in tiles) + 6; ch = max(t.height for t in tiles) + 6
sh = Image.new("RGB", (cw * cols, ch * n), "white")
for j, t in enumerate(tiles): sh.paste(t, ((j % cols) * cw, (j // cols) * ch))
sh.save(run / "sayfa.png"); print(run / "sayfa.png", sh.size)
