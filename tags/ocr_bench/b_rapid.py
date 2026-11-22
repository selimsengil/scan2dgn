"""B: RapidOCR on the level crops: full det+rec on the crop, texts joined."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, time
from ortak import *
from rapidocr_onnxruntime import RapidOCR
eng = RapidOCR(); res = {}; t0 = time.time()
for oid, kind, tag, ims in crops():
    outs = []
    for im in ims:
        r, _ = eng(cv2.resize(im, None, fx=1.5, fy=1.5)); outs.append(" ".join(x[1] for x in (r or [])))
    res[oid] = outs
r = score_crops(res, "RapidOCR kirpim", {"sure_s": round(time.time() - t0, 1)}); json.dump(r, open(H / "B_rapid.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: r[k] for k in ("door", "window", "sure_s")}); print([x["okunan"] for x in r["satirlar"] if x["kind"] == "window"][:6])
