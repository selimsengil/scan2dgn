"""Does Qwen point better in its own grounding format (0-1000 normalised) than by reading a ruler?

Same four Tekke rooms and prompt rules as qwen_test 'see', but the view is the plain crop (no ruler)
with the entrance in orange, and the corners are asked as 0..1000 x 0..1000 coordinates of the image.
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import ImageDraw

import qwen_client as QC
import qwen_test as QT

Q = ("Görüntü bir kat planından '{target}' odasının kesiti. TURUNCU çizgi odanın giriş kapısı açıklığı (sınır buradan düz geçer).\n"
     + QT.RULES +
     "\nOdanın iç sınırının köşelerini saat yönünde sırayla işaretle. Koordinatları görüntünün 0-1000 normalize koordinatlarında "
     "ver (sol üst [0,0], sağ alt [1000,1000]). Yalnız JSON: {{\"points\": [[x, y], ...]}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("runs") / "qwen_ground_a"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, crop, target, hint, ref in QT.cases():
        v = crop.copy(); d = ImageDraw.Draw(v)
        (p, q) = hint.jambs
        d.line([tuple(p), tuple(q)], fill="#ef5000", width=3)
        v.save(out / f"{name}.png")
        W, H = v.size
        t0 = time.time()
        ans, tm = QC.ask(Q.format(target=target), [np.asarray(v)[:, :, ::-1]], think=a.think, max_tokens=6000 if a.think else 800)
        pts = QT.parse_json(ans).get("points") or []
        try:
            px = [[float(x) * W / 1000.0, float(y) * H / 1000.0] for x, y in pts]
        except Exception:
            px = []
        sc = QT.brief(QT.score(px, ref, hint)) if len(px) >= 3 else {"valid": False, "reason": "fewer than 3 points"}
        res[name] = {"points_norm": pts, "points_px": px, "score": sc, "seconds": round(time.time() - t0, 1), "raw": ans[-400:]}
        print(name, sc, res[name]["seconds"], "s", flush=True)
        json.dump(res, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
