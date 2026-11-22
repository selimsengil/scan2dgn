"""Room boundary by Qwen3.8-27B in its own grounding format (0-1000), coarse then per edge.

  coarse : the room crop with the verified entrance in orange; Qwen lists the inner corners (0-1000).
  edges  : for each coarse edge A->B, a crop along that wall (upscaled so its long side is ~1000 px),
           the room's inside tinted blue, A green and B red; Qwen lists every turning point from A to B
           along the room's inner face (column notches included), A and B themselves corrected.
  merge  : the edge walks are joined; a corner shared by two walks is their average.
Scored with the benchmark metrics against the DGN room. Images and polygons are saved per room for
the review page (review_page.py).

usage: python qwen_room.py [--cases 01_mutfak,...] [--out runs/room_a]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import qwen_client as QC
import qwen_test as QT

Q_COARSE = ("Görüntü bir kat planından '{target}' odasının kesiti. TURUNCU çizgi odanın giriş kapısı açıklığı (sınır buradan düz geçer).\n"
            + QT.RULES +
            "\nOdanın iç sınırının köşelerini saat yönünde sırayla işaretle. Koordinatları görüntünün 0-1000 normalize "
            "koordinatlarında ver (sol üst [0,0], sağ alt [1000,1000]). Yalnız JSON: {{\"points\": [[x, y], ...]}}")

Q_EDGE = ("Görüntü bir kat planında '{target}' odasının bir duvarının büyütülmüş yakın görünümü. Odanın içi AÇIK MAVİ boyalı "
          "(yaklaşık). YEŞİL nokta A, KIRMIZI nokta B: odanın sınırının bu duvar boyunca yaklaşık başı ve sonu. TURUNCU çizgi "
          "varsa odanın giriş kapısı açıklığıdır (sınır oradan düz geçer).\n" + QT.RULES +
          "\nA'dan başlayıp B'ye kadar odanın İÇ YÜZ sınırını takip et ve yol üzerindeki bütün köşe/dönüş noktalarını sırayla "
          "ver: A'nın düzeltilmiş konumu (gerçek iç köşe), aradaki kolon girintisi ya da duvar basamağı dönüşleri, ve B'nin "
          "düzeltilmiş konumu. Arada dönüş yoksa yalnız iki nokta ver. Koordinatlar bu görüntünün 0-1000 normalize "
          "koordinatları. Yalnız JSON: {{\"points\": [[x, y], ...]}}")


def ask_points(question, img: Image.Image, max_tokens=900):
    ans, tm = QC.ask(question, [np.asarray(img.convert("RGB"))[:, :, ::-1]], think=False, max_tokens=max_tokens)
    pts = QT.parse_json(ans).get("points") or []
    out = []
    for p in pts:
        try:
            out.append([float(p[0]), float(p[1])])
        except Exception:
            pass
    return out, ans


def coarse(crop: Image.Image, target, hint):
    v = crop.convert("RGB").copy(); d = ImageDraw.Draw(v)
    a, b = hint.jambs; d.line([tuple(a), tuple(b)], fill="#ef5000", width=3)
    pts, raw = ask_points(Q_COARSE.format(target=target), v)
    W, H = crop.size
    return [[x * W / 1000.0, y * H / 1000.0] for x, y in pts], raw


def edge_walk(crop: Image.Image, target, hint, poly, i, margin=70, long_side=1000, max_up=3.0):
    """Turning points from vertex i to i+1 of the coarse polygon, in crop pixels."""
    W, H = crop.size
    A, B = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
    # a SQUARE window centred on the edge: Qwen's 0-1000 points drift on long thin images
    # (1000x214 strips put every edge ~20-27 px inside the room)
    c = (A + B) / 2
    side = int(max(abs(B[0] - A[0]), abs(B[1] - A[1])) + 2 * margin)
    x0, y0 = int(c[0] - side / 2), int(c[1] - side / 2)
    x1, y1 = x0 + side, y0 + side
    reg = Image.new("RGB", (side, side), "white")
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        reg.paste(crop.convert("RGB").crop((sx0, sy0, sx1, sy1)), (sx0 - x0, sy0 - y0))
    s = min(max_up, long_side / max(reg.width, reg.height))
    s = max(1.0, s)
    big = reg.resize((max(1, int(reg.width * s)), max(1, int(reg.height * s))), Image.Resampling.NEAREST)
    # the room's inside, tinted (the coarse polygon clipped to the window)
    tint = Image.new("L", big.size, 0); td = ImageDraw.Draw(tint)
    td.polygon([((x - x0) * s, (y - y0) * s) for x, y in poly], fill=70)
    blue = Image.new("RGB", big.size, (90, 160, 255))
    big = Image.composite(blue, big, tint)
    big = Image.blend(big, reg.resize(big.size, Image.Resampling.NEAREST), 0.0) if False else big
    d = ImageDraw.Draw(big)
    ja, jb = hint.jambs
    d.line([((ja[0] - x0) * s, (ja[1] - y0) * s), ((jb[0] - x0) * s, (jb[1] - y0) * s)], fill="#ef5000", width=3)
    r = 7
    for P, col in ((A, (0, 170, 0)), (B, (220, 0, 0))):
        X, Y = (P[0] - x0) * s, (P[1] - y0) * s
        d.ellipse((X - r, Y - r, X + r, Y + r), outline=col, width=3)
    pts, raw = ask_points(Q_EDGE.format(target=target), big)
    bw, bh = big.size
    px = [[x0 + (x * bw / 1000.0) / s, y0 + (y * bh / 1000.0) / s] for x, y in pts]
    return px, big, raw


def merge(walks, poly):
    """Join edge walks: walk i runs from corner i to corner i+1; a shared corner is averaged."""
    n = len(poly)
    out = []
    for i, w in enumerate(walks):
        if len(w) < 2:   # no answer: keep the coarse corner
            w = [poly[i], poly[(i + 1) % n]]
        start = w[0]
        prev_end = walks[i - 1][-1] if len(walks[i - 1]) >= 2 else poly[i]
        corner = [(start[0] + prev_end[0]) / 2, (start[1] + prev_end[1]) / 2]
        out.append(corner)
        out.extend(w[1:-1])
    # drop near-duplicates and collinear points
    clean = []
    for p in out:
        if clean and math.dist(p, clean[-1]) < 3:
            continue
        clean.append(p)
    if len(clean) > 3 and math.dist(clean[0], clean[-1]) < 3:
        clean.pop()
    return clean


def run(name, crop, target, hint, ref, out: Path):
    t0 = time.time()
    c, raw_c = coarse(crop, target, hint)
    res = {"target": target, "coarse": c, "coarse_score": QT.brief(QT.score(c, ref, hint)) if len(c) >= 3 else {"valid": False}}
    walks, raws = [], []
    if len(c) >= 3:
        for i in range(len(c)):
            w, img, raw = edge_walk(crop, target, hint, c, i)
            img.save(out / f"{name}_edge{i:02d}.png")
            walks.append(w); raws.append(raw[-200:])
        fin = merge(walks, c)
        res["refined"] = fin
        res["refined_score"] = QT.brief(QT.score(fin, ref, hint)) if len(fin) >= 3 else {"valid": False}
    res["walks"] = walks
    res["reference"] = ref
    res["seconds"] = round(time.time() - t0, 1)
    crop.convert("RGB").save(out / f"{name}_crop.png")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--out", default=str(Path(__file__).with_name("runs") / "room_a"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want = set(a.cases.split(",")) if a.cases else None
    allr = {}
    for name, crop, target, hint, ref in QT.cases():
        if want and name not in want:
            continue
        r = run(name, crop, target, hint, ref, out)
        r["hint"] = {"jambs": [list(hint.jambs[0]), list(hint.jambs[1])], "inside": list(hint.inside)}
        allr[name] = r
        print(name, "coarse", r["coarse_score"], "| refined", r.get("refined_score"), "|", r["seconds"], "s", flush=True)
        json.dump(allr, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
