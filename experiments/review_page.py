"""HTML review page for room drawings: per room the source crop with the DGN room (green), and each
drawn version (coarse orange, snapped magenta, ...) overlaid, plus the scores.

usage: python review_page.py RUN_DIR      (RUN_DIR/RESULT.json from qwen_room.py; writes RUN_DIR/INCELE.html)
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import html
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import qwen_test as QT
import snap

COLORS = {"reference": (0, 170, 60), "coarse": (255, 130, 0), "snapped": (200, 0, 200), "refined": (0, 110, 255)}
NAMES = {"reference": "DGN (doğru)", "coarse": "Model kaba", "snapped": "Model + oturtma", "refined": "Model ayrıntı"}


def overlay(crop: Image.Image, polys: dict, path: Path, up: float = 1.0):
    im = crop.convert("RGB")
    if up != 1.0:
        im = im.resize((int(im.width * up), int(im.height * up)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    for key, pts in polys.items():
        if not pts or len(pts) < 3:
            continue
        q = [(x * up, y * up) for x, y in pts]
        d.line(q + q[:1], fill=COLORS[key], width=3 if key == "reference" else 2)
        for x, y in q:
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=COLORS[key], width=2)
    im.save(path)


def page(run: Path):
    res = json.load(open(run / "RESULT.json", encoding="utf-8"))
    cases = {name: (crop, target, hint, ref) for name, crop, target, hint, ref in QT.cases()}
    rows = []
    for name, r in res.items():
        crop, target, hint, ref = cases[name]
        polys = {"reference": ref, "coarse": r.get("coarse")}
        if r.get("coarse") and len(r["coarse"]) >= 3:
            polys["snapped"] = snap.snap_polygon(crop, r["coarse"])
        if r.get("refined"):
            polys["refined"] = r["refined"]
        overlay(crop, polys, run / f"{name}_review.png")
        scores = {k: QT.brief(QT.score(v, ref, hint)) for k, v in polys.items() if k != "reference" and v and len(v) >= 3}
        rows.append((name, target, scores, r.get("seconds")))
    legend = " ".join(f'<span style="color:rgb{COLORS[k]}">■ {NAMES[k]}</span>' for k in COLORS)
    parts = ["<!doctype html><meta charset='utf-8'><title>Oda çizim inceleme</title>",
             "<style>body{font-family:Segoe UI,Arial;margin:16px;background:#f5f6f8}h1{font-size:20px}"
             ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}"
             "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}"
             "img{max-width:100%;border:1px solid #ccc}.ok{background:#dff5df}.mid{background:#fff3cd}.bad{background:#f8d7da}</style>",
             f"<h1>Oda çizim inceleme — {html.escape(run.name)}</h1><p>{legend}</p>",
             "<p>IoU: alan örtüşmesi (1 = tam). F1@5: sınırın gerçek sınıra 5 piksel içindeki oranı. Köşe@10: köşelerin 10 piksel "
             "içindeki oranı. Köşe sayısı: model / DGN.</p>"]
    for name, target, scores, secs in rows:
        parts.append(f"<div class='card'><h2>{html.escape(name)} — {html.escape(target)}</h2>")
        parts.append("<table><tr><th>Sürüm</th><th>IoU</th><th>F1@5</th><th>F1@10</th><th>Köşe@10</th><th>Köşe sayısı</th></tr>")
        for k, s in scores.items():
            if not s.get("valid", True):
                parts.append(f"<tr><td>{NAMES[k]}</td><td colspan=5>geçersiz: {html.escape(str(s.get('reason')))}</td></tr>"); continue
            cls = "ok" if s["IoU"] >= 0.95 and s["F1@5"] >= 0.9 else "mid" if s["IoU"] >= 0.9 else "bad"
            parts.append(f"<tr class='{cls}'><td>{NAMES[k]}</td><td>{s['IoU']}</td><td>{s['F1@5']}</td><td>{s['F1@10']}</td>"
                         f"<td>{s['corners@10']}</td><td>{s['turns'][0]} / {s['turns'][1]}</td></tr>")
        parts.append(f"</table><p>Süre: {secs} sn</p><img src='{name}_review.png'></div>")
    (run / "INCELE.html").write_text("\n".join(parts), encoding="utf-8")
    return run / "INCELE.html"


if __name__ == "__main__":
    print(page(Path(sys.argv[1])))
