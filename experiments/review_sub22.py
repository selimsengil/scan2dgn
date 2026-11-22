"""HTML review of the 22-room comparison: for every room one tile per model, the DGN room (green)
and the model's snapped polygon (magenta) drawn on the crop, IoU on top.

usage: python review_sub22.py  ->  runs/INCELE_22oda.html
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

R = Path(__file__).with_name("runs")
RUNS = [("Qwen 27B Q4 (2 kart)", "qwenq4_sub22"), ("Qwen 27B Q5 (3 kart)", "qwenq5_sub22"),
        ("Strata DeepSeek IQ2", "strata_sub22"), ("DeepSeek 500 GB", "ds500vl_sub22")]
TILE = 360


def tile(run_dir: Path, key: str, r: dict) -> str:
    p = run_dir / f"{key}.png"
    if not p.exists():
        return ""
    im = Image.open(p).convert("RGB")
    s = TILE / max(im.size)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    d = ImageDraw.Draw(im)
    for pts, col, w in ((r.get("reference"), (0, 170, 60), 3), (r.get("snapped") or r.get("points"), (210, 0, 210), 2)):
        if pts and len(pts) >= 3:
            q = [(x * s, y * s) for x, y in pts]
            d.line(q + q[:1], fill=col, width=w)
    b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()


def main():
    data = {tag: json.load(open(R / f"b132_{tag}" / "RESULT.json", encoding="utf-8")) for _, tag in RUNS}
    keys = sorted(data[RUNS[0][1]])
    means = {}
    for label, tag in RUNS:
        v = [data[tag][k]["snap"].get("IoU", 0.0) if data[tag].get(k, {}).get("snap", {}).get("valid", True) else 0.0
             for k in keys if k in data[tag]]
        means[label] = (sum(v) / len(v), sum(x >= .9 for x in v), len(v),
                        sum(data[tag][k]["seconds"] for k in keys if k in data[tag]) / len(v))
    css = """
    :root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e2e2dc;--good:#1f8a4c;--mid:#b7791f;--bad:#c0392b}
    @media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ececE8;--mut:#9a9a94;--line:#33332f}}
    body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
    main{max-width:1560px;margin:0 auto;padding:20px 16px}
    h1{font-size:20px;margin:0 0 4px} p.n{color:var(--mut);margin:0 0 16px}
    table.s{border-collapse:collapse;margin:0 0 20px;background:var(--card)}
    table.s td,table.s th{border:1px solid var(--line);padding:6px 10px;text-align:right} table.s th:first-child,table.s td:first-child{text-align:left}
    .room{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0 0 14px}
    .room h2{font-size:15px;margin:0 0 8px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
    figure{margin:0} figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
    figcaption{font-size:12.5px;color:var(--mut);display:flex;justify-content:space-between;gap:6px}
    b.g{color:var(--good)} b.m{color:var(--mid)} b.b{color:var(--bad)}
    .key span{display:inline-block;width:14px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
    """
    out = ["<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>22 Oda Karşılaştırma</title><style>{css}</style></head><body><main>",
           "<h1>22 oda — model karşılaştırması</h1>",
           "<p class='n key'>Her karede: <span style='background:#00aa3c'></span>DGN'deki doğru sınır "
           "<span style='background:#d200d2'></span>modelin çizimi (snap sonrası). IoU 1.0 = birebir aynı. "
           "Kesit DGN'den alınır, kırmızı nokta çizilecek odayı gösterir.</p>",
           "<table class='s'><tr><th>Model</th><th>Ortalama IoU</th><th>IoU ≥ 0.90</th><th>sn/oda</th></tr>"]
    for label, (m, g, n, sec) in means.items():
        out.append(f"<tr><td>{label}</td><td>{m:.3f}</td><td>{g}/{n}</td><td>{sec:.1f}</td></tr>")
    out.append("</table>")
    for k in keys:
        name = data[RUNS[0][1]][k]["name"]
        out.append(f"<section class='room'><h2>{k} — {name}</h2><div class='grid'>")
        for label, tag in RUNS:
            r = data[tag].get(k)
            if not r:
                continue
            iou = r["snap"].get("IoU", 0.0) if r["snap"].get("valid", True) else 0.0
            cls = "g" if iou >= .9 else "m" if iou >= .6 else "b"
            img = tile(R / f"b132_{tag}", k, r)
            out.append(f"<figure><img alt='{label} {k}' src='data:image/jpeg;base64,{img}'>"
                       f"<figcaption><span>{label}</span><b class='{cls}'>IoU {iou:.2f}</b></figcaption></figure>")
        out.append("</div></section>")
    out.append("</main></body></html>")
    dst = R / "INCELE_22oda.html"
    dst.write_text("".join(out), encoding="utf-8")
    print(dst, round(dst.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
