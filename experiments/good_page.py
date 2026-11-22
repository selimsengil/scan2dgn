"""HTML page of the rooms of a bench132 run with IoU >= a threshold (or below it with --below).
DGN green, model magenta, red dot = the label.

usage: python good_page.py TAG [MIN_IOU] [--below]  ->  runs/<TAG>_iou<MIN>.html
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from pathlib import Path

from review_sub22 import tile

R = Path(__file__).with_name("runs")


def iou(r):
    x = r.get("snap", {})
    return x.get("IoU", 0.0) if x.get("valid", True) and "IoU" in x else 0.0


def main():
    tag = sys.argv[1]
    lo = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 0.90
    below = "--below" in sys.argv
    d = R / f"b132_{tag}"
    res = json.load(open(d / "RESULT.json", encoding="utf-8"))
    keys = [k for k in res if (iou(res[k]) < lo if below else iou(res[k]) >= lo)]
    keys.sort(key=lambda k: -iou(res[k]))
    css = """
    :root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e2e2dc;--good:#1f8a4c;--mid:#b7791f}
    @media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ececE8;--mut:#9a9a94;--line:#33332f}}
    body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
    main{max-width:1560px;margin:0 auto;padding:20px 16px}
    h1{font-size:20px;margin:0 0 4px} p{color:var(--mut);margin:0 0 16px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
    figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
    figure img{width:100%;height:auto;border-radius:4px;background:#fff}
    figcaption{font-size:12.5px;display:flex;justify-content:space-between;gap:6px;margin-top:4px}
    b.g{color:var(--good)} b.m{color:var(--mid)}
    .key span{display:inline-block;width:14px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
    """
    word = f"IoU &lt; {lo:.2f}" if below else f"IoU ≥ {lo:.2f}"
    out = ["<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>Qwen {word}</title><style>{css}</style></head><body><main>",
           f"<h1>Qwen Q4 — {word} olan {len(keys)} oda (toplam {len(res)})</h1>",
           "<p class='key'><span style='background:#00aa3c'></span>DGN'deki doğru sınır "
           "<span style='background:#d200d2'></span>Qwen'in çizimi (snap sonrası). Kırmızı nokta = oda etiketi. "
           "Yüksekten düşüğe sıralı.</p><div class='grid'>"]
    for k in keys:
        r = res[k]
        v = iou(r)
        cls = "g" if v >= .9 else "m"
        out.append(f"<figure><img alt='{k}' loading='lazy' src='data:image/jpeg;base64,{tile(d, k, r)}'>"
                   f"<figcaption><span>{k} — {r['name']}</span><b class='{cls}'>{v:.3f}</b></figcaption></figure>")
    out.append("</div></main></body></html>")
    dst = R / f"{tag}_{'alt' if below else 'iou'}{int(lo * 100)}.html"
    dst.write_text("".join(out), encoding="utf-8")
    print(dst, len(keys), round(dst.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
