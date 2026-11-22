"""Side-by-side page of a few rooms across several bench132 runs, rooms in the given order.

usage: python compare3.py KEYS OUT_NAME TAG:LABEL [TAG:LABEL ...]
  e.g. python compare3.py 013_05,016_01 DUSUNME_7oda.html qwenq4_fewshot132:few-shot qwenq4_teach132:öğretim
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from pathlib import Path

from compare_teach import metrics
from review_sub22 import tile

R = Path(__file__).with_name("runs")


def main():
    keys = sys.argv[1].split(",")
    runs = [(t.split(":", 1)[0], t.split(":", 1)[1]) for t in sys.argv[3:]]
    data = {}
    for tag, _ in runs:
        f = R / f"b132_{tag}" / "RESULT.json"
        data[tag] = json.load(open(f, encoding="utf-8")) if f.exists() else {}
    css = """
    :root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e2e2dc;--good:#1f8a4c;--mid:#b7791f;--bad:#c0392b}
    @media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ececE8;--mut:#9a9a94;--line:#33332f}}
    body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
    main{max-width:1400px;margin:0 auto;padding:20px 16px}
    h1{font-size:20px;margin:0 0 6px} p{color:var(--mut)}
    table{border-collapse:collapse;background:var(--card);margin:0 0 18px;font-size:13px}
    td,th{border:1px solid var(--line);padding:5px 9px;text-align:right} td:first-child,th:first-child{text-align:left}
    .room{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0 0 12px}
    .room h2{font-size:14.5px;margin:0 0 8px}
    .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
    figure{margin:0} figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
    figcaption{font-size:12.5px;color:var(--mut)}
    .key span{display:inline-block;width:14px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
    """
    out = ["<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>Düşünme Testi</title><style>{css}</style></head><body><main>",
           f"<h1>Qwen Q4 — {len(keys)} hatalı oda, ağırdan basite</h1>",
           "<p class='key'><span style='background:#00aa3c'></span>DGN <span style='background:#d200d2'></span>Qwen (snap sonrası). "
           "Köşe = köşelerin 10 px içinde tutma oranı (1.0 = hepsi doğru).</p>"]
    out.append("<table><tr><th>Çalışma</th><th>ort. IoU</th><th>ort. köşe</th><th>ort. sn/oda</th></tr>")
    for tag, lab in runs:
        ks = [k for k in keys if k in data[tag]]
        if not ks:
            continue
        m = [metrics(data[tag][k]) for k in ks]
        sec = sum(data[tag][k]["seconds"] for k in ks) / len(ks)
        out.append(f"<tr><td>{lab} ({len(ks)} oda)</td><td>{sum(x['IoU'] for x in m) / len(m):.3f}</td>"
                   f"<td>{sum(x['corners@10'] for x in m) / len(m):.3f}</td><td>{sec:.0f}</td></tr>")
    out.append("</table>")
    for i, k in enumerate(keys, 1):
        name = next((data[t][k]["name"] for t, _ in runs if k in data[t]), "")
        out.append(f"<section class='room'><h2>{i}. {k} — {name}</h2><div class='row'>")
        for tag, lab in runs:
            r = data[tag].get(k)
            if not r:
                out.append(f"<figure><figcaption>{lab}: yok</figcaption></figure>")
                continue
            m = metrics(r)
            extra = " · zorla cevap" if r.get("forced") else ""
            out.append(f"<figure><img alt='{lab} {k}' src='data:image/jpeg;base64,{tile(R / f'b132_{tag}', k, r)}'>"
                       f"<figcaption>{lab}: IoU {m['IoU']:.2f} · köşe {m['corners@10']:.2f} · {r['seconds']:.0f} sn{extra}</figcaption></figure>")
        out.append("</div></section>")
    out.append("</main></body></html>")
    dst = R / sys.argv[2]
    dst.write_text("".join(out), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
