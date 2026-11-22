"""Teaching set vs plain few-shot on the 132 rooms: area IoU, boundary F1@5 and corner F1@10 (snapped),
overall and by room type; plus a side-by-side HTML (rooms sorted by corner gain).

The old run's broken polygons are repaired the same way as the new run before scoring.
Rooms used in teaching sheet C (013_04, 001_01) are left out.

usage: python compare_teach.py OLD_TAG NEW_TAG  ->  runs/KARSILASTIRMA_ogretim.html
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from collections import defaultdict
from pathlib import Path

from shapely.geometry import Polygon

import qwen_test as QT
import repair_rescore as RR
import snap
from compare_runs import kind
from review_sub22 import tile

R = Path(__file__).with_name("runs")
LEAK = {"013_04", "001_01"}
H = type("H", (), {"inside": None})()


def metrics(r):
    m = r["snap"]
    pts = r.get("snapped") or []
    if pts and len(pts) >= 3 and not Polygon(pts).is_valid:
        m = QT.brief(QT.score(RR.repair(pts), r["reference"], H))
    if not m.get("valid", True) or "IoU" not in m:
        return {"IoU": 0.0, "F1@5": 0.0, "corners@10": 0.0}
    return {k: m[k] for k in ("IoU", "F1@5", "corners@10")}


def main():
    old_t, new_t = sys.argv[1], sys.argv[2]
    old = json.load(open(R / f"b132_{old_t}" / "RESULT.json", encoding="utf-8"))
    new = json.load(open(R / f"b132_{new_t}" / "RESULT.json", encoding="utf-8"))
    keys = sorted(k for k in set(old) & set(new) if k not in LEAK)
    mo = {k: metrics(old[k]) for k in keys}
    mn = {k: metrics(new[k]) for k in keys}
    cols = ("IoU", "F1@5", "corners@10")
    avg = lambda m, ks, c: sum(m[k][c] for k in ks) / len(ks)
    lines = [f"{len(keys)} rooms (teaching rooms excluded)",
             f"{'':12}" + "".join(f"{c:>14}" for c in cols) + f"{'IoU>=.90':>10}{'crn>=.8':>9}{'s/oda':>7}"]
    for lab, m, res in (("few-shot", mo, old), ("öğretim", mn, new)):
        lines.append(f"{lab:12}" + "".join(f"{avg(m, keys, c):>14.3f}" for c in cols)
                     + f"{sum(m[k]['IoU'] >= .9 for k in keys):>10}{sum(m[k]['corners@10'] >= .8 for k in keys):>9}"
                     + f"{sum(res[k]['seconds'] for k in keys) / len(keys):>7.1f}")
    g = defaultdict(list)
    for k in keys:
        g[kind(new[k]["name"])].append(k)
    lines.append(f"\n{'tip':10}{'n':>4}{'IoU eski':>10}{'IoU yeni':>10}{'köşe eski':>11}{'köşe yeni':>11}")
    for t, ks in sorted(g.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"{t:10}{len(ks):>4}{avg(mo, ks, 'IoU'):>10.3f}{avg(mn, ks, 'IoU'):>10.3f}"
                     f"{avg(mo, ks, 'corners@10'):>11.3f}{avg(mn, ks, 'corners@10'):>11.3f}")
    better = sum(mn[k]["corners@10"] > mo[k]["corners@10"] + .05 for k in keys)
    worse = sum(mn[k]["corners@10"] < mo[k]["corners@10"] - .05 for k in keys)
    lines.append(f"\nköşe skoru: {better} odada iyileşti, {worse} odada kötüleşti, {len(keys) - better - worse} aynı")
    txt = "\n".join(lines)
    print(txt)

    css = """
    :root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e2e2dc;--good:#1f8a4c;--mid:#b7791f;--bad:#c0392b}
    @media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ececE8;--mut:#9a9a94;--line:#33332f}}
    body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
    main{max-width:1200px;margin:0 auto;padding:20px 16px}
    h1{font-size:20px;margin:0 0 6px} p{color:var(--mut)} pre{background:var(--card);border:1px solid var(--line);padding:10px;overflow-x:auto;font-size:12.5px}
    .room{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0 0 12px}
    .room h2{font-size:14.5px;margin:0 0 8px}
    .pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    @media (max-width:640px){.pair{grid-template-columns:1fr}}
    figure{margin:0} figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
    figcaption{font-size:12.5px;color:var(--mut)} b.g{color:var(--good)} b.b{color:var(--bad)}
    .key span{display:inline-block;width:14px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
    """
    order = sorted(keys, key=lambda k: -(mn[k]["corners@10"] - mo[k]["corners@10"]))
    out = ["<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>Öğretim Seti Karşılaştırması</title><style>{css}</style></head><body><main>",
           "<h1>Qwen Q4: few-shot ve öğretim seti</h1>",
           "<p class='key'><span style='background:#00aa3c'></span>DGN <span style='background:#d200d2'></span>Qwen (snap sonrası). "
           "Köşe = corners@10 (köşelerin 10 px içinde tutma oranı, 1.0 = hepsi doğru). Sıra: öğretim setinin köşe skorunu en çok "
           "artırdığı odadan en çok düşürdüğüne.</p>", f"<pre>{txt}</pre>"]
    for k in order:
        d = mn[k]["corners@10"] - mo[k]["corners@10"]
        cls = "g" if d > .05 else "b" if d < -.05 else ""
        out.append(f"<section class='room'><h2>{k} — {new[k]['name']} <b class='{cls}'>köşe {d:+.2f}</b></h2><div class='pair'>")
        for lab, res, m, t in (("few-shot", old, mo, old_t), ("öğretim seti", new, mn, new_t)):
            out.append(f"<figure><img loading='lazy' alt='{lab} {k}' src='data:image/jpeg;base64,{tile(R / f'b132_{t}', k, res[k])}'>"
                       f"<figcaption>{lab}: IoU {m[k]['IoU']:.2f} · köşe {m[k]['corners@10']:.2f} · F1@5 {m[k]['F1@5']:.2f}</figcaption></figure>")
        out.append("</div></section>")
    out.append("</main></body></html>")
    dst = R / "KARSILASTIRMA_ogretim.html"
    dst.write_text("".join(out), encoding="utf-8")
    print(dst, round(dst.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
