"""One page with every model test we ran: for each room, one tile per run that drew it (DGN green, model magenta,
red dot = the room label), all drawn on the same crop and scored the same way (snapped polygon, broken rings
repaired). Rooms: the 22-room subset where DeepSeek was tested, the 7 thinking-test rooms, the agent rooms.

usage: python all_compare.py  ->  runs/TUM_TESTLER.html
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

import bench132 as B
import qwen_test as QT
import repair_rescore as RR

R = Path(__file__).with_name("runs")
H = type("H", (), {"inside": None})()
RUNS = [  # (label, family, tag)
    ("DeepSeek 500 GB · tek atış", "ds", "ds500_sub22"),
    ("DeepSeek 500 GB · görüntü bias", "ds", "ds500vl_sub22"),
    ("DeepSeek IQ2 · Strata", "ds", "strata_sub22"),
    ("DeepSeek ajan v2 · öğretim + zoom", "ds", "AGENT"),
    ("Qwen Q4 · örneksiz", "qw", "qwenq4_noshot132"),
    ("Qwen Q4 · few-shot", "qw", "qwenq4_fewshot132"),
    ("Qwen Q5 · few-shot", "qw", "qwenq5_sub22"),
    ("Qwen Q4 · öğretim seti", "qw", "qwenq4_teach132"),
    ("Qwen Q4 · öğretim + düşünme", "qw", "qwenq4_teach_think7b"),
    ("Raster2Seq · oda kesiti", "sp", "SPEC:r2s_crop"),
    ("Raster2Seq · geniş pencere", "sp", "SPEC:r2s_ctx"),
    ("Raster2Seq geniş + snap", "sp", "SPEC:r2s_ctx_snap"),
    ("SAM 2.1 · etiket noktası", "sp", "SPEC:sam2_crop"),
    ("SAM 3 · etiket noktası", "sp", "SPEC:sam3_point"),
    ("SAM 3 · nokta + Raster2Seq kutusu", "sp", "SPEC:sam3_r2s"),
    ("SAM 3 · nokta + Qwen kutusu", "sp", "SPEC:sam3_qwen"),
]
SPEC = Path(__file__).with_name("runs") / "spec_inputs"


def _ctx_to_crop(m):
    """The context and crop frames differ by a scale and an offset: fit it on the reference polygon,
    which is the same vertex list in both frames."""
    import numpy as np
    a = np.asarray(m["ctx"]["ref"], float); b = np.asarray(m["crop"]["ref"], float)
    s = (b - b.mean(0)).std() / max((a - a.mean(0)).std(), 1e-9)
    off = b.mean(0) - s * a.mean(0)
    return lambda pts: [[s * x + off[0], s * y + off[1]] for x, y in pts]
TILE = 330


def load(tag):
    if tag.startswith("SPEC:"):
        name = tag[5:]
        f = SPEC / f"{name}.json"
        if not f.exists():
            return {}
        d = json.load(open(f, encoding="utf-8"))
        meta = json.load(open(SPEC / "rooms.json", encoding="utf-8"))
        out = {}
        for k, v in d.items():
            pts = v["pick"]
            if name.endswith("_ctx") and pts:
                pts = _ctx_to_crop(meta[k])(pts)
            out[k] = {"pts": pts, "seconds": 0.5, "size": None}
        return out
    if tag == "AGENT":
        d = json.load(open(R / "ds_agent_b" / "RESULT.json", encoding="utf-8"))["results"]
        return {k: {"pts": v.get("snapped") or v.get("final"), "seconds": v["seconds"], "size": v.get("size")} for k, v in d.items()}
    d = json.load(open(R / f"b132_{tag}" / "RESULT.json", encoding="utf-8"))
    return {k: {"pts": v.get("snapped") or v.get("points"), "seconds": v["seconds"], "size": v.get("size")} for k, v in d.items()}


def score(pts, ref):
    if not pts or len(pts) < 3:
        return {"IoU": 0.0, "corners": 0.0}
    if not Polygon(pts).is_valid:
        pts = RR.repair(pts)
    m = QT.brief(QT.score(pts, ref, H))
    if not m.get("valid", True) or "IoU" not in m:
        return {"IoU": 0.0, "corners": 0.0}
    return {"IoU": m["IoU"], "corners": m["corners@10"]}


def tile(crop, ref, label, pts):
    s = TILE / max(crop.size)
    im = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))))
    d = ImageDraw.Draw(im)
    for p, col, w in ((ref, (0, 170, 60), 3), (pts, (210, 0, 210), 2)):
        if p and len(p) >= 3:
            q = [(x * s, y * s) for x, y in p]
            d.line(q + q[:1], fill=col, width=w)
    r = 4
    d.ellipse((label[0] * s - r, label[1] * s - r, label[0] * s + r, label[1] * s + r), fill=(230, 0, 0))
    b = io.BytesIO(); im.save(b, "JPEG", quality=78)
    return base64.b64encode(b.getvalue()).decode()


def main():
    data = {tag: load(tag) for _, _, tag in RUNS}
    sub22 = sorted(data["ds500_sub22"])
    think = ["013_05", "016_01", "019_21", "001_26", "019_07", "015_07", "015_06"]
    agent = sorted(data["AGENT"])
    want = list(dict.fromkeys(sub22 + think + agent))
    crops = {}
    for key, f, name, crop, ref, label in B.rooms(1400):
        if key in want:
            crops[key] = (name, crop, ref, label)
    sc = {tag: {k: score(v["pts"], crops[k][2]) for k, v in data[tag].items() if k in crops} for _, _, tag in RUNS}

    def mean(tag, keys, f):
        ks = [k for k in keys if k in sc[tag]]
        return (sum(sc[tag][k][f] for k in ks) / len(ks), len(ks)) if ks else (None, 0)

    css = """
    :root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e2e2dc;--ds:#2f6fb5;--qw:#b06a12;--sp:#7a3fb0;--good:#1f8a4c;--bad:#c0392b}
    @media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ececE8;--mut:#9a9a94;--line:#33332f}}
    body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
    main{max-width:1700px;margin:0 auto;padding:20px 16px}
    h1{font-size:20px;margin:0 0 6px} h2.sec{font-size:17px;margin:26px 0 8px} p{color:var(--mut);margin:0 0 12px}
    .tw{overflow-x:auto} table{border-collapse:collapse;background:var(--card);font-size:13px}
    td,th{border:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap} td:first-child,th:first-child{text-align:left}
    .room{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:0 0 12px}
    .room h3{font-size:14.5px;margin:0 0 8px}
    .row{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:9px}
    figure{margin:0} figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
    figcaption{font-size:12px;line-height:1.35} figcaption .l{font-weight:600} .ds .l{color:var(--ds)} .qw .l{color:var(--qw)} .sp .l{color:var(--sp)}
    figcaption .n{color:var(--mut)}
    .key span{display:inline-block;width:14px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
    """
    out = ["<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>Tüm Model Testleri</title><style>{css}</style></head><body><main>",
           "<h1>Tüm model testleri — aynı odalar, yan yana</h1>",
           "<p class='key'><span style='background:#00aa3c'></span>DGN'deki doğru sınır <span style='background:#d200d2'></span>modelin çizimi. "
           "Kırmızı nokta = oda etiketi. Hepsi aynı kesitte ve aynı yöntemle puanlandı. IoU = alan örtüşmesi; köşe = köşelerin "
           "10 px içinde tutma oranı. <b style='color:var(--ds)'>Mavi</b> DeepSeek, <b style='color:var(--qw)'>turuncu</b> Qwen, <b style='color:var(--sp)'>mor</b> uzman modeller (eğitimsiz, sıfır atış).</p>",
           "<div class='tw'><table><tr><th>Çalışma</th><th>22 oda: IoU</th><th>22 oda: köşe</th><th>7 düşünme odası: IoU</th>"
           "<th>7 düşünme odası: köşe</th><th>sn/oda</th></tr>"]
    for lab, fam, tag in RUNS:
        cells = []
        for keys in (sub22, think):
            for f in ("IoU", "corners"):
                m, n = mean(tag, keys, f)
                cells.append("—" if m is None else f"{m:.3f}" + (f" <span style='color:var(--mut)'>({n})</span>" if n != len(keys) else ""))
        secs = [v["seconds"] for k, v in data[tag].items() if k in crops]
        out.append(f"<tr><td style='color:var(--{fam})'>{lab}</td>" + "".join(f"<td>{c}</td>" for c in cells)
                   + f"<td>{sum(secs) / len(secs):.0f}</td></tr>")
    out.append("</table></div>")

    def section(title, note, keys):
        out.append(f"<h2 class='sec'>{title}</h2><p>{note}</p>")
        for k in keys:
            name, crop, ref, label = crops[k]
            out.append(f"<section class='room'><h3>{k} — {name}</h3><div class='row'>")
            for lab, fam, tag in RUNS:
                v = data[tag].get(k)
                if not v:
                    continue
                m = sc[tag][k]
                out.append(f"<figure class='{fam}'><img loading='lazy' alt='{lab} {k}' src='data:image/jpeg;base64,{tile(crop, ref, label, v['pts'])}'>"
                           f"<figcaption><span class='l'>{lab}</span><br><span class='n'>IoU {m['IoU']:.2f} · köşe {m['corners']:.2f} · "
                           f"{v['seconds']:.0f} sn</span></figcaption></figure>")
            out.append("</div></section>")

    section("1. DeepSeek'in denendiği 22 oda", "Tüm modellerin çizdiği ortak oda seti.", sub22)
    section("2. Düşünme testindeki 7 oda (ağırdan basite)", "Qwen'in en bariz hata yaptığı odalar.", think)
    extra = [k for k in agent if k not in sub22 and k not in think]
    if extra:
        section("3. DeepSeek ajan odaları", "Öğretim seti + zoom + köşe kontrolü ile çizilen odalar.", extra)
    out.append("</main></body></html>")
    dst = R / "TUM_TESTLER.html"
    dst.write_text("".join(out), encoding="utf-8")
    print(dst, round(dst.stat().st_size / 1e6, 1), "MB")
    for lab, fam, tag in RUNS:
        print(lab, mean(tag, sub22, "IoU"), mean(tag, sub22, "corners"), mean(tag, think, "IoU"), mean(tag, think, "corners"))


if __name__ == "__main__":
    main()
