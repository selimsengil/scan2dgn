"""HTML page comparing every model's unaided drawing of floor 015 with the office DGN, room by room, plus tokens.
Labels: KAT_ROOT=kat_test/kor_015/floors (015 rebuilt from the Kışla 83 DGN snapshot of 2026-11-15).
Output: kat_test/015_karsilastirma/KARSILASTIRMA_015.html + img/
usage: KAT_ROOT=... python kat_sayfa015.py"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import html, json, os, re
from pathlib import Path

import cv2
import numpy as np

import kat_skor2 as K2
import token_say as TS
import v0_watershed as V0

T = config.KT; O = T / "015_karsilastirma"; (O / "img").mkdir(parents=True, exist_ok=True)
NAMES = {"015_sol_uzman": "Sol (uzman yönergeli)", "015_sol_sifir": "Sol (sıfırdan, yüksek)", "015_sol_sifir_low": "Sol (sıfırdan, düşük)",
         "015_sol_sifir_terra_medium": "GPT 6.1 Terra (orta)", "015_gemini4neon_high": "Gemini 4 Neon High", "015_gemini4argon": "Gemini 4 Argon High",
         "015_gemini38flash": "Gemini 3.8 Flash", "015_sonnet_medium": "Claude Sonnet 5.5 (orta)", "015_haiku": "Claude Haiku 5.5",
         "015_deepseek_agent": "DeepSeek (API, ajan)", "015_deepseek_araclar_agent": "DeepSeek (API, araçlı ajan)", "015_deepseek_single": "DeepSeek (API, tek istek)",
         "015_deepseek_harness": "DeepSeek (yönergeli)", "015_deepseek_harness_mini": "DeepSeek (yönergeli, mini)", "015_deepseek_harness_mini_nomark": "DeepSeek (yönergeli, mini, işaretsiz)",
         "015_qwen38max_single": "Qwen 3.8 Max (API, tek istek)", "015_qwen38max_agent": "Qwen 3.8 Max (API, ajan)", "015_qwen_staged": "Qwen 27B yerel (aşamalı)",
         "015_destekli_neon_high": "Gemini 4 Neon High + çizgi ağı", "015_destekli_sonnet_medium": "Claude Sonnet 5.5 (orta) + çizgi ağı"}
GT_C, PR_C = (0, 140, 255), (210, 70, 0)     # BGR: office DGN orange, model blue


def minutes(d):
    p = d / "DONE.txt"
    if not p.exists(): return None
    m = re.search(r"minutes=([\d.]+)", p.read_text(encoding="utf-8-sig", errors="replace")); return float(m.group(1)) if m else None


def draw(base, gt, pr, box=None):
    v = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR); v = (v * 0.45 + 140).astype(np.uint8)
    for g in gt: cv2.polylines(v, [np.round(np.asarray(g.exterior.coords)).astype(np.int32)], True, GT_C, 6 if box is None else 5)
    for p in pr: cv2.polylines(v, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], True, PR_C, 2)
    if box: x0, y0, x1, y1 = box; v = v[max(0, y0):y1, max(0, x0):x1]
    return v


def col(x): return "#1a7f37" if x >= 0.9 else ("#b58500" if x >= 0.7 else "#c62828")


if __name__ == "__main__":
    base = cv2.imread(str(T / "015_sol_sifir" / "kat_015.png"), 0); gts = [g for g in V0.load_rooms("015") if g["poly"].geom_type == "Polygon"]
    runs = []
    for d in sorted(T.glob("015_*")):
        if not (d / "SONUC_015.json").exists() or d.name == "015_karsilastirma" or not (d / "DONE.txt").exists() and "destekli" in d.name: continue
        S = K2.score(d, "015"); R = json.load(open(d / "SKOR2.json", encoding="utf-8"))["rooms"]
        ans = K2.load_answer(d / "SONUC_015.json")
        preds = [K2.SS.clean(r["boundary"]) for r in ans.get("rooms", []) if r.get("type") != "balcony" and r.get("boundary") and len(r["boundary"]) >= 3]
        preds = [p for p in preds if p is not None]
        runs.append({"dir": d, "name": NAMES.get(d.name, d.name), "S": S, "R": R, "preds": preds, "tok": TS.tokens(d), "min": minutes(d)})
    runs.sort(key=lambda r: -r["S"]["cizgi"])
    # matched polygon per GT room (same Hungarian match as the scorer)
    from scipy.optimize import linear_sum_assignment
    for r in runs:
        M = np.array([[g["poly"].intersection(p).area / max(g["poly"].union(p).area, 1e-9) for p in r["preds"]] for g in gts]) if r["preds"] else np.zeros((len(gts), 1))
        gi, pj = linear_sum_assignment(-M) if r["preds"] else ([], []); r["match"] = {i: r["preds"][j] for i, j in zip(gi, pj) if M[i, j] > 0.3}
        r["extra"] = len(r["preds"]) - len(r["match"])
        slug = r["dir"].name
        cv2.imwrite(str(O / "img" / f"{slug}_kat.jpg"), cv2.resize(draw(base, [g["poly"] for g in gts], r["preds"]), None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
        for i, g in enumerate(gts):
            x0, y0, x1, y1 = [int(v) for v in g["poly"].bounds]; p = 45; t = draw(base, [g["poly"]], [r["match"][i]] if i in r["match"] else [], (x0 - p, y0 - p, x1 + p, y1 + p))
            s = 300 / max(t.shape[:2]); cv2.imwrite(str(O / "img" / f"{slug}_r{i}.jpg"), cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    fmt = lambda n: "—" if n is None else (f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.0f} k")
    css = """body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f6f6f4;color:#1d1d1d}main{max-width:1500px;margin:auto;padding:16px}
h1{font-size:22px}h2{font-size:18px;margin-top:34px;border-top:2px solid #ddd;padding-top:14px}table{border-collapse:collapse;background:#fff;font-size:13px}
td,th{border:1px solid #d6d6d6;padding:4px 7px;text-align:right}th{background:#eee}td.l,th.l{text-align:left}.lg span{display:inline-block;width:28px;height:5px;margin:0 6px 2px 14px;vertical-align:middle}
.tiles{display:flex;flex-wrap:wrap;gap:8px}.t{background:#fff;border:1px solid #ddd;padding:4px;width:300px;font-size:12px}.t img{display:block;max-width:300px}.bad{opacity:.85;border-color:#c62828}
.wrap{overflow-x:auto}.note{font-size:13px;color:#555;max-width:1100px}img.kat{max-width:100%;border:1px solid #ccc}"""
    h = [f"<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>015 model karşılaştırması</title><style>{css}</style></head><body><main>",
         "<h1>Kat 015 (Kışla 83, 1. kat): modellerin desteksiz çizimi ve senin DGN'in</h1>",
         "<p class='lg'>Çizgiler:<span style='background:#ff8c00'></span>senin DGN sınırın (kalın turuncu)<span style='background:#0046d2'></span>modelin çizdiği (ince mavi)</p>",
         "<p class='note'><b>Çizgi</b>: senin sınırının ne kadarının 3 piksel (~3 cm) içinde modelin çizgisiyle örtüştüğü. <b>Fazla</b>: modelin çizgisinin senin sınırından uzak kalan payı. "
         "<b>Kabul</b>: düzeltmesiz kabul edilebilecek oda (çizgi ≥ 0,97, fazla ≤ 0,03, bütün köşeler tutuyor). <b>Alan</b>: alan örtüşmesi (eski ölçü). "
         "Etiket: Kışla 83 DGN'inin 15 Kasım hâli, 21 oda (balkonlar hariç). Token: modelin okuduğu + önbellekten okuduğu + yazdığı (düşünme dahil).</p>",
         "<div class='wrap'><table><tr><th class='l'>Model</th><th>Çizgi</th><th>Fazla</th><th>Kabul</th><th>Çizgi ≥ 0,90</th><th>Alan</th><th>Bulunan oda</th><th>Fazladan poligon</th><th>Köşe</th><th>Süre (dk)</th><th>Yeni girdi</th><th>Önbellek</th><th>Çıktı</th><th>Toplam token</th><th>Not</th></tr>"]
    for r in runs:
        S, t = r["S"], r["tok"] or {}; note = ""
        if r["dir"].name == "015_sol_uzman": note = "Token sayısı Sol'un önceki uzun sohbetini de içeriyor (çatallanmış oturum); bu test için gerçek değil."
        if t.get("maliyet_usd") is not None: note = f"Liste fiyatıyla ~${t['maliyet_usd']}"
        h.append(f"<tr><td class='l'><a href='#{r['dir'].name}'>{html.escape(r['name'])}</a></td><td style='color:{col(S['cizgi'])}'><b>{S['cizgi']:.3f}</b></td><td>{S['fazla']:.3f}</td><td>{S['kabul']}</td><td>{S['cizgi>=0.90']}</td>"
                 f"<td>{S['iou']:.3f}</td><td>{S['found']}/{S['rooms']}</td><td>{r['extra']}</td><td>{S['kose']}</td><td>{'' if r['min'] is None else r['min']}</td>"
                 f"<td>{fmt(t.get('giris'))}</td><td>{fmt(t.get('onbellek'))}</td><td>{fmt(t.get('cikti'))}</td><td><b>{fmt(t.get('toplam'))}</b></td><td class='l'>{html.escape(note)}</td></tr>")
    h.append("</table></div><h2>Oda oda çizgi skoru</h2><div class='wrap'><table><tr><th class='l'>Oda</th>" + "".join(f"<th>{html.escape(r['name'])}</th>" for r in runs) + "</tr>")
    for i, g in enumerate(gts):
        h.append(f"<tr><td class='l'>{i + 1}. {html.escape(g['name'] or '')}</td>" + "".join(
            (f"<td style='color:{col(r['R'][i]['cizgi'])}'>{r['R'][i]['cizgi']:.2f}</td>" if i in r["match"] else "<td style='color:#c62828'>yok</td>") for r in runs) + "</tr>")
    h.append("</table></div>")
    for r in runs:
        S = r["S"]; slug = r["dir"].name
        h.append(f"<h2 id='{slug}'>{html.escape(r['name'])}: çizgi {S['cizgi']:.3f}, {S['kabul']} oda düzeltmesiz, {S['found']}/{S['rooms']} oda bulundu, {r['extra']} fazladan poligon</h2>"
                 f"<a href='img/{slug}_kat.jpg'><img class='kat' src='img/{slug}_kat.jpg' loading='lazy'></a><div class='tiles'>")
        for i, g in enumerate(gts):
            rr = r["R"][i]; found = i in r["match"]
            cap = (f"<b>{i + 1}. {html.escape(g['name'] or '')}</b> — çizgi <b style='color:{col(rr['cizgi'])}'>{rr['cizgi']:.2f}</b>, fazla {rr['fazla']:.2f}, köşe {rr['kose']}/{rr['kose_gt']}, alan {rr['iou']:.2f}"
                   + (" ✔ düzeltmesiz" if rr["kabul"] else "")) if found else f"<b>{i + 1}. {html.escape(g['name'] or '')}</b> — <b style='color:#c62828'>model bu odayı çizmedi</b>"
            h.append(f"<div class='t{'' if found else ' bad'}'><img src='img/{slug}_r{i}.jpg' loading='lazy'>{cap}</div>")
        h.append("</div>")
    (O / "KARSILASTIRMA_015.html").write_text("\n".join(h + ["</main></body></html>"]), encoding="utf-8")
    print("ok", len(runs), "runs", O / "KARSILASTIRMA_015.html")
