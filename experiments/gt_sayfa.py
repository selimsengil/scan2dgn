"""HTML page of the label audit: every floor with its rooms green (on the line) / red (off), table per project."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, html
from pathlib import Path
import numpy as np
import gt_goster as G
O = G.GD.OUT; D = json.load(open(O / "DENETIM.json", encoding="utf-8"))
rows = []; proj = {}
for fid, F in D.items():
    G.floor_pic(fid, O / "img" / f"{fid}.jpg", width=1500)
    o = np.array([r["oturma"] for r in F["rooms"]]); n = len(o); c = int((o >= 0.97).sum())
    p = proj.setdefault(F["project"], [0, 0]); p[0] += n; p[1] += c; rows.append((fid, F, n, c))
css = "body{font-family:Segoe UI,Arial;margin:16px;background:#fafafa;color:#222}table{border-collapse:collapse;margin-bottom:20px}td,th{border:1px solid #ccc;padding:4px 8px;font-size:14px}.f{margin:18px 0;background:#fff;padding:10px;border:1px solid #ddd}img{max-width:100%}.r{color:#c00}.g{color:#080}"
h = [f"<html><head><meta charset='utf-8'><title>Etiket denetimi</title><style>{css}</style></head><body><h2>DGN odalari taramadaki cizgiye oturuyor mu?</h2>",
     "<p>Yesil: oda siniri cizginin uzerinde (2,5 cm icinde). Kirmizi: cizgiden ayriliyor. Mavi kutu: kapi/pencere (orada cizgi olmamasi normal). Odanin sol ustundeki sayi: cizgi uzerindeki pay. 0,97 ve uzeri temiz sayiliyor.</p>",
     "<table><tr><th>Proje</th><th>Oda</th><th>Temiz</th><th>Pay</th></tr>"]
for k, (n, c) in sorted(proj.items(), key=lambda t: -t[1][1] / t[1][0]): h.append(f"<tr><td>{html.escape(k)}</td><td>{n}</td><td>{c}</td><td>{c / n:.0%}</td></tr>")
h.append("</table>")
for fid, F, n, c in rows:
    bad = sorted([r for r in F["rooms"] if r["oturma"] < 0.97], key=lambda r: r["oturma"])
    h.append(f"<div class='f'><h3>{html.escape(fid)} &mdash; {c}/{n} temiz</h3><p class='r'>" + ", ".join(f"{html.escape(r['name'])} {r['oturma']:.2f}" for r in bad[:20]) + f"</p><a href='img/{fid}.jpg'><img src='img/{fid}.jpg' loading='lazy'></a></div>")
open(O / "ETIKET_DENETIM.html", "w", encoding="utf-8").write("\n".join(h + ["</body></html>"]))
print("ok", sum(p[0] for p in proj.values()), sum(p[1] for p in proj.values()))
