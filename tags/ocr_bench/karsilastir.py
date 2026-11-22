"""Re-score every stored result with the current ortak.match and print one table."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
from ortak import *
rows = []
for f, kind in [("A_rapid.json", "A"), ("A_ppocr.json", "A"), ("A_vl.json", "A"), ("B_rapid.json", "B"), ("B_ppocr.json", "B"), ("B_vl.json", "B"), ("B_glm.json", "B"), ("B_mineru.json", "B")]:
    p = H / f
    if not p.exists(): continue
    old = json.load(open(p, encoding="utf-8"))
    if kind == "B": r = score_crops({x["id"]: x["okunan"] for x in old["satirlar"]}, old["model"], {"sure_s": old.get("sure_s")})
    else:
        src = {"A_rapid.json": H.parent / "el_etiket" / "malkoc_k2" / "ADAY_OCR.json", "A_ppocr.json": H / "A_ppocr_okumalar.json", "A_vl.json": H / "A_vl_okumalar.json"}[f]
        r = score_spot(json.load(open(src, encoding="utf-8")), old["model"]); r["sure_s"] = old.get("sure_s")
    json.dump(r, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); rows.append((kind, r))
print(f"{'olcum':5} {'model':28} {'kapi tam':>9} {'kapi rakam':>10} {'pencere tam':>11} {'pencere rakam':>13} {'sure s':>7}")
for k, r in rows:
    print(f"{k:5} {r['model']:28} {r['door']['tam']:>4}/{r['door']['n']:<4} {r['door']['rakam_dahil']:>5}/{r['door']['n']:<4} {r['window']['tam']:>6}/{r['window']['n']:<4} {r['window']['rakam_dahil']:>8}/{r['window']['n']:<4} {str(r.get('sure_s')):>7}")
