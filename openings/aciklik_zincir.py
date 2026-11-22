"""The whole opening chain on one floor:
  1. opening net (a3, no tag input) -> boxes
  2. aciklik_son.duzelt: faces + corner trimming;  Citymax rule check -> drawn (valid) or flagged (invalid)
  3. numbers: the nearest tag read (ocr_test/SONUC2: image matching 'esle' + RapidOCR/docTR/Florence-2) within 0.6 m whose
     first number fits the box width (+-15 cm) gives width/height; none -> width from the box, height '?' (flagged)
Output kat_test/zincir/<KEY>.json (for the engine) + a score: found / valid / height right (Malkoc: doors vs the hand label,
Kisla: vs the DGN heights).   usage: python aciklik_zincir.py KEY [--net a3]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon
import aciklik_son as S  # noqa: E402
import test2  # noqa: E402
from aciklik_kural import gecerli, _frame  # noqa: E402
K = config.KT; OUT = K / "zincir"; OUT.mkdir(exist_ok=True)


def tag_reads(key):
    out = []
    for name in ("esle", "rapidocr_4yon", "doctr_db50_parseq", "florence2_large"):
        p = K / "ocr_test" / "SONUC2" / f"{name}__{key}.json"
        if not p.exists(): continue
        for r in json.load(open(p, encoding="utf-8"))["okumalar"]:
            ps = [q for q in test2.pairs(r["text"]) if 40 <= q[0] <= 300 and 100 <= q[1] <= 300]
            if ps: out.append({"c": np.mean(np.asarray(r["box"], float), 0), "pair": ps[0], "src": name})
    return out


def zincir(key, net="a3", ipucu=False, ad=None):
    G, s = test2.gt(key); A, _ = S.areas_of(key)
    ad = ad or f"{net}{'' if ipucu else '_etiketsiz'}"
    P = json.load(open(K / "aciklik_ag" / net / f"tahmin_{ad}__{key}.json"))["tahmin"]; T = tag_reads(key); res = []
    uc = K / "aciklik_ag" / net / f"UC_{ad}.json"
    if uc.exists(): S.UC_CM = json.load(open(uc))["UC_CM"]          # end correction measured on the training floors for this net
    from collections import Counter
    for p in P:
        box, ok = S.duzelt(p["box"], A, s); _, u, _, L, _, _ = _frame(box); wcm = L / s * 100; B = Polygon(box)
        hmin = 170 if p["kind"] == "door" else 60                      # a door is not 100 high: a misread
        votes = Counter(); near = {}
        for t in T:
            d = B.distance(Point(*t["c"])) / s
            if d <= 0.6 and abs(t["pair"][0] - wcm) <= 15 and t["pair"][1] >= hmin:
                votes[t["pair"]] += 1; near[t["pair"]] = min(near.get(t["pair"], 9), d)
        if votes:
            pair = max(votes, key=lambda q: (votes[q], -near[q])); src = f"okuma ({votes[pair]} oy)"
            res.append({"tur": p["kind"], "box": box, "gecerli": bool(ok), "genislik_cm": pair[0], "yukseklik_cm": pair[1], "oy": votes[pair], "kutu_cm": round(wcm, 1),
                        "durum": "ciz" if ok else "kontrol", "yukseklik_kaynak": src})
        else:
            res.append({"tur": p["kind"], "box": box, "gecerli": bool(ok), "genislik_cm": int(round(wcm / 10) * 10), "yukseklik_cm": None, "oy": 0, "kutu_cm": round(wcm, 1),
                        "durum": "ciz" if ok else "kontrol", "yukseklik_kaynak": "yok"})
    # floor majority per (type, width class): a single odd read is replaced, an unread opening gets it (both flagged as inference)
    def cls(r): return (r["tur"], int(round(r["genislik_cm"] / 10.0)))
    maj = {}
    for r in res:
        if r["yukseklik_cm"] is not None: maj.setdefault(cls(r), Counter())[r["yukseklik_cm"]] += r["oy"]
    tmaj = {}
    for r in res:
        if r["yukseklik_cm"] is not None: tmaj.setdefault(r["tur"], Counter())[r["yukseklik_cm"]] += r["oy"]
    for r in res:
        k0 = cls(r); m = Counter()
        for dk in (-1, 0, 1): m.update(maj.get((k0[0], k0[1] + dk), Counter()))       # same type, width within ~10 cm
        if not m: continue
        top, n = m.most_common(1)[0]
        if r["yukseklik_cm"] is None: r["yukseklik_cm"] = top; r["yukseklik_kaynak"] = "cikarim: ayni tur/genislik kat cogunlugu"
        elif r["yukseklik_cm"] % 10 != 0 and r["oy"] == 1 and top % 10 == 0:            # '201', '212': a misread digit
            r["yukseklik_kaynak"] = f"cikarim: okuma {r['yukseklik_cm']} -> kat cogunlugu"; r["yukseklik_cm"] = top
    # score
    bad = []
    if key != "malkoc_k2":
        from shapely.geometry import shape
        V = json.load(open(K / "aciklik_ag" / "veri" / f"{test2.FLOORS[key]}.json", encoding="utf-8")); bad = [shape(b["poly"]).buffer(0.1 * s) for b in V["bad"]]
    used = set(); sc = {"aciklik": len(G), "cizilen": sum(r["durum"] == "ciz" for r in res), "bulunan_gecerli": 0, "yanlis_cizilen": 0, "yukseklik_dogru": 0, "yukseklik_dogru_cikarim": 0, "yukseklik_yanlis": 0, "yukseklik_yanlis_cikarim": 0, "yukseklik_yok": 0}
    for r in res:
        c = Point(*np.mean(np.asarray(r["box"], float), 0)); g = next((g for g in G if g["id"] not in used and g["poly"].buffer(0.10 * s).contains(c)), None)
        if r["durum"] != "ciz": continue
        if g is None:
            if not any(b.contains(c) for b in bad): sc["yanlis_cizilen"] += 1
            continue
        used.add(g["id"]); sc["bulunan_gecerli"] += 1
        cik = r["yukseklik_kaynak"].startswith("cikarim")
        if r["yukseklik_cm"] is None: sc["yukseklik_yok"] += 1
        elif g["h"] is not None and abs(r["yukseklik_cm"] - g["h"]) <= 2: sc["yukseklik_dogru_cikarim" if cik else "yukseklik_dogru"] += 1
        else: sc["yukseklik_yanlis_cikarim" if cik else "yukseklik_yanlis"] += 1
    json.dump({"kat": key, "ag": net, "ipucu": ipucu, "sonuc": sc, "acikliklar": res}, open(OUT / f"{key}__{ad}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return sc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("--net", default="a3"); ap.add_argument("--ipucu", action="store_true"); ap.add_argument("--ad", default=None); a = ap.parse_args()
    print(a.key, a.ad or a.net, zincir(a.key, a.net, a.ipucu, a.ad))
