"""Merge both harvest passes into one verified tag set per opening.
Sources: dar/<floor>.json (tight level crop of the opening, GLM-OCR + PaddleOCR-VL, 0/180 deg; tag location = inside the box)
         oto/<floor>.json (spotting on box + 60 cm; tag location = the read's 4-point box; assigned greedily, one tag per opening).
Grade per opening: 'iki' = first number within 12 cm of the DGN width AND second number = DGN height (+-2);
                   'genislik' = width only.  Best source wins (iki > genislik; tie: more readers agreeing).
Output ETIKETLER.json {floor: [{id, kind, dgn_cm, dgn_h, sayilar, onek, metin, derece, yer: 'kutu_ici'|'kutu_disi', box, oy}]}"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, re
from collections import Counter
from pathlib import Path
import numpy as np
from shapely.geometry import Point, shape
H = config.KT / "ocr_test"; K = config.KT
PAT = re.compile(r"([A-Z]{1,2}\d{0,2})?[\s\-]*(\d{2,3})[/xX×*1Il|]?(\d{3})")


def parse(t):
    t = t.upper().replace("O", "0").replace(" ", "")
    out = []
    for m in PAT.finditer(t):
        a, b = int(m.group(2)), int(m.group(3))
        if 100 <= b <= 300: out.append((m.group(1) or "", a, b))
    return out


def grade(c, w, h):
    if abs(c[1] - w) > 12: return None
    return "iki" if h is not None and abs(c[2] - h) <= 2 else "genislik"


res = {}; tot = Counter()
for f in sorted((K / "aciklik_ag" / "veri").glob("*.json")):
    fid = f.stem
    if "kisla" in fid.lower() or "malkoc_361_79__k2" in fid.lower(): continue
    L = json.load(open(K / "dgn_katlar" / fid / "LABELS_COMMON.json", encoding="utf-8")); tags = {x["id"]: x.get("tags", {}) for x in L["structures"]}
    V = json.load(open(f, encoding="utf-8")); s = V["px_per_m"]
    dar = json.load(open(H / "dar" / f"{fid}.json", encoding="utf-8")) if (H / "dar" / f"{fid}.json").exists() else {}
    oto = json.load(open(H / "oto" / f"{fid}.json", encoding="utf-8")) if (H / "oto" / f"{fid}.json").exists() else {"okumalar": []}
    # spotting tags (unique)
    sp = []
    for r in oto["okumalar"]:
        for c in parse(r["text"]):
            ctr = np.mean(np.asarray(r["box"], float), 0)
            if not any(np.hypot(*(ctr - q["c"])) < 15 and q["n"] == c for q in sp): sp.append({"c": ctr, "n": c, "text": r["text"], "box": r["box"]})
    rows = []; used_sp = set()
    ops = sorted(V["openings"], key=lambda o: o["id"])
    for o in ops:
        tg = tags.get(o["id"], {}); h = tg.get("Kapı Yüksekliği", tg.get("Pencere Yüksekliği"))
        P = shape(o["poly"]); c4 = np.array(P.minimum_rotated_rectangle.exterior.coords)[:4]
        w = max(np.hypot(*(c4[(k + 1) % 4] - c4[k])) for k in range(4)) / s * 100
        cands = []
        for reader, ts in dar.get(o["id"], {}).get("okumalar", {}).items():
            for t in ts:
                for c in parse(t):
                    gr = grade(c, w, h)
                    if gr: cands.append((gr, c, t, "kutu_ici", o["poly"], reader))
        for i, q in enumerate(sp):
            gr = grade(q["n"], w, h)
            if gr and P.distance(Point(*q["c"])) / s <= 0.8: cands.append((gr, q["n"], q["text"], "kutu_disi" if not P.buffer(3).contains(Point(*q["c"])) else "kutu_ici", q["box"], f"spot#{i}"))
        if not cands:
            rows.append({"id": o["id"], "kind": o["kind"], "dgn_cm": round(w, 1), "dgn_h": h, "derece": None}); tot[o["kind"] + "_yok"] += 1; continue
        votes = Counter((c[1][1], c[1][2]) for c in cands)
        best = max(cands, key=lambda c: (c[0] == "iki", votes[(c[1][1], c[1][2])], c[3] == "kutu_disi"))
        pre = Counter(c[1][0] for c in cands if (c[1][1], c[1][2]) == (best[1][1], best[1][2]) and c[1][0]).most_common(1)
        rows.append({"id": o["id"], "kind": o["kind"], "dgn_cm": round(w, 1), "dgn_h": h, "sayilar": [best[1][1], best[1][2]], "onek": pre[0][0] if pre else "",
                     "metin": best[2], "derece": best[0], "yer": best[3], "box": best[4], "oy": votes[(best[1][1], best[1][2])], "okuyucu": best[5]})
        tot[o["kind"] + "_" + best[0]] += 1
    res[fid] = rows
    n = len(rows); print(f"{fid[:44]:46} {n:3} aciklik | iki {sum(r['derece']=='iki' for r in rows):3}  genislik {sum(r['derece']=='genislik' for r in rows):3}  yok {sum(r['derece'] is None for r in rows):3}")
json.dump(res, open(H / "ETIKETLER.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("TOPLAM", dict(tot))
