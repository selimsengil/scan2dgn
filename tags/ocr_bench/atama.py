"""Assign the auto-read tags to openings, floor by floor, and grade them.
Tag = any spotting read containing number+number (PAT of oto_etiket).  A tag may go to one opening only, an opening gets
one tag: greedy by distance (tag centre to opening outline) among pairs with |first - DGN width| <= 12 cm and distance <= 0.8 m.
Grade: 'iki' = second number equals the DGN height too (Kapi/Pencere Yuksekligi), 'genislik' = only the width agrees.
Output ATAMA.json (all floors) and a summary."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, re
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon, shape
H = config.KT / "ocr_test"; K = config.KT; PAT = re.compile(r"(\d{2,3})[/xX×*1Il|]?(\d{3})")
res = {}; tot = {}
for f in sorted((H / "oto").glob("*.json")):
    d = json.load(open(f, encoding="utf-8")); fid = d["floor"]; s = d["px_per_m"]
    L = json.load(open(K / "dgn_katlar" / fid / "LABELS_COMMON.json", encoding="utf-8")); tags = {x["id"]: x.get("tags", {}) for x in L["structures"]}
    V = json.load(open(K / "aciklik_ag" / "veri" / f"{fid}.json", encoding="utf-8")); ops = {o["id"]: shape(o["poly"]) for o in V["openings"]}
    wid = {}
    for o in d["openings"]: wid[o["id"]] = o["dgn_cm"]
    # unique tag reads
    T = []
    for r in d["okumalar"]:
        t = r["text"].replace(" ", ""); m = [(int(a), int(b)) for a, b in PAT.findall(t) if 100 <= int(b) <= 260]
        if not m: continue
        c = np.mean(np.asarray(r["box"], float), 0)
        if any(np.hypot(*(c - q["c"])) < 15 and q["n"] == m[0] for q in T): continue
        T.append({"c": c, "n": m[0], "text": r["text"], "box": r["box"]})
    pairs = []
    for ti, q in enumerate(T):
        for oid, P in ops.items():
            if oid not in wid or abs(q["n"][0] - wid[oid]) > 12: continue
            dist = P.distance(Point(*q["c"])) / s
            if dist <= 0.8: pairs.append((dist + abs(q["n"][0] - wid[oid]) / 100, ti, oid, dist))
    pairs.sort(); ut, uo = set(), set(); rows = []
    for _, ti, oid, dist in pairs:
        if ti in ut or oid in uo: continue
        ut.add(ti); uo.add(oid); q = T[ti]; tg = tags.get(oid, {}); kind = "door" if "Kapı Yüksekliği" in tg else "window"
        hgt = tg.get("Kapı Yüksekliği", tg.get("Pencere Yüksekliği"))
        rows.append({"id": oid, "kind": kind, "dgn_genislik": wid[oid], "dgn_yukseklik": hgt, "metin": q["text"], "sayilar": list(q["n"]), "box": q["box"],
                     "uzaklik_m": round(dist, 2), "derece": "iki" if hgt is not None and abs(q["n"][1] - hgt) <= 2 else "genislik"})
    res[fid] = rows
    for k in ("door", "window"):
        t = tot.setdefault(k, {"aciklik": 0, "iki": 0, "genislik": 0})
        t["aciklik"] += sum(1 for o in d["openings"] if o["kind"] == k); t["iki"] += sum(r["kind"] == k and r["derece"] == "iki" for r in rows); t["genislik"] += sum(r["kind"] == k and r["derece"] == "genislik" for r in rows)
    print(f"{fid[:44]:46} tag {len(T):3} atanan {len(rows):3} iki {sum(r['derece']=='iki' for r in rows):3}")
json.dump(res, open(H / "ATAMA.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0); print("TOPLAM", json.dumps(tot))
