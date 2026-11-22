"""Common test for all tag finders/readers: Malkoc K2 (hand label, 40 tags) and Kisla 85/1 (DGN width/height of the
reliable openings).  A candidate gives, per floor, reads = [{box: 4 pts (sheet px), text}] from the WHOLE sheet
(the openings are not known to it).  An opening is found when a read 0.6 m around it has a number pair whose first
number matches the width (Malkoc: = tag; Kisla: within 12 cm of the DGN width) and whose second = the height (+-2).
Noise = number-pair reads farther than 0.6 m from every opening.
python test2.py           -> table of all SONUC2/*.json"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, re
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon, shape
H = config.KT / "ocr_test"; K = config.KT; OUT = H / "SONUC2"; OUT.mkdir(exist_ok=True)
FLOORS = {"malkoc_k2": "Gonen_Malkoc_361_79__K2_56155", "kisla_85_1": "Susurluk_Kisla_143_85__1_54883", "kisla_84_1": "Susurluk_Kisla_143_84__1_55569",
          "kisla_85_z": "Susurluk_Kisla_143_85__Z_55052", "kisla_84_z": "Susurluk_Kisla_143_84__Z_55390",
          "plevne_1": "Gonen_Plevne_228_28__1_55699", "tekke_k1": "Havran_Tekke_384_5__K1_55650", "ebubekir_k1": "Havran_Ebubekir_271_172__K1_55736"}   # training floors (bias only)
PAT = re.compile(r"(\d{2,3})\s*[/xX×*1Il|\\]?\s*(\d{3})")


def pairs(t):
    t = t.upper().replace("O", "0").replace("Z", "2").replace(" ", "")
    return [(int(a), int(b)) for a, b in PAT.findall(t) if 30 <= int(a) <= 400 and 100 <= int(b) <= 300]


def gt(key):
    fid = FLOORS[key]
    if key == "malkoc_k2":
        E = json.load(open(K / "el_etiket" / "malkoc_k2" / "ETIKET.json", encoding="utf-8")); s = E["px_per_m"]; out = []
        for o in E["openings"]:
            a, b = [int(x) for x in re.findall(r"\d+", o["etiket"].replace("P1", ""))[:2]]
            out.append({"id": o["id"], "kind": o["kind"], "poly": Polygon(o["poly"]), "w": a, "h": b, "exact": True})
        return out, s
    L = json.load(open(K / "dgn_katlar" / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
    tags = {x["id"]: x.get("tags", {}) for x in L["structures"]}; V = json.load(open(K / "aciklik_ag" / "veri" / f"{fid}.json", encoding="utf-8")); out = []
    for o in V["openings"]:
        P = shape(o["poly"]); c = np.array(P.minimum_rotated_rectangle.exterior.coords)[:4]; w = max(np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)) / s * 100
        tg = tags.get(o["id"], {}); h = tg.get("Kapı Yüksekliği", tg.get("Pencere Yüksekliği"))
        out.append({"id": o["id"], "kind": o["kind"], "poly": P, "w": w, "h": h, "exact": False})
    return out, s


def score(key, reads):
    G, s = gt(key); found = {g["id"]: None for g in G}; noise = 0; npairs = 0; yer = set(); gen = set()
    for r in reads:
        c0 = Point(*np.mean(np.asarray(r["box"], float), 0))
        for g in G:
            if g["poly"].distance(c0) / s <= 0.6:
                yer.add(g["id"])
                for a0, b0 in pairs(r["text"]) + [(int(x), 0) for x in re.findall(r"^(\d{2,3})/\?$", r["text"])]:
                    if (a0 == g["w"]) if g["exact"] else abs(a0 - g["w"]) <= 12: gen.add(g["id"])
    for r in reads:
        ps = pairs(r["text"])
        if not ps: continue
        npairs += 1; c = Point(*np.mean(np.asarray(r["box"], float), 0)); near = False
        for g in G:
            if g["poly"].distance(c) / s > 0.6: continue
            near = True
            for a, b in ps:
                okw = (a == g["w"]) if g["exact"] else abs(a - g["w"]) <= 12
                if okw and g["h"] is not None and abs(b - g["h"]) <= 2: found[g["id"]] = r["text"]
        noise += not near
    res = {"n": {k: sum(g["kind"] == k for g in G) for k in ("door", "window")},
           "bulunan": {k: sum(found[g["id"]] is not None and g["kind"] == k for g in G) for k in ("door", "window")}, "sayi_cifti": npairs, "gurultu": noise,
           "kacan": [g["id"] for g in G if found[g["id"]] is None],
           "yer": {k: sum(g["id"] in yer and g["kind"] == k for g in G) for k in ("door", "window")},
           "genislik": {k: sum(g["id"] in gen and g["kind"] == k for g in G) for k in ("door", "window")}}
    return res


def save(name, key, reads, seconds, extra=None):
    r = score(key, reads); r.update({"aday": name, "kat": key, "sure_s": round(seconds, 1), **(extra or {})})
    json.dump({"sonuc": r, "okumalar": reads}, open(OUT / f"{name}__{key}.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(name, key, "yer", r["yer"], "genislik", r["genislik"], "tam", r["bulunan"], "/", r["n"], "gurultu", r["gurultu"], f"{seconds:.0f}s", flush=True); return r


if __name__ == "__main__":
    rows = [json.load(open(p, encoding="utf-8"))["sonuc"] for p in sorted(OUT.glob("*.json"))]
    print(f"{'aday':34} {'kat':11} {'kapi':>9} {'pencere':>9} {'gurultu':>8} {'sure':>7}")
    for r in sorted(rows, key=lambda r: (r["kat"], -(r["bulunan"]["door"] + r["bulunan"]["window"]))):
        print(f"{r['aday']:34} {r['kat']:11} {r['bulunan']['door']:>4}/{r['n']['door']:<4} {r['bulunan']['window']:>4}/{r['n']['window']:<4} {r['gurultu']:>8} {r['sure_s']:>6.0f}s")
