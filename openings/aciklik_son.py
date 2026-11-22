"""Post-processing + measurement of predicted opening boxes.
duzelt(box, areas, s): long sides put on the nearest parallel area face (within 30 px, then 2 px across it, Citymax accepts a
box touching or crossing the faces), ends trimmed where a face ends / a corner falls inside (2.5 px clear), max 3 rounds.
olc(pred, key): match to the labelled openings (centre within the label + 10 cm, along its axis), jamb error (cm),
Citymax validity (aciklik_kural.gecerli) and 'hazir' = matched + valid + jamb error <= 5 cm.
usage: python aciklik_son.py KEY PRED.json [--ham]      (--ham: no post-processing, for comparison)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon, shape
from aciklik_kural import gecerli, kasa_hata_cm, _frame  # noqa: E402
K = config.KT
UC_CM = 2.8               # a3 draws each end ~2.8 cm long (median on training floors Plevne 1, Tekke K1, Ebubekir K1)
ARA, KIRP = 30, 0.3        # face search (px) and the shortest box kept after trimming (share of its length)


def areas_of(key):
    sys.path.insert(0, str(K / "ocr_test")); import test2
    if key == "malkoc_k2":
        E = json.load(open(K / "el_etiket" / "malkoc_k2" / "ETIKET.json", encoding="utf-8"))
        A = [(f'{r["name"]}:{r["id"]}', Polygon(r["poly"]).buffer(0)) for r in E["rooms"]] + [(b["id"], Polygon(b["poly"]).buffer(0)) for b in E["balkonlar"]]
        A.append(("KAT_ALANI", Polygon(E["kat_alani"]).buffer(0))); return A, True
    fid = test2.FLOORS[key]; KA = K / "kat_alani" / f"{fid}.json"
    if KA.exists():                                 # rooms + DGN common areas + balconies + derived floor outline (kat_alani_uret.py)
        D = json.load(open(KA, encoding="utf-8")); L = json.load(open(K / "dgn_katlar" / fid / "LABELS_COMMON.json", encoding="utf-8"))
        A = [(r["name"] + ":" + r["id"], shape(r["poly"]).buffer(0)) for r in D["rooms"]] + [(f"BALKON{i}", shape(b)) for i, b in enumerate(D["balkonlar"])]
        from shapely.ops import unary_union
        RU = unary_union([a[1] for a in A])
        for x in L["structures"]:
            if x["kind"] != "common": continue
            C = shape(x["geometry"]).buffer(0).difference(RU.buffer(2))          # only the part no room covers
            for j, q in enumerate([C] if C.geom_type == "Polygon" else list(getattr(C, "geoms", []))):
                if q.area > 2000: A.append((f"ORTAK:{x['id']}:{j}", q.buffer(1).buffer(-1)))
        A.append(("KAT_ALANI", Polygon(D["kat_alani"]).buffer(0))); return A, True
    V = json.load(open(K / "aciklik_ag" / "veri" / f"{fid}.json", encoding="utf-8"))
    return [(x["name"] + ":" + x["id"], shape(x["poly"]).buffer(0)) for x in V["rooms"]], False      # incomplete: no floor outline


def kisalt(box, s, cm=None):
    """Take the net's systematic overshoot off both ends."""
    cm = UC_CM if cm is None else cm
    if cm <= 0: return box
    c = np.asarray(box, float)[:4]; a, u, n, L, t0, t1 = _frame(c.tolist()); mid = c.mean(0); d = cm / 100 * s
    if L - 2 * d < 0.5 * L: return box
    return [(p - np.sign((p - mid) @ u) * d * u).tolist() for p in c]


def duzelt(box, areas, s):
    box = kisalt(box, s)
    ok0, why0, _ = gecerli(box, areas)
    if ok0: return box, True                       # already valid: leave it as the net drew it
    c2, ok = _duzelt(box, areas, s)
    if ok or len(gecerli(c2, areas)[1]) < len(why0): return c2, ok
    return box, False


def _duzelt(box, areas, s):
    c = np.asarray(box, float)[:4]
    for _ in range(3):
        a, u, n, L, t0, t1 = _frame(c.tolist()); mid = c.mean(0)
        # faces: parallel area edges overlapping the box along the axis, within 30 px of each long side
        offs = []
        for _, P in areas:
            R = np.asarray(P.exterior.coords)
            for p, q in zip(R[:-1], R[1:]):
                d = q - p; l = np.hypot(*d)
                if l < 10 or abs(abs(d @ u) / l - 1) > 0.01: continue
                ta, tb = sorted([(p - a) @ u, (q - a) @ u])
                if min(tb, L) - max(ta, 0) < 0.5 * L: continue
                offs.append(float((p - mid) @ n))
        lo, hi = t0 - float((mid - a) @ n), t1 - float((mid - a) @ n)       # long sides relative to the mid line
        nl = [o for o in offs if abs(o - lo) <= ARA]; nh = [o for o in offs if abs(o - hi) <= ARA and o > (min(nl, key=lambda o: abs(o - lo)) if nl else lo) + 4]
        if nl: lo = min(nl, key=lambda o: abs(o - lo)) - 2
        if nh: hi = max(nh, key=lambda o: -abs(o - hi)) + 2
        p0, p1 = a + float((mid - a) @ u) * 0 * u, None
        c = np.array([mid + u * (-L / 2) + n * lo, mid + u * (L / 2) + n * lo, mid + u * (L / 2) + n * hi, mid + u * (-L / 2) + n * hi])
        ok, why, _ = gecerli(c.tolist(), areas)
        if ok: return c.tolist(), True
        s0, s1 = -L / 2, L / 2; changed = False
        for w in why:
            m = re.search(r"\((-?[\d.]+)\.\.(-?[\d.]+) / ([\d.]+)\)", w)
            if m:
                fa, fb, LL = float(m.group(1)), float(m.group(2)), float(m.group(3))
                if fa > 0: s0 = max(s0, -L / 2 + fa + 2.5); changed = True
                if fb < LL: s1 = min(s1, -L / 2 + fb - 2.5); changed = True
            if "kose" in w:
                nm = w.split(":")[0] + ":" + w.split(":")[1] if w.count(":") > 1 else w.split(":")[0]
                for name, P in areas:
                    if not w.startswith(name): continue
                    for q in np.asarray(P.exterior.coords)[:-1]:
                        if Polygon(c).buffer(-0.5).contains(Point(*q)):
                            tq = float((q - mid) @ u)
                            if tq < 0: s0 = max(s0, tq + 2.5)
                            else: s1 = min(s1, tq - 2.5)
                            changed = True
        # a corner touching the box end: trim that end 2.5 px past the corner (reason gives its t-range from the box start)
        for w in why:
            m2 = re.search(r"kose kutunun ucuna degiyor \(t (-?[\d.]+)\.\.(-?[\d.]+) / ([\d.]+)\)", w)
            if not m2: continue
            ta, tb, LL = float(m2.group(1)), float(m2.group(2)), float(m2.group(3))
            if tb <= LL / 2: s0 = max(s0, -L / 2 + tb + 2.5); changed = True
            elif ta >= LL / 2: s1 = min(s1, -L / 2 + ta - 2.5); changed = True
        if not changed or s1 - s0 < KIRP * L: return c.tolist(), False
        c = np.array([mid + u * s0 + n * lo, mid + u * s1 + n * lo, mid + u * s1 + n * hi, mid + u * s0 + n * hi])
    ok, _, _ = gecerli(c.tolist(), areas); return c.tolist(), ok


def olc(pred, key, ham=False):
    sys.path.insert(0, str(K / "ocr_test")); import test2
    G, s = test2.gt(key); A, full = areas_of(key); bad = []
    if key != "malkoc_k2":
        V = json.load(open(K / "aciklik_ag" / "veri" / f"{test2.FLOORS[key]}.json", encoding="utf-8")); bad = [shape(b["poly"]) for b in V["bad"]]
    used = set(); rows = []
    for p in pred:
        box = p["box"] if ham else duzelt(p["box"], A, s)[0]
        c = Point(*np.mean(np.asarray(box, float), 0)); _, u, _, L, _, _ = _frame(box); best = None
        for g in G:
            if g["id"] in used or not g["poly"].buffer(0.10 * s).contains(c): continue
            _, ug, _, Lg, _, _ = _frame(list(g["poly"].exterior.coords)[:4]) if len(g["poly"].exterior.coords) >= 5 else (None, u, None, L, None, None)
            if abs(u @ ug) < 0.8: continue
            e = kasa_hata_cm(box, list(g["poly"].minimum_rotated_rectangle.exterior.coords)[:4], s)
            if best is None or e < best[1]: best = (g, e)
        v = gecerli(box, A)[0] if full or True else None
        if best: used.add(best[0]["id"])
        onbad = any(b.buffer(0.1 * s).contains(c) for b in bad)
        rows.append({"kind": p["kind"], "eslesen": best[0]["id"] if best else None, "gt_kind": best[0]["kind"] if best else None, "kasa_cm": round(best[1], 1) if best else None,
                     "gecerli": bool(v), "guvenilmez_uzerinde": bool(onbad), "box": box})
    M = [r for r in rows if r["eslesen"]]; e = [r["kasa_cm"] for r in M]
    res = {"kat": key, "tahmin": len(rows), "eslesen": len(M), "etiket": {k: sum(g["kind"] == k for g in G) for k in ("door", "window")},
           "bulunan": {k: sum(r["gt_kind"] == k for r in M) for k in ("door", "window")}, "tur_dogru": sum(r["kind"] == r["gt_kind"] for r in M),
           "yanlis": sum(1 for r in rows if not r["eslesen"] and not r["guvenilmez_uzerinde"]),
           "kasa_med_cm": round(float(np.median(e)), 1) if e else None, "kasa_3cm": int(sum(x <= 3 for x in e)), "kasa_5cm": int(sum(x <= 5 for x in e)),
           "gecerli_eslesen": sum(bool(r["gecerli"]) for r in M), "hazir": int(sum(bool(r["gecerli"]) for r in M)), "hazir_kasa5": int(sum(bool(r["gecerli"]) and r["kasa_cm"] <= 5 for r in M)),
           "not": "" if full else "Kisla: oda seti eksik, kat alani yok -> gecerlilik dis cephe aciklarinda olculemez"}
    return res, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("pred"); ap.add_argument("--ham", action="store_true"); a = ap.parse_args()
    P = json.load(open(a.pred)); P = P["tahmin"] if isinstance(P, dict) else P
    res, rows = olc(P, a.key, a.ham); print(json.dumps(res, ensure_ascii=False))
    json.dump({"sonuc": res, "satirlar": rows}, open(str(a.pred).replace(".json", f"_olcum{'_ham' if a.ham else ''}.json"), "w", encoding="utf-8"), ensure_ascii=False)
