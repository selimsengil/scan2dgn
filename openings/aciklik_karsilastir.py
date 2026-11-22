"""Compare opening-net variants on the 5 test floors; the end-overshoot correction (aciklik_son.UC_CM) is measured per variant on
the TRAINING floors (plevne_1, tekke_k1, ebubekir_k1), never on the test floors.
usage: python aciklik_karsilastir.py NAME [NAME...]   NAME = prediction suffix, e.g. a3_etiketsiz  a3_etiketsiz_k3  a5_etiketsiz_k3"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
from pathlib import Path
import numpy as np
from shapely.geometry import Point
import aciklik_son as S  # noqa: E402
import test2  # noqa: E402
from aciklik_kural import _frame  # noqa: E402
K = config.KT / "aciklik_ag"
TEST = ("malkoc_k2", "kisla_85_1", "kisla_84_1", "kisla_85_z", "kisla_84_z"); TRAIN = ("plevne_1", "tekke_k1", "ebubekir_k1")


def pred(name, key):
    return json.load(open(K / name.split("_")[0] / f"tahmin_{name}__{key}.json"))["tahmin"]


def bias(name):
    e = []
    for key in TRAIN:
        G, s = test2.gt(key)
        for p in pred(name, key):
            c = Point(*np.mean(p["box"], 0)); g = next((g for g in G if g["poly"].buffer(0.1 * s).contains(c)), None)
            if g is None: continue
            gc = np.asarray(g["poly"].minimum_rotated_rectangle.exterior.coords)[:4]; a, u, n, L, _, _ = _frame(gc.tolist()); pc = np.asarray(p["box"], float)
            if abs(_frame(pc.tolist())[1] @ u) < 0.8: continue
            ts = sorted((q - a) @ u for q in pc); e += [(0 - (ts[0] + ts[1]) / 2) / s * 100, ((ts[2] + ts[3]) / 2 - L) / s * 100]
    return float(np.median(e)) if e else 0.0


def degerlendir(name):
    S.UC_CM = max(0.0, round(bias(name), 1)); json.dump({"UC_CM": S.UC_CM}, open(K / name.split("_")[0] / f"UC_{name}.json", "w")); T = {"n": 0, "bulunan": 0, "kacan": 0, "yanlis": 0, "yanlis_gecerli": 0, "gecersiz": 0, "kasa3": 0, "kasa5": 0}; per = []
    for key in TEST:
        r, rows = S.olc(pred(name, key), key); m = [x for x in rows if x["eslesen"]]; n = r["etiket"]["door"] + r["etiket"]["window"]
        f = [x for x in rows if not x["eslesen"] and not x["guvenilmez_uzerinde"]]
        T["n"] += n; T["bulunan"] += len(m); T["kacan"] += n - len(m); T["yanlis"] += len(f); T["yanlis_gecerli"] += sum(x["gecerli"] for x in f)
        T["gecersiz"] += sum(not x["gecerli"] for x in m); T["kasa3"] += sum(x["kasa_cm"] <= 3 for x in m); T["kasa5"] += sum(x["kasa_cm"] <= 5 for x in m)
        per.append(f"{key}:{len(m)}/{n} med{r['kasa_med_cm']}")
    T["dogru_gecerli"] = T["bulunan"] - T["gecersiz"]
    print(f"{name:22} uc_duzeltme {S.UC_CM} cm | {T} | {' '.join(per)}")
    return T


if __name__ == "__main__":
    for nm in sys.argv[1:]: degerlendir(nm)
