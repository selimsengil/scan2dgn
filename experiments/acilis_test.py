"""Doors / windows / room labels test (no boundaries): can a model find every door and window opening with its printed
tag (K1-100/210, P1-120/140 ...) and every room label with its name and printed m2?  Labels: the user's DGN
(KapiAcikligi, PencereAcikligi, OdaAlani with Oda Adı / Oda M2).
  prep:  python acilis_test.py hazirla --floor Gonen_Gundogdu_771_11__K1_55537 --dir kat_test/acilis_gundogdu1_haiku [--scale 0.42]
  score: python acilis_test.py puanla --dir kat_test/acilis_gundogdu1_haiku
The test folder gets only kat.png and GOREV_METNI.md; the answer key stays in kat_test/acilis_anahtar/<floor>.json."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, re, sys, unicodedata
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Point, shape
from shapely import affinity

Image.MAX_IMAGE_PIXELS = None
H = Path(__file__).parent; R = H / "kat_test" / "dgn_katlar"; KEY = H / "kat_test" / "acilis_anahtar"

GOREV = """Bir mimarlık/CAD ofisi için taranmış kat planlarını okuyoruz. Çalışma klasörün: {dir}
Klasör dışındaki dosyalara bakma; bu körlemesine bir test. Çalışma dosyalarını yalnız bu klasöre yaz.

Görev: `kat.png` taranmış bir kat planıdır ({w} x {h} piksel, yaklaşık {ppm:.0f} piksel = 1 metre). Görüntüyü istediğin
kadar kesip yakınlaştırabilirsin (Python/PIL). Oda sınırı ÇİZME; yalnız şunları bul:

1. Bütün KAPI açıklıkları: her biri için açıklığın orta noktası (x, y) ve yanında yazan etiket (örn. "K1-100/210";
   okunamıyorsa null). Daire giriş kapıları, iç kapılar, ortak alan kapıları dahil.
2. Bütün PENCERE açıklıkları: orta nokta ve etiket (örn. "P1-120/140"; okunamıyorsa null). Balkon kapısı/pencere
   ayrımında paftadaki etikete göre karar ver (K... kapı, P... pencere).
3. Bütün ODA etiketleri: odanın adı (paftada yazdığı gibi, örn. "SALON", "YATAK ODASI", "BANYO"), paftada yazan m²
   (sayı) ve etiket yazısının ortasındaki nokta (x, y). Balkonları da yaz ama `"type": "balcony"` ile.

Koordinatlar `kat.png` piksel çerçevesinde: x sağa, y aşağı, (0,0) sol üst köşe.

Çıktı: bu klasöre `SONUC.json` yaz:
```json
{{"doors": [{{"x": 512, "y": 830, "tag": "K1-100/210"}}],
  "windows": [{{"x": 120, "y": 400, "tag": "P1-120/140"}}],
  "rooms": [{{"name": "SALON", "m2": 23.41, "x": 700, "y": 900, "type": "room"}}]}}
```
Bitirince kısa bir özet yaz: kaç kapı, kaç pencere, kaç oda, emin olmadığın yerler, kaç dakika sürdü.
"""


def norm(s): return re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKD", str(s or "").upper().replace("İ", "I").replace("ı", "I")).encode("ascii", "ignore").decode())


def hazirla(a):
    L = json.load(open(R / a.floor / "LABELS_COMMON.json", encoding="utf-8")); s = a.scale; ppm = L["px_per_m"] * s
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    im = Image.open(R / a.floor / "clean.png").convert("L"); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS); im.save(d / "kat.png")
    sc = lambda g: affinity.scale(shape(g).buffer(0), s, s, origin=(0, 0))
    key = {"floor": a.floor, "scale": s, "px_per_m": ppm, "image": [im.width, im.height],
           "doors": [sc(x["geometry"]).wkt for x in L["structures"] if x["kind"] == "door"],
           "windows": [sc(x["geometry"]).wkt for x in L["structures"] if x["kind"] == "window"],
           "rooms": [{"name": r["name"], "m2": r["area_text_value"], "poly": sc(r["interior_polygon"]).wkt} for r in L["rooms"]],
           "balconies": [sc(x["geometry"]).wkt for x in L["structures"] if x["kind"] == "balcony"]}
    KEY.mkdir(parents=True, exist_ok=True); json.dump(key, open(KEY / f"{d.name}.json", "w", encoding="utf-8"), ensure_ascii=False)
    (d / "GOREV_METNI.md").write_text(GOREV.format(dir=d.resolve(), w=im.width, h=im.height, ppm=ppm), encoding="utf-8")
    print("prepared", d, im.size, "doors", len(key["doors"]), "windows", len(key["windows"]), "rooms", len(key["rooms"]))


def match(points, polys, tol):
    """greedy one-to-one: each labelled opening takes the nearest unused answer point within tol (px)."""
    used = set(); pairs = []
    for i, P in enumerate(polys):
        best = None
        for j, (x, y) in enumerate(points):
            if j in used: continue
            dd = P.distance(Point(x, y))
            if dd <= tol and (best is None or dd < best[1]): best = (j, dd)
        if best: used.add(best[0]); pairs.append((i, best[0]))
    return pairs


def width_cm(P, ppm):
    c = np.array(P.minimum_rotated_rectangle.exterior.coords); e = sorted(np.hypot(*(c[i + 1] - c[i])) for i in range(2)); return e[1] / ppm * 100


def puanla(a):
    from shapely import wkt
    d = Path(a.dir); K = json.load(open(KEY / f"{d.name}.json", encoding="utf-8")); ppm = K["px_per_m"]
    b = (d / "SONUC.json").read_bytes(); A = None
    for enc in ("utf-8-sig", "cp1254"):
        try: A = json.loads(b.decode(enc)); break
        except Exception: pass
    out = {}
    for kind in ("doors", "windows"):
        polys = [wkt.loads(x) for x in K[kind]]; ans = [x for x in A.get(kind, []) if x.get("x") is not None]
        pr = match([(x["x"], x["y"]) for x in ans], polys, 0.5 * ppm); tagok = 0; wok = 0; tags = 0
        for i, j in pr:
            t = ans[j].get("tag")
            if t:
                tags += 1; m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(t))
                if m and abs(int(m.group(1)) - width_cm(polys[i], ppm)) <= 12: wok += 1
        out[kind] = {"etiket": len(polys), "cevap": len(ans), "bulunan": len(pr), "yanlis_yer": len(ans) - len(pr), "etiket_okunan": tags, "genislik_tutan": wok}
    rooms = [{"name": r["name"], "m2": r["m2"], "P": wkt.loads(r["poly"])} for r in K["rooms"]]; ans = [x for x in A.get("rooms", []) if x.get("type") != "balcony" and x.get("x") is not None]
    hit = 0; name_ok = 0; m2_ok = 0; used = set(); wrong = 0
    for j, x in enumerate(ans):
        inside = [i for i, r in enumerate(rooms) if r["P"].contains(Point(x["x"], x["y"])) and i not in used]
        if not inside: wrong += 1; continue
        i = inside[0]; used.add(i); hit += 1
        if norm(x.get("name")) == norm(rooms[i]["name"]) or norm(rooms[i]["name"]) in norm(x.get("name")): name_ok += 1
        try:
            if rooms[i]["m2"] is not None and abs(float(str(x.get("m2")).replace(",", ".")) - float(rooms[i]["m2"])) < 0.015: m2_ok += 1
        except Exception: pass
    out["rooms"] = {"etiket": len(rooms), "cevap": len(ans), "dogru_odada": hit, "oda_disinda_ya_da_tekrar": wrong, "ad_tutan": name_ok, "m2_tutan": m2_ok}
    json.dump(out, open(d / "PUAN.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--floor"); ap.add_argument("--dir"); ap.add_argument("--scale", type=float, default=0.42); a = ap.parse_args()
    {"hazirla": hazirla, "puanla": puanla}[a.cmd](a)
