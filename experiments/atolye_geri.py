"""Score a workshop answer (SONUC_<floor>.json in WORKDIR) and write feedback files back into WORKDIR.

usage: python atolye_geri.py WORKDIR LOGDIR TURN [--floor 019]
- kat_score.py runs on a copy in LOGDIR/skor_turN (keeps SKOR.html with tiles out of the workshop).
- WORKDIR/GERI_BILDIRIM_turN.md: summary + per-room table (IoU, corners@10, area ratio pred/DGN).
- WORKDIR/GERI_BILDIRIM_turN.png: full floor, DGN rooms green, its rooms magenta (unmatched predictions orange).
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Polygon

import v0_watershed as V0  # noqa: E402

PY = r"C:\Users\Selim\Documents\Proje\.venv\Scripts\python.exe"


def main():
    work, log, turn = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    floor = sys.argv[sys.argv.index("--floor") + 1] if "--floor" in sys.argv else "019"
    sk = log / f"skor_tur{turn}"
    sk.mkdir(parents=True, exist_ok=True)
    for f in (f"kat_{floor}.png", f"SONUC_{floor}.json"):
        shutil.copy(work / f, sk / f)
    subprocess.run([PY, "kat_score.py", str(sk), floor], cwd=Path(__file__).parent, check=True,
                   stdout=open(sk / "SKOR.txt", "w"), stderr=subprocess.STDOUT)
    S = json.load(open(sk / "SKOR.json", encoding="utf-8"))
    ans = json.load(open(work / f"SONUC_{floor}.json", encoding="utf-8"))
    preds = [r for r in ans.get("rooms", []) if r.get("type") != "balcony" and r.get("boundary") and len(r["boundary"]) >= 3]
    gts = [g for g in V0.load_rooms(floor) if g["poly"].geom_type == "Polygon"]
    s = S["summary"]
    L = [f"# Geri bildirim, tur {turn} (kat {floor})", "",
         f"- DGN'deki oda sayısı: {s['dgn_rooms']}. Senin poligonların (balkon hariç): {s['predicted_polygons']}.",
         f"- Bulunan (IoU >= 0.5): {s['found_IoU>=0.5']}. IoU >= 0.90: {s['IoU>=0.90']}. IoU >= 0.95: {s['IoU>=0.95']}.",
         f"- Ortalama IoU (tüm DGN odaları): {s['mean_IoU_all_dgn']}. Bulunanlarda: {s['mean_IoU_found']}.",
         f"- Köşe isabeti (corners@10, bulunan odalarda; DGN köşesinin 10 px içinde köşen var mı): {s['mean_corners_found']}.",
         f"- Fazla poligon (DGN'de karşılığı yok): {s['extra_predictions']}. Birleştirme hatası: {len(s['merges'])}. Bölme hatası: {len(s['splits'])}.",
         "", "Alan oranı = senin poligonun alanı / DGN poligonunun alanı (piksel). 1'den büyükse fazla geniş çizmişsin "
         "(ör. duvar ortası/dış yüzü), küçükse içeride kalmışsın.", "",
         f"Görüntü: `GERI_BILDIRIM_tur{turn}.png` (yeşil = ofisin DGN çizimi, mor = senin çizimin, turuncu = karşılığı olmayan "
         "poligonun). Tam çözünürlükte; kesip yakınlaştırarak bakabilirsin.", "",
         "| DGN odası | senin adın | IoU | köşe@10 | alan oranı |", "|---|---|---|---|---|"]
    matched = set()
    for r in S["rooms"]:
        if r["pred"] is None or r["pred"] == "":
            L.append(f"| {r['dgn_name']} ({r['dgn_area']} m²) | — bulunamadı | 0 | 0 | — |")
            continue
        j = int(r["pred"]); matched.add(j)
        ratio = Polygon(preds[j]["boundary"]).buffer(0).area / gts[r["dgn"]]["poly"].area
        L.append(f"| {r['dgn_name']} ({r['dgn_area']} m²) | {r['pred_name']} | {r['IoU']:.3f} | {r['corners@10']:.2f} | {ratio:.3f} |")
    (work / f"GERI_BILDIRIM_tur{turn}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    im = Image.open(work / f"kat_{floor}.png").convert("RGB")
    d = ImageDraw.Draw(im)
    for g in gts:
        q = list(g["poly"].exterior.coords)
        d.line(q, fill=(0, 170, 60), width=3)
    for j, r in enumerate(preds):
        q = [tuple(p) for p in r["boundary"]]
        d.line(q + q[:1], fill=(210, 0, 210) if j in matched else (255, 140, 0), width=2)
    im.save(work / f"GERI_BILDIRIM_tur{turn}.png")
    print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
