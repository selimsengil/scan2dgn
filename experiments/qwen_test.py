"""Can Qwen3.8-27B see and choose like 6.1 Sol? Tekke's four held-out rooms.

View (as Sol's FloorCanvas `focus`): the room crop at ORIGINAL pixels (no downscale), a 40 px
ruler margin on top and left, ticks every 50 source px labelled in crop pixels; the supplied
entrance (the same verified door span as the earlier DeepSeek test) drawn in orange.

  test 1 'see'   : Qwen writes the ordered inner-face corners in ruler coordinates
  test 2 'choose': measured line hypotheses (source_hypotheses.detect_hypotheses, F-ids) drawn
                   on the same view; Qwen writes the ordered F-ids; compile_face_loop
                   intersects them. Coverage = can the reference edges be matched by some F
                   at all (the candidate generator's share of the error).
Scored with the benchmark's geometry_metrics against the DGN room.

usage: python qwen_test.py [--tests see,choose] [--think] [--out runs/qwen_tekke_a]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r"C:\Users\Selim\Documents\cizim_projeler\asistan_cizim\20261107_tekke_dort_oda")
sys.path.insert(0, r"C:\Users\Selim\Documents\cizim_projeler\egitim_verisi\room_boundary_benchmark_v1_20261031")
sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\src")
from run_pilot import SOURCE_DIR, source  # noqa: E402
from evaluate_pilot import verified_hint  # noqa: E402
from geometry_metrics import metrics  # noqa: E402
from local_computer_agent.architect import source_hypotheses as SH  # noqa: E402

import qwen_client as QC  # noqa: E402

CASES = {   # the earlier DeepSeek four-room test (workbench.py): box, room id, door id, door normal, target
    "01_mutfak": ((2050, 1690, 2790, 2575), 55013, 55385, (0, 1), "MUTFAK 13.00"),
    "02_yatak": ((2280, 650, 2910, 1660), 54937, 55381, (0, -1), "ODA 14.00"),
    "03_ebeveyn": ((5950, 820, 6740, 1780), 54965, 55433, (0, -1), "EBEVEYN ODASI 15.70"),
    "04_salon": ((8920, 1770, 9930, 2655), 55037, 55467, (0, 1), "SALON 20.00"),
}
PAD = 40
FONT = ImageFont.truetype("arial.ttf", 12)
FONTB = ImageFont.truetype("arialbd.ttf", 12)


def cases():
    img = source()
    items = {r["id"]: r for r in json.loads((SOURCE_DIR / "DGN_TARGETS.json").read_text(encoding="utf8"))["elements"]}
    for name, (box, room, door, normal, target) in CASES.items():
        x, y = box[:2]
        door_shape = [[a - x, b - y] for a, b in items[door]["points"]]
        hint = verified_hint(door_shape, normal)
        ref = [[a - x, b - y] for a, b in items[room]["points"]]
        if ref[0] == ref[-1]:
            ref = ref[:-1]
        yield name, img.crop(box), target, hint, ref


def ruled(crop: Image.Image, hint, faces=None) -> Image.Image:
    """Original pixels with a 40 px ruler (ticks every 50 crop px), entrance in orange."""
    w, h = crop.size
    out = Image.new("RGB", (w + PAD, h + PAD), "white")
    out.paste(crop, (PAD, PAD))
    d = ImageDraw.Draw(out)
    for v in range(0, max(w, h) + 1, 10):
        L = 10 if v % 50 == 0 else 4
        if v <= w:
            d.line((PAD + v, PAD - L, PAD + v, PAD), fill="navy")
            if v % 100 == 0:
                d.text((PAD + v + 2, 2), str(v), fill="navy", font=FONT)
        if v <= h:
            d.line((PAD - L, PAD + v, PAD, PAD + v), fill="navy")
            if v % 100 == 0:
                d.text((2, PAD + v + 1), str(v), fill="navy", font=FONT)
    a, b = hint.jambs
    d.line([(a[0] + PAD, a[1] + PAD), (b[0] + PAD, b[1] + PAD)], fill="#ef5000", width=3)
    if faces:
        used = []
        for r in faces:
            for s in r["segments"]:
                d.line([(s["start"][0] + PAD, s["start"][1] + PAD), (s["end"][0] + PAD, s["end"][1] + PAD)], fill=(0, 120, 255), width=1)
            lg = max(r["segments"], key=lambda s: s["length"])
            ax, ay = (lg["start"][0] + lg["end"][0]) / 2 + PAD, (lg["start"][1] + lg["end"][1]) / 2 + PAD
            tb = d.textbbox((0, 0), r["id"], font=FONTB); tw, th = tb[2] + 3, tb[3] + 2
            for dx, dy in [(4, 2), (4, -th - 3), (-tw - 3, 2), (-tw - 3, -th - 3), (8, 16), (-tw - 6, 16)]:
                x0 = max(0, min(out.width - tw, ax + dx)); y0 = max(0, min(out.height - th, ay + dy))
                bx = [x0, y0, x0 + tw, y0 + th]
                if not any(bx[0] < o[2] and bx[2] > o[0] and bx[1] < o[3] and bx[3] > o[1] for o in used):
                    used.append(bx); d.line([(ax, ay), (x0, y0)], fill="blue")
                    d.rectangle(bx, fill="white", outline=(140, 190, 255)); d.text((x0 + 1, y0 - 1), r["id"], fill="blue", font=FONTB)
                    break
    return out


RULES = ("Kurallar: sınır, odayı çevreleyen duvarların ODAYA BAKAN İÇ YÜZ çizgisinden geçer. Kapı ve pencere "
         "açıklıkları iç yüz çizgisinin uzantısıyla düz kapatılır (kanat, yay, kasa çizilmez). Odaya taşan kolonların "
         "yüzleri izlenir (girinti yapılır). Mobilya, yazı, ölçü ve aks çizgileri, tarama, mühür ve imza sınır değildir.")

Q_SEE = ("Görüntü bir kat planından '{target}' odasının ÖZGÜN çözünürlükte kesiti. Üstte ve solda cetvel var; sayılar "
         "piksel koordinatı (her büyük çentik 50 px, yazılar her 100 px). TURUNCU çizgi odanın giriş kapısı açıklığı "
         "(doğrulanmış, bu açıklık sınıra dahil edilip düz geçilir).\n" + RULES +
         "\nOdanın iç sınırının köşelerini saat yönünde, sırayla, cetvel koordinatlarıyla ver. Yalnız JSON: "
         "{{\"points\": [[x, y], ...]}}")

Q_CHOOSE = ("Görüntü bir kat planından '{target}' odasının ÖZGÜN çözünürlükte kesiti. Üstte ve solda cetvel var (piksel). "
            "TURUNCU çizgi odanın giriş kapısı açıklığı. MAVİ ince çizgiler ve F ile başlayan kutular, görüntüden ölçülmüş "
            "çizgi adayları; her biri gerçek bir mürekkep çizgisi ama duvar, mobilya, ölçü ya da aks olabilir.\n" + RULES +
            "\nOdanın iç sınırını oluşturan duvar iç yüzlerini, odanın etrafında saat yönünde DOLAŞARAK sırayla F "
            "kimlikleriyle seç (her kenar için bir aday; kapı açıklığında açıklığın iki yanındaki aynı duvar yüzünü "
            "kullan). Köşeleri araç hesaplayacak. Yalnız JSON: {{\"faces\": [\"F001\", ...]}}. Emin olamadığın bir "
            "kenar varsa yine de en olası adayı seç.")


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # an answer cut right after its points (generation stopped at "neden"): the points alone
    p = re.search(r'"points"\s*:\s*(\[\s*\[.*?\]\s*\])', text, re.S)
    if p:
        try:
            return {"points": json.loads(p.group(1))}
        except Exception:
            return {}
    return {}


def score(points, ref, hint):
    try:
        return metrics(points, ref, seed=hint.inside)
    except Exception as e:
        return {"valid": False, "reason": str(e)[:120]}


def brief(m: dict) -> dict:
    if not m.get("valid"):
        return {"valid": False, "reason": m.get("reason")}
    return {"IoU": round(m["area_IoU"], 3), "F1@5": round(m["boundary_5"]["F1"], 3), "F1@10": round(m["boundary_10"]["F1"], 3),
            "corners@10": round(m["corners_10"]["F1"], 3), "turns": [m["predicted_turns"], m["reference_turns"]]}


def coverage(faces, ref, tol=4.0, ang=3.0):
    """Reference edges that some measured face lies on (direction within ang deg, offset within tol px)."""
    hit = 0
    for i in range(len(ref)):
        a, b = np.asarray(ref[i], float), np.asarray(ref[(i + 1) % len(ref)], float)
        u = (b - a) / (np.linalg.norm(b - a) + 1e-9); n = np.array([-u[1], u[0]]); mid = (a + b) / 2
        ok = False
        for r in faces:
            v = np.asarray(r["direction"]); o = np.asarray(r["origin"])
            if abs(float(v @ u)) < math.cos(math.radians(ang)):
                continue
            if abs(float((mid - o) @ n)) <= tol:
                ok = True; break
        hit += ok
    return hit, len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="see,choose")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("runs") / "qwen_tekke_a"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tests = a.tests.split(",")
    results = {}
    for name, crop, target, hint, ref in cases():
        r = {"target": target}
        if "see" in tests:
            view = ruled(crop, hint)
            view.save(out / f"{name}_see.png")
            t0 = time.time()
            ans, tm = QC.ask(Q_SEE.format(target=target), [np.asarray(view)[:, :, ::-1]], think=a.think, max_tokens=6000 if a.think else 800)
            pts = parse_json(ans).get("points") or []
            try:
                pts = [[float(p[0]), float(p[1])] for p in pts]
            except Exception:
                pts = []
            r["see"] = {"points": pts, "raw": ans[-600:], "seconds": round(time.time() - t0, 1),
                        "tokens": tm.get("predicted_n"), "score": brief(score(pts, ref, hint)) if len(pts) >= 3 else {"valid": False, "reason": "fewer than 3 points"}}
            print(name, "see", r["see"]["score"], r["see"]["seconds"], "s", flush=True)
        if "choose" in tests:
            crop_path = out / f"{name}_crop.png"; crop.save(crop_path)
            cat = SH.shortlist(SH.detect_hypotheses(str(crop_path)))
            faces = cat["faces"]
            cov = coverage(faces, ref)
            view = ruled(crop, hint, faces)
            view.save(out / f"{name}_choose.png")
            t0 = time.time()
            ans, tm = QC.ask(Q_CHOOSE.format(target=target), [np.asarray(view)[:, :, ::-1]], think=a.think, max_tokens=6000 if a.think else 800)
            ids = [str(x) for x in (parse_json(ans).get("faces") or [])]
            res = {"faces": ids, "raw": ans[-600:], "seconds": round(time.time() - t0, 1), "tokens": tm.get("predicted_n"),
                   "candidates": len(faces), "coverage": f"{cov[0]}/{cov[1]} reference edges have a candidate"}
            try:
                loop = SH.compile_face_loop(cat, ids, anchor=hint.inside)
                res["score"] = brief(score(loop["vertices_scene"], ref, hint))
            except Exception as e:
                res["score"] = {"valid": False, "reason": str(e)[:150]}
            r["choose"] = res
            print(name, "choose", res["coverage"], res["score"], res["seconds"], "s", flush=True)
        results[name] = r
        json.dump(results, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

