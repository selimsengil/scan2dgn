"""Room labels read by Qwen (no OCR): name, printed area if any, and where (0-1000 of the tile).

The floor is cut into overlapping tiles of at most ~1100 px; Qwen lists the room labels of each.
Scored against the DGN rooms: a room is found if a Qwen label point falls inside it; compared with
the OCR labels (ocr/ocr_<floor>.json) the same way.

usage: python qwen_labels_test.py --floors 015,013
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import qwen_client as QC
import qwen_test as QT
import v0_watershed as V0

Q = ("Görüntü taranmış bir mimari kat planından bir parça. Plandaki ODA ETİKETLERİNİ oku: her odanın içine yazılmış oda adı "
     "(örneğin SALON, YATAK ODASI, MUTFAK, BANYO, WC, ANTRE, BALKON, KİLER, HOL, N.O.A; kısaltmalar olabilir: 'yatod.nişi', "
     "'ban.wc') ve varsa yanındaki alan değeri (örneğin 'A=14.02 m2', '11.4m2', 'al:7.50 m2').\n"
     "Kurallar: Ölçü sayılarını (260, 120...), kapı/pencere kodlarını (K1-100/210, P2-130/140), malzeme notlarını "
     "(DÖŞ SERAMİK, DUV PL.BOYA), mühür ve imza yazılarını ODA ETİKETİ SAYMA. Alan yazılmamışsa ya da okunamıyorsa "
     "area_m2 null olsun, alanı tahmin etme. Kısaltmayı açık adıyla da yaz. Parçanın kenarında kesilmiş etiketleri de yaz.\n"
     "Her etiket için adın ortasının konumunu görüntünün 0-1000 normalize koordinatlarında ver. Yalnız JSON:\n"
     "{\"labels\": [{\"text\": \"yazıldığı gibi\", \"name\": \"açık ad\", \"area_m2\": 14.02 veya null, \"point\": [x, y]}]}")


def tiles(W, H, size=1100, overlap=200):
    xs = list(range(0, max(1, W - overlap), size - overlap)) or [0]
    ys = list(range(0, max(1, H - overlap), size - overlap)) or [0]
    for y in ys:
        for x in xs:
            yield x, y, min(W, x + size), min(H, y + size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="015,013")
    ap.add_argument("--tile", type=int, default=1100)
    ap.add_argument("--out", default=str(Path(__file__).with_name("runs") / "qwen_labels_a"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for f in a.floors.split(","):
        img = cv2.imread(str(V0.B / f / "clean.png"))
        H, W = img.shape[:2]
        labels, t0 = [], time.time()
        for (x0, y0, x1, y1) in tiles(W, H, a.tile):
            tile = img[y0:y1, x0:x1]
            ans, _ = QC.ask(Q, [tile], think=False, max_tokens=2500)
            for l in QT.parse_json(ans).get("labels") or []:
                try:
                    px = x0 + float(l["point"][0]) * (x1 - x0) / 1000.0; py = y0 + float(l["point"][1]) * (y1 - y0) / 1000.0
                except Exception:
                    continue
                # the same label seen by two overlapping tiles: keep one
                if any(abs(px - m["xy"][0]) < 60 and abs(py - m["xy"][1]) < 60 and m["name"] == l.get("name") for m in labels):
                    continue
                labels.append({"text": l.get("text"), "name": l.get("name"), "area_m2": l.get("area_m2"), "xy": [round(px, 1), round(py, 1)]})
        secs = round(time.time() - t0, 1)
        gts = V0.load_rooms(f)
        ocr = json.load(open(Path(__file__).with_name("ocr") / f"ocr_{f}.json", encoding="utf-8"))["labels"]
        def found(pts):
            hit, area_ok, area_n = 0, 0, 0
            for g in gts:
                m = V0.poly_mask(g["poly"], (H, W))
                inside = [p for p in pts if 0 <= int(p["xy"][1]) < H and 0 <= int(p["xy"][0]) < W and m[int(p["xy"][1]), int(p["xy"][0])]]
                if inside:
                    hit += 1
                    if g["area_m2"]:
                        area_n += 1
                        v = inside[0].get("area_m2")
                        try:
                            area_ok += abs(float(v) - g["area_m2"]) < 0.051
                        except Exception:
                            pass
            return hit, area_ok, area_n
        q_hit, q_area, q_n = found(labels)
        o_hit, o_area, o_n = found([{"xy": l["xy"], "area_m2": l["area_m2"]} for l in ocr])
        summary[f] = {"rooms": len(gts), "qwen_labels": len(labels), "qwen_rooms_found": q_hit, "qwen_area_correct": f"{q_area}/{q_n}",
                      "ocr_rooms_found": o_hit, "ocr_area_correct": f"{o_area}/{o_n}", "seconds": secs}
        print(f, summary[f], flush=True)
        json.dump({"floor": f, "labels": labels}, open(out / f"labels_{f}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(summary, open(out / "SUMMARY.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
