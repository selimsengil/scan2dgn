"""OCR of a floor raster: every text box, then the room labels among them.

Room label = a name line (YATAK ODASI, SALON, ...) with an area line "A=14.02 m2" just
below it. Output per floor: ocr_<floor>.json with all boxes and the paired labels
{name, area_m2, xy (centre of the name+area pair), box}.

usage: python ocr_labels.py --floors 015 [--tile 1600]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

B = Path(r"C:\Users\Selim\Documents\cizim_projeler\egitim_verisi\sam_etiket_hassas_v3_20261022\floors")
OUT = Path(__file__).with_name("ocr")

# "A=14.02 m2", "A:3,5m²", "11.4m2", "ANTRE6m2", "24m2" (no dimension-only numbers: m2 required)
AREA_RE = re.compile(r"(?:A\s*[=:]\s*)?([0-9]{1,3}(?:[.,][0-9]{1,2})?)\s*m\s*[2²]", re.I)


def ocr_tiles(img: np.ndarray, eng: RapidOCR, tile: int, overlap: int) -> list[dict]:
    H, W = img.shape[:2]
    boxes = []
    for y0 in range(0, max(1, H - overlap), tile - overlap):
        for x0 in range(0, max(1, W - overlap), tile - overlap):
            crop = img[y0:y0 + tile, x0:x0 + tile]
            if crop.size == 0 or (crop < 128).mean() < 0.002:
                continue
            res, _ = eng(crop)
            for quad, text, conf in res or []:
                q = np.asarray(quad, float) + [x0, y0]
                boxes.append({"text": text, "conf": float(conf), "quad": q.round(1).tolist()})
    # the same text found by two overlapping tiles: keep the more confident
    keep = []
    for b in sorted(boxes, key=lambda b: -b["conf"]):
        c = np.mean(b["quad"], axis=0)
        if any(k["text"] == b["text"] and np.linalg.norm(np.mean(k["quad"], axis=0) - c) < 20 for k in keep):
            continue
        keep.append(b)
    return keep


def pair_labels(boxes: list[dict]) -> list[dict]:
    """An area line and the text line right above it (or the area line alone, if it carries the name)."""
    def bb(b):
        q = np.asarray(b["quad"]); return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
    labels = []
    for a in boxes:
        m = AREA_RE.search(a["text"].replace(" ", ""))
        if not m:
            continue
        ax0, ay0, ax1, ay1 = bb(a); h = ay1 - ay0; cx = (ax0 + ax1) / 2
        name, nb = None, None
        head = AREA_RE.split(a["text"])[0].strip()
        for n in boxes:
            if n is a or AREA_RE.search(n["text"].replace(" ", "")):
                continue
            nx0, ny0, nx1, ny1 = bb(n)
            if 0 <= ay0 - ny1 < 1.5 * h and nx0 - h < cx < nx1 + h and len(n["text"]) >= 2:
                if nb is None or ny1 > bb(nb)[3]:
                    name, nb = n["text"], n
        if name is None and len(head) >= 2:
            name = head
        x0, y0, x1, y1 = ax0, ay0, ax1, ay1
        if nb is not None:
            nx0, ny0, nx1, ny1 = bb(nb); x0, y0, x1, y1 = min(x0, nx0), min(y0, ny0), max(x1, nx1), max(y1, ny1)
        labels.append({"name": name, "area_m2": float(m.group(1).replace(",", ".")),
                       "xy": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
                       "box": [round(v, 1) for v in (x0, y0, x1, y1)], "area_text": a["text"]})
    return labels


ROOM_WORDS = ["salon", "oda", "yatak", "yatod", "yat.od", "mutfak", "banyo", "ban.", "wc", "balkon", "antre", "hol",
               "holü", "koridor", "kiler", "depo", "dükkan", "dukkan", "otopark", "sığınak", "siginak", "kalorifer",
               "kazan", "merdiven", "asansör", "asansor", "n.o.a", "elektrik", "su merkezi", "teras", "lavabo",
               "giyinme", "çamaşır", "camasir", "ofis", "büro", "buro", "işyeri", "isyeri", "dairesi", "merkezi",
               "nişi", "nisi", "sahanlık", "sahanlik", "hobi", "çalışma", "calisma", "yemek", "ebv"]
NOT_ROOM = ["seramik", "boya", "fayans", "krec", "kireç", "badana", "parke", "mermer", "plboya", "sıva", "siva"]


def name_labels(boxes: list[dict], have: list[dict]) -> list[dict]:
    """Room-name lines with no area line paired: seeds without a printed area."""
    out = []
    for b in boxes:
        t = b["text"].lower().replace("i̇", "i")
        if AREA_RE.search(t.replace(" ", "")) or any(w in t for w in NOT_ROOM):
            continue
        if not any(w in t for w in ROOM_WORDS) or len(t) < 2:
            continue
        q = np.asarray(b["quad"]); x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
        if any(h["box"][0] - 5 <= (x0 + x1) / 2 <= h["box"][2] + 5 and h["box"][1] - 5 <= (y0 + y1) / 2 <= h["box"][3] + 5 for h in have):
            continue   # the name of a paired label
        out.append({"name": b["text"], "area_m2": None, "xy": [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)],
                    "box": [round(v, 1) for v in (x0, y0, x1, y1)], "area_text": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default="015")
    ap.add_argument("--tile", type=int, default=1600)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    eng = RapidOCR()
    for f in a.floors.split(","):
        img = cv2.imread(str(B / f / "clean.png"))
        boxes = ocr_tiles(img, eng, a.tile, 200)
        labels = pair_labels(boxes)
        labels += name_labels(boxes, labels)
        json.dump({"floor": f, "boxes": boxes, "labels": labels}, open(OUT / f"ocr_{f}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f, "boxes", len(boxes), "labels", len(labels))
        for l in labels:
            print("  ", l["name"], l["area_m2"], l["xy"])


if __name__ == "__main__":
    main()
