"""Few-shot: solved rooms in the context before the question (as Selim taught 6.1 Sol).

Examples: Tekke teaching rooms (OGRETIM_ORNEKLERI.json; none is a test room), each shown as
question + crop -> answer {"points": DGN room in 0-1000 of the crop, "neden": the teaching note}.
Then the four Tekke test rooms are asked the same way. Scored raw and snapped.

usage: python qwen_fewshot_test.py [--examples 06,05,03] [--think] [--out runs/fewshot_a]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import time
from pathlib import Path

from PIL import ImageDraw

import qwen_agent as QA
import qwen_test as QT
import qwen_ground_test as QG
import snap
from run_pilot import SOURCE_DIR, source

TEACH = json.loads((SOURCE_DIR / "OGRETIM_ORNEKLERI.json").read_text(encoding="utf8"))["examples"]
ITEMS = {r["id"]: r for r in json.loads((SOURCE_DIR / "DGN_TARGETS.json").read_text(encoding="utf8"))["elements"]}

SYSTEM = ("Sen mimari kat planlarından oda sınırı çizen dikkatli bir teknik ressamsın. " + QT.RULES +
          " Ölçü çizgileri ince, uçlarında çentik ya da ok olan ve üstünde sayı yazan çizgilerdir; sınır değildir. "
          "Önce çözülmüş örnekleri incele; yeni odada aynı kuralları aynı şekilde uygula. Yalnız JSON yaz.")

Q_EX = ("Görüntü bir kat planından bir odanın kesiti. Odanın iç sınırının köşelerini saat yönünde sırayla, görüntünün 0-1000 "
        "normalize koordinatlarında ver. JSON: {\"points\": [[x, y], ...], \"neden\": \"kısa açıklama\"}")


def example_messages(numbers):
    img = source()
    msgs = []
    for ex in TEACH:
        if ex["number"] not in numbers:
            continue
        x0, y0, x1, y1 = ex["crop_box"]
        crop = img.crop((x0, y0, x1, y1)); w, h = crop.size
        pts = ITEMS[ex["room_id"]]["points"]
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        norm = [[round((x - x0) * 1000 / w), round((y - y0) * 1000 / h)] for x, y in pts]
        msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": QA.to_data_url(crop)}},
                                                 {"type": "text", "text": f"ÖRNEK {ex['number']} ({ex['title']}). " + Q_EX}]})
        msgs.append({"role": "assistant", "content": json.dumps({"points": norm, "neden": ex["focus"]}, ensure_ascii=False)})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="06,05,03")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).with_name("runs") / "fewshot_a"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shots = example_messages(set(a.examples.split(",")))
    res = {}
    for name, crop, target, hint, ref in QT.cases():
        v = crop.convert("RGB").copy(); d = ImageDraw.Draw(v)
        ja, jb = hint.jambs; d.line([tuple(ja), tuple(jb)], fill="#ef5000", width=3)
        W, H = v.size
        msgs = [{"role": "system", "content": SYSTEM}] + shots + [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": QA.to_data_url(v)}},
                                         {"type": "text", "text": f"YENİ ODA: '{target}'. TURUNCU çizgi odanın giriş kapısı açıklığı "
                                                                  "(sınır buradan düz geçer). Örneklerdeki gibi çöz. " + Q_EX}]}]
        t0 = time.time()
        text, tm = QA.chat(msgs, a.think, 14000 if a.think else 900)
        js = QT.parse_json(text)
        pts = js.get("points") or []
        try:
            px = [[float(x) * W / 1000.0, float(y) * H / 1000.0] for x, y in pts]
        except Exception:
            px = []
        raw_s = QT.brief(QT.score(px, ref, hint)) if len(px) >= 3 else {"valid": False, "reason": "no polygon"}
        sn = snap.snap_polygon(crop, px) if len(px) >= 3 else []
        sn_s = QT.brief(QT.score(sn, ref, hint)) if len(sn) >= 3 else {"valid": False}
        res[name] = {"target": target, "coarse": px, "snapped": sn, "neden": js.get("neden"), "raw_score": raw_s, "snapped_score": sn_s,
                     "seconds": round(time.time() - t0, 1), "tokens": tm.get("predicted_n")}
        print(name, "raw", raw_s, "| snapped", sn_s, "|", res[name]["seconds"], "s |", (js.get("neden") or "")[:120], flush=True)
        json.dump(res, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
