"""Qwen3.8-27B drawing a room the way 6.1 Sol does (FloorCanvas): it looks, zooms and measures
with tools, and writes the corners itself. No candidate lists.

Tools (one JSON action per turn):
  {"action":"focus","box":[x0,y0,x1,y1],"scale":1|2}   -> that region at original pixels (x2 nearest),
        ruler on top/left in room-crop coordinates (ticks every 50 px at x1, 25 px at x2)
  {"action":"profile","box":[x0,y0,x1,y1],"axis":"x"|"y"} -> dark-ink density peaks across the box:
        axis "x" gives the x positions of vertical lines, "y" the y positions of horizontal lines
  {"action":"final","points":[[x,y],...]}              -> the ordered inner-face corners
Tekke's four held-out rooms, scored with the benchmark metrics against the DGN rooms.

usage: python qwen_agent.py [--think] [--steps 12] [--out runs/qwen_agent_a] [--cases 04_salon]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import base64
import io
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import qwen_test as QT

URL = "http://127.0.0.1:8091/v1/chat/completions"
PAD = 40
FONT = ImageFont.truetype("arial.ttf", 12)

SYSTEM = """Sen mimari kat planlarından oda sınırı çizen dikkatli bir teknik ressamsın. Taranmış bir planda tek bir odanın iç sınırını çizeceksin.

Kurallar:
- Sınır, odayı çevreleyen duvarların ODAYA BAKAN İÇ YÜZ çizgisinden geçer (duvarın ortasından ya da dış yüzünden değil).
- Kapı ve pencere açıklıkları, iç yüz çizgisinin uzantısıyla düz kapatılır; kanat, yay, kasa çizilmez.
- Odaya taşan kolonların yüzleri izlenir (girinti yapılır). Eğik duvar varsa eğik izlenir.
- Mobilya, yazı, ölçü ve aks çizgileri, fayans taraması, mühür ve imza sınır değildir.
- Bir duvarın iki yüzü vardır (iki paralel çizgi); odaya bakan yüzü seç.

Araçların (her mesajda TEK bir JSON eylemi yaz, başka bir şey yazma):
1. {"action":"focus","box":[x0,y0,x1,y1],"scale":1}  -> o bölgenin özgün piksellerde görüntüsü (scale 2: iki kat büyütülmüş). Cetveldeki sayılar oda kesitinin piksel koordinatı. Küçük ayrıntıda (köşe, kolon, kapı kasası) scale 2 ve dar kutu kullan.
2. {"action":"profile","box":[x0,y0,x1,y1],"axis":"x"} -> kutudaki koyu çizgilerin konumlarını sayı olarak verir. axis "x": dikey çizgilerin x konumları; axis "y": yatay çizgilerin y konumları. Bir duvarın iki yüzünü ayırmak için duvara dik, dar bir kutu kullan.
3. {"action":"final","points":[[x,y],...]} -> odanın iç sınır köşeleri, saat yönünde sırayla. Koordinatları ölçtüğün çizgi konumlarından al, tahmin etme.

Yöntem: Önce genel görünümde odayı ve duvarlarını bul. Sonra her duvar için iç yüzün konumunu profile ile ölç; köşe, kolon ve kapıları focus ile yakından doğrula. Ölçümler bitince final yaz."""


def to_data_url(img: Image.Image) -> str:
    b = io.BytesIO(); img.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def ruled_view(crop: Image.Image, box, scale: int, hint=None) -> Image.Image:
    """Region `box` (crop coordinates) at original pixels x scale, ruler labelled in crop coordinates."""
    x0, y0, x1, y1 = box
    reg = crop.crop((x0, y0, x1, y1))
    if scale > 1:
        reg = reg.resize((reg.width * scale, reg.height * scale), Image.Resampling.NEAREST)
    w, h = reg.size
    out = Image.new("RGB", (w + PAD, h + PAD), "white"); out.paste(reg, (PAD, PAD))
    d = ImageDraw.Draw(out)
    step, lab = (50, 100) if scale == 1 else (25, 50)
    for v in range((x0 // step) * step, x1 + 1, step // 5 if scale > 1 else 10):
        if v < x0: continue
        X = PAD + (v - x0) * scale; L = 10 if v % step == 0 else 4
        d.line((X, PAD - L, X, PAD), fill="navy")
        if v % lab == 0: d.text((X + 2, 2), str(v), fill="navy", font=FONT)
    for v in range((y0 // step) * step, y1 + 1, step // 5 if scale > 1 else 10):
        if v < y0: continue
        Y = PAD + (v - y0) * scale; L = 10 if v % step == 0 else 4
        d.line((PAD - L, Y, PAD, Y), fill="navy")
        if v % lab == 0: d.text((2, Y + 1), str(v), fill="navy", font=FONT)
    if hint is not None:
        a, b = hint.jambs
        d.line([(PAD + (a[0] - x0) * scale, PAD + (a[1] - y0) * scale), (PAD + (b[0] - x0) * scale, PAD + (b[1] - y0) * scale)], fill="#ef5000", width=3)
    return out


def profile(crop: Image.Image, box, axis: str, k: int = 12) -> list:
    g = np.asarray(crop.convert("L"), dtype=np.float32)
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = g[max(0, y0):y1, max(0, x0):x1]
    if sub.size == 0:
        return []
    dark = (sub < 140).astype(np.float32)
    prof = dark.mean(axis=0) if axis == "x" else dark.mean(axis=1)
    base = x0 if axis == "x" else y0
    peaks = []
    for i in range(len(prof)):
        if prof[i] < 0.25: continue
        if (i > 0 and prof[i - 1] > prof[i]) or (i + 1 < len(prof) and prof[i + 1] > prof[i]): continue
        if peaks and i - peaks[-1][0] <= 1:   # plateau: keep its middle
            if prof[i] >= peaks[-1][1]: peaks[-1] = (i, float(prof[i]))
            continue
        peaks.append((i, float(prof[i])))
    peaks = sorted(peaks, key=lambda p: -p[1])[:k]
    return [{"pos": base + i, "coverage": round(v, 2)} for i, v in sorted(peaks)]


def chat(messages, think: bool, max_tokens: int):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["choices"][0]["message"].get("content") or "", r.get("timings", {})


def action_of(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return {}
    s = m.group(0)
    for cand in (s, s[: s.rfind("}") + 1]):
        try: return json.loads(cand)
        except Exception: pass
    return {}


def run_case(name, crop, target, hint, ref, a, out: Path):
    W, H = crop.size
    first = ruled_view(crop, (0, 0, W, H), 1, hint)
    first.save(out / f"{name}_00_overview.png")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": to_data_url(first)}},
                    {"type": "text", "text": f"Bu, '{target}' odasının bulunduğu kesit ({W}x{H} piksel, özgün çözünürlük). "
                                             "TURUNCU çizgi odanın giriş kapısı açıklığı (doğrulanmış; sınır bu açıklıktan düz geçer). "
                                             "İlk eylemini yaz."}]}]
    log, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        text, tm = chat(messages, a.think, 8000 if a.think else 600)
        act = action_of(text)
        messages.append({"role": "assistant", "content": text})
        entry = {"step": step, "action": act, "tokens": tm.get("predicted_n"), "raw": text[-300:]}
        kind = act.get("action")
        if kind == "final":
            entry["points"] = act.get("points"); log.append(entry); break
        if kind == "focus":
            try:
                bx = [max(0, int(act["box"][0])), max(0, int(act["box"][1])), min(W, int(act["box"][2])), min(H, int(act["box"][3]))]
                sc = 2 if int(act.get("scale", 1)) >= 2 else 1
                if (bx[2] - bx[0]) * sc > 1120 or (bx[3] - bx[1]) * sc > 1000:   # Sol's focus limit
                    raise ValueError(f"kutu çok büyük: scale {sc} ile en fazla {1120 // sc}x{1000 // sc} piksel")
                if bx[2] - bx[0] < 8 or bx[3] - bx[1] < 8:
                    raise ValueError("kutu çok küçük")
                v = ruled_view(crop, bx, sc, hint)
                v.save(out / f"{name}_{step:02d}_focus.png")
                messages.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": to_data_url(v)}},
                    {"type": "text", "text": f"focus {bx} scale {sc}. Sonraki eylemin?"}]})
            except Exception as e:
                messages.append({"role": "user", "content": f"Hata: {e}. Geçerli bir eylem yaz."})
        elif kind == "profile":
            try:
                pk = profile(crop, act["box"], act.get("axis", "x"))
                entry["result"] = pk
                messages.append({"role": "user", "content": f"profile {act['box']} axis {act.get('axis')}: {json.dumps(pk)}. Sonraki eylemin?"})
            except Exception as e:
                messages.append({"role": "user", "content": f"Hata: {e}. Geçerli bir eylem yaz."})
        else:
            messages.append({"role": "user", "content": "Eylem okunamadı. Yalnız tek bir JSON eylemi yaz: focus, profile ya da final."})
        log.append(entry)
        print(f"  {name} step {step}: {kind} {act.get('box', '')} {act.get('axis', '')}", flush=True)
    pts = []
    for e in log:
        if e.get("points"):
            try: pts = [[float(p[0]), float(p[1])] for p in e["points"]]
            except Exception: pts = []
    sc = QT.brief(QT.score(pts, ref, hint)) if len(pts) >= 3 else {"valid": False, "reason": "no final polygon"}
    return {"target": target, "steps": len(log), "seconds": round(time.time() - t0, 1), "points": pts, "score": sc, "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--cases", default="")
    ap.add_argument("--out", default=str(Path(__file__).with_name("runs") / "qwen_agent_a"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want = set(a.cases.split(",")) if a.cases else None
    res = {}
    for name, crop, target, hint, ref in QT.cases():
        if want and name not in want: continue
        r = run_case(name, crop, target, hint, ref, a, out)
        res[name] = r
        print(name, r["score"], r["steps"], "steps", r["seconds"], "s", flush=True)
        json.dump(res, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
