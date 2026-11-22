"""Overnight report for a Qwen platform run: replays every tool call from transcript.json through qwen_cizim to rebuild
the final state (so the drawings exist even if Qwen never exported), scores against the DGN, and writes RAPOR.html
(self-contained, images embedded) in the run folder.
usage: python rapor_gece.py kat_test\\qwen_019_p3 019"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import base64
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

import qwen_cizim as C  # noqa: E402
import v0_watershed as V0  # noqa: E402

run = Path(sys.argv[1]); floor = sys.argv[2] if len(sys.argv) > 2 else "019"
img_path = run / f"kat_{floor}.png"
notes = []


def font(n):
    try:
        return ImageFont.truetype("arial.ttf", n)
    except Exception:
        return ImageFont.load_default()


def b64(im, w=None):
    if w and im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- replay
tr_path = run / "transcript.json"
state = {}
ctx = {"image": img_path, "out": run / "_replay", "state": state}
ctx["out"].mkdir(exist_ok=True)
n_calls = 0
if tr_path.exists():
    log = json.load(open(tr_path, encoding="utf-8"))
    for e in log:
        for tc in e.get("tool_calls") or []:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                args = {}
            if name in ("submit",):
                continue
            try:
                C.handle(name, args, ctx); n_calls += 1
            except Exception as ex:
                notes.append(f"replay: {name} failed ({ex})")
else:
    log = []
    notes.append("transcript.json yok (deney beklenmedik şekilde bitti); çizimler yeniden kurulamadı.")
s = state

# ---------------------------------------------------------------- result + score
res_path = run / f"SONUC_{floor}.json"
exported = res_path.exists()
if s.get("rooms") and not exported:   # Qwen did not export: build the result from the replayed state
    rooms = [{"id": r["id"], "name": r.get("name"), "printed_area_m2": r.get("m2"), "type": r.get("type", "room"),
              "flat": r.get("flat"), "source_labels": [f"{r.get('name')} {r.get('m2')}"], "boundary": s["polys"].get(r["id"]),
              "boundary_status": C._level(s, r["id"]) or "missing"} for r in s["rooms"]]
    res_path.write_text(json.dumps({"floor": floor, "image": img_path.name, "rooms": rooms}, ensure_ascii=False), encoding="utf-8")
    notes.append("Qwen export çağırmadı; sonuç, çağrıları yeniden oynatarak platformdaki son halden kuruldu.")
score = {}
if res_path.exists():
    import subprocess
    p = subprocess.run([sys.executable, str(Path(__file__).parent / "kat_score.py"), str(run), floor], capture_output=True, text=True)
    if (run / "SKOR.json").exists():
        score = json.load(open(run / "SKOR.json", encoding="utf-8"))
    else:
        notes.append("kat_score hata: " + (p.stderr or "")[-400:])

meta = json.load(open(run / "RUN.json")) if (run / "RUN.json").exists() else {}
runner = (run / "runner.txt").read_text(encoding="utf-8", errors="replace") if (run / "runner.txt").exists() else ""
budgets = Counter(re.findall(r"think budget (\w+)", runner))
gts = [g for g in V0.load_rooms(floor) if g["poly"].geom_type == "Polygon"]

# ---------------------------------------------------------------- pictures
base = Image.open(img_path).convert("RGB")
W, H = base.size
COL = {"walked": (0, 110, 255), "drawn": (150, 0, 200), "sketch": (255, 150, 0)}


def floor_picture(with_gt):
    im = base.copy(); d = ImageDraw.Draw(im, "RGBA")
    if with_gt:
        for g in gts:
            q = [(x, y) for x, y in g["poly"].exterior.coords]; d.line(q, fill=(0, 170, 0, 230), width=5)
    for r in s.get("rooms", []):
        P = s["polys"].get(r["id"])
        if not P:
            continue
        col = COL.get(C._level(s, r["id"]), (200, 0, 0))
        q = [tuple(p) for p in P]
        d.polygon(q, fill=col + (35,)); d.line(q + q[:1], fill=col + (255,), width=4)
        c = Polygon(P).representative_point(); d.text((c.x - 20, c.y - 10), r["id"], fill=(0, 0, 0), font=font(22))
    return im


rows_by_id = {}
for row in (score.get("rooms") or []):
    rid = row.get("pred_id") or row.get("id")
    if rid:
        rows_by_id[str(rid)] = row


def room_tile(rid):
    P = s["polys"].get(rid)
    if not P:
        return None
    Pp = Polygon(P)
    best = max(gts, key=lambda g: g["poly"].intersection(Pp.buffer(0)).area, default=None)
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    if best is not None:
        bx = best["poly"].bounds; xs += [bx[0], bx[2]]; ys += [bx[1], bx[3]]
    m = 30
    x0, y0, x1, y1 = max(0, min(xs) - m), max(0, min(ys) - m), min(W, max(xs) + m), min(H, max(ys) + m)
    crop = base.crop((int(x0), int(y0), int(x1), int(y1)))
    sc = min(520 / crop.width, 420 / crop.height, 4.0)
    crop = crop.resize((int(crop.width * sc), int(crop.height * sc)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(crop, "RGBA"); T = lambda p: ((p[0] - x0) * sc, (p[1] - y0) * sc)
    iou = None
    if best is not None:
        q = [T(p) for p in best["poly"].exterior.coords]; d.line(q, fill=(0, 170, 0, 230), width=4)
        iou = best["poly"].intersection(Pp.buffer(0)).area / best["poly"].union(Pp.buffer(0)).area
    q = [T(p) for p in P]; d.line(q + q[:1], fill=(0, 90, 255, 255), width=3)
    t = (s.get("traces") or {}).get(rid)
    if t:
        for k, (p, ev) in enumerate(zip(t["pts"], t["events"])):
            X, Y = T(p); d.ellipse([X - 4, Y - 4, X + 4, Y + 4], fill=(0, 190, 220, 255))
            if ev not in ("corner", "start"):
                d.text((X + 5, Y - 14), ev, fill=(200, 0, 0), font=font(13))
    return crop, iou, best


# ---------------------------------------------------------------- html
summ = score.get("summary", {})
out = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
       f"<title>Qwen kat {floor} raporu</title><style>",
       ":root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--card:#f5f5f7;--line:#ddd}",
       "@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#eee;--mut:#aaa;--card:#222226;--line:#333}}",
       "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}",
       "img{max-width:100%;height:auto;border-radius:6px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}",
       ".card{background:var(--card);border-radius:10px;padding:10px}.kpi{display:flex;flex-wrap:wrap;gap:10px}.kpi div{background:var(--card);padding:10px 14px;border-radius:10px}",
       ".kpi b{display:block;font-size:22px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:13px}",
       ".mut{color:var(--mut)}</style></head><body>",
       f"<h1>Qwen 3.8 · kat {floor} · çizim platformu</h1>"]
mins = (meta.get("seconds") or 0) / 60
walked = sum(1 for r in s.get("rooms", []) if C._level(s, r["id"]) == "walked")
out.append("<div class='kpi'>"
           f"<div><b>{mins:.0f} dk</b>süre</div><div><b>{meta.get('steps', '?')}</b>adım</div>"
           f"<div><b>{len(s.get('rooms', []))}</b>kayıtlı oda (DGN: {len(gts)})</div><div><b>{walked}</b>yürünerek çizilen</div>"
           f"<div><b>{summ.get('found_IoU>=0.5', '?')}</b>bulunan oda (IoU≥0.5)</div>"
           f"<div><b>{summ.get('mean_IoU_found', '?')}</b>ort. IoU (bulunan)</div>"
           f"<div><b>{summ.get('mean_corners_found', '?')}</b>köşe@10</div>"
           f"<div><b>{'evet' if meta.get('submitted') else 'hayır'}</b>Qwen export etti</div></div>")
usage = meta.get("usage") or {}
out.append(f"<p class='mut'>Token: girdi {usage.get('prompt_tokens', '?'):,} · çıktı {usage.get('completion_tokens', '?'):,}. "
           f"Düşünme bütçeleri (adım sayısı): {', '.join(f'{k}: {v}' for k, v in sorted(budgets.items()))}.</p>"
           if usage else "")
for n in notes:
    out.append(f"<p><b>Not:</b> {n}</p>")
out.append("<h2>Qwen'in çizimi</h2><p class='mut'>Mavi: duvar yürüyüşüyle çizilen · mor: elle çizilen · turuncu: kabataslak kalan</p>")
out.append(f"<img src='{b64(floor_picture(False), 1600)}'>")
out.append("<h2>DGN ile karşılaştırma</h2><p class='mut'>Yeşil: DGN (doğru cevap) · renkli: Qwen</p>")
out.append(f"<img src='{b64(floor_picture(True), 1600)}'>")
out.append("<h2>Oda oda</h2><p class='mut'>Yeşil: DGN · mavi: Qwen · camgöbeği noktalar: yürüyüş noktaları (kapı/kolon olayları kırmızı yazılı)</p><div class='grid'>")
table = []
for r in s.get("rooms", []):
    rid = r["id"]; tile = room_tile(rid)
    lvl = C._level(s, rid) or "çizilmedi"; nv = len(s.get("versions", {}).get(rid, []))
    t = (s.get("traces") or {}).get(rid)
    evs = Counter(t["events"]) if t else {}
    if tile:
        crop, iou, best = tile
        out.append(f"<div class='card'><img src='{b64(crop)}'><div><b>{rid}</b> {r.get('name')} {r.get('m2') or ''} m² · {lvl} · "
                   f"IoU {iou:.2f} · {nv} sürüm" + (f" · yürüyüş: {', '.join(f'{k} {v}' for k, v in evs.items())}" if evs else "") + "</div></div>")
        table.append((rid, r.get("name"), r.get("m2"), lvl, f"{iou:.3f}", nv, C._flag(s, rid)))
    else:
        table.append((rid, r.get("name"), r.get("m2"), lvl, "-", nv, "-"))
out.append("</div><h2>Tablo</h2><table><tr><th>id</th><th>ad</th><th>yazılı m²</th><th>durum</th><th>IoU (en yakın DGN)</th><th>sürüm</th><th>duvar kontrolü</th></tr>")
for row in table:
    out.append("<tr>" + "".join(f"<td>{v if v is not None else ''}</td>" for v in row) + "</tr>")
out.append("</table>")
if s.get("plan"):
    out.append(f"<h2>Qwen'in planı</h2><p>{s['plan']['understanding']}</p><ol>" + "".join(f"<li>{x}</li>" for x in s["plan"]["steps"]) + "</ol>")
    if s["plan"]["notes"]:
        out.append("<p><b>Notları:</b> " + " | ".join(s["plan"]["notes"]) + "</p>")
out.append("<h2>Dosyalar</h2><ul><li>Konuşma kaydı: SOHBET.md (bu klasörde)</li><li>Skor ayrıntısı: SKOR.html</li>"
           f"<li>Yeniden oynatılan çağrı sayısı: {n_calls}</li></ul></body></html>")
(run / "RAPOR.html").write_text("\n".join(out), encoding="utf-8")
print("wrote", run / "RAPOR.html", "| rooms", len(s.get("rooms", [])), "| walked", walked, "| score", summ)
