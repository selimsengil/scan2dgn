"""DECISION test (dev floors, never 015): the model works in the encoding's own words instead of typing coordinates.
  step 1  rough polygon of the room (as gorunum_test; --rough-from reuses a saved run, so only step 2 is paid for)
  step 2  the plan's pieces are grouped into ZONES (pieces with nothing or only a faint line between them). The zones at
          the edge of the rough drawing are painted and numbered on the picture and the model says for each one:
          room (floor of this room) / wall (wall band, column, window band) / other (another room, outside).
  result  zones called "room" + the undisputed inside; the outline is the pieces' own exact outline.
--oracle answers step 2 from the labels: the ceiling of this design, no model call.
usage: python karar_test.py --rough-from kat_test/gorunum/019_scan_q4_t1 --tag k1 [--oracle] [--prompt kisa|uzun] [--port 8091]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, re, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point
from shapely.ops import unary_union

import v0_watershed as V0, tek_oda_test as T, qwen_ornekli as QO, oda_parca as OP, cizgi_kod as CK, cizgi_hat as CH, cizgi_yuz as CY  # noqa: E402
import kod_paket as KP, kod_bolge as KB, sinir_skor as SS  # noqa: E402

OUT = config.KT / "karar"
COL = [(230, 25, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (0, 150, 136), (240, 50, 230), (128, 128, 0), (170, 110, 40),
       (0, 0, 128), (128, 0, 0), (60, 180, 75), (70, 240, 240), (255, 105, 180), (100, 100, 100), (210, 180, 0), (0, 90, 60)]

KISA = ("You check a room boundary on a scanned architectural floor plan. The room in question has a red dot. "
        "Some regions of the plan are painted and numbered. For every number say what the region is: "
        '"room" = floor of the red-dot room, "wall" = wall, column or window band, "other" = another room or outside. '
        'JSON only: {"1": "room", "2": "wall", ...}')

UZUN = """You help an architecture office trace room boundaries on scanned floor plans. The office draws a room's boundary on the
INNER face of its walls: the line where this room's floor meets the wall. You do not type coordinates. A program has
already cut the plan along its drawn lines into exact regions; your job is to JUDGE regions, the program measures.

What you see: a part of the plan. The room in question has a RED DOT on its label. Its sure floor area is tinted light
green. Around it, the regions the program is unsure about are painted in colours and carry a NUMBER.

How to work, in this order:
1. Find the room: read its label at the red dot, then find the walls around it. Walls are bands between two parallel
   dark lines (often about 10-25 px thick here), drawn all around the room. Columns are filled or hatched rectangles
   sitting in or at the walls.
2. For each numbered region decide ONE word:
   "room"  - it is floor of the red-dot room. This includes: floor under furniture, counters, sinks, bath tubs, toilets;
             the area swept by a door leaf that opens into this room; niches and extensions not separated by a wall;
             floor covered by a stamp, a dimension line, a number or a text.
   "wall"  - it is not floor: a wall band (the strip BETWEEN the two lines of a wall), a column, a window band in the
             wall, a shaft. A long thin strip along the edge of the green area is almost always the wall itself.
   "other" - floor of a different room (it has its own label or lies behind a wall or a door), a balcony, a corridor
             behind a door, or the outside.
3. Doors: the room ends at the door opening, on the line that continues the inner wall face. What lies beyond the
   opening is "other", the wall thickness inside the opening (the threshold strip) is "wall".
4. Do not let thin lines fool you: tile grids, dimension lines, furniture outlines and stamps are not walls. A region
   bounded only by such lines and lying inside the walls is "room".
5. If a region is partly wall and partly floor, answer by its larger part.

Answer with JSON only, every number once, no other text: {"1": "room", "2": "wall", "3": "other"}"""


def zone_list(fs, bord):
    """Zones: pieces joined across faint lines and across short stretches where nothing is drawn."""
    par = list(range(len(fs))); strip = [KB.strip_like(f) for f in fs]

    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for i, j, L, ink, g in bord:
        if strip[i] != strip[j]: continue
        if (ink >= 0.5 and g >= 160 and not strip[i]) or (ink < 0.25 and L < 28): par[find(i)] = find(j)
    gr = {}
    for i in range(len(fs)): gr.setdefault(find(i), []).append(i)
    return list(gr.values())


def picture(im, box, side, xy, sure, cands):
    c = im.crop(box).convert("RGBA"); s = side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    ov = Image.new("RGBA", c.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    tr = lambda P: [((x - box[0]) * s, (y - box[1]) * s) for x, y in P.exterior.coords]
    for P in ([sure] if sure is not None and sure.geom_type == "Polygon" else list(getattr(sure, "geoms", []))):
        d.polygon(tr(P), fill=(120, 220, 120, 70))
    for k, Z in enumerate(cands):
        for P in ([Z] if Z.geom_type == "Polygon" else list(Z.geoms)): d.polygon(tr(P), fill=COL[k % len(COL)] + (110,), outline=COL[k % len(COL)] + (255,))
    c = Image.alpha_composite(c, ov).convert("RGB"); d = ImageDraw.Draw(c)
    try: font = ImageFont.truetype("arialbd.ttf", 17)
    except Exception: font = ImageFont.load_default()
    for k, Z in enumerate(cands):
        p = Z.representative_point(); x, y = (p.x - box[0]) * s, (p.y - box[1]) * s; t = str(k + 1); w = d.textlength(t, font=font)
        d.rectangle([x - w / 2 - 3, y - 10, x + w / 2 + 3, y + 10], fill="white", outline=COL[k % len(COL)], width=2); d.text((x - w / 2, y - 9), t, fill=COL[k % len(COL)], font=font)
    return QO.mark(c, box, xy, s)


def ask(url, sysmsg, img, name, m2, n, temp, model):
    q = f'The room with the red dot: "{name}"{m2}. Regions are numbered 1 to {n}. Give the JSON.'
    body = {"messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": [QO.img_part(img), {"type": "text", "text": q}]}], "temperature": temp, "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    if model: body["model"] = model
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    txt = r["choices"][0]["message"].get("content") or ""
    out = {int(k): v.lower() for k, v in re.findall(r'"(\d+)"\s*:\s*"(\w+)"', txt)}
    return out, r.get("usage", {}), txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rough-from", required=True); ap.add_argument("--tag", default="k1"); ap.add_argument("--floor", default="019"); ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--prompt", default="uzun", choices=["kisa", "uzun"]); ap.add_argument("--port", type=int, default=8091); ap.add_argument("--model-name", default="")
    ap.add_argument("--side", type=int, default=1024); ap.add_argument("--max-zones", type=int, default=14); ap.add_argument("--min-area", type=float, default=150.0); ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--small", type=float, default=0.05)
    a = ap.parse_args(); assert a.floor != "015"
    out = OUT / f"{a.floor}_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); H, W = gray.shape; enc = CK.encode(gray); gts = V0.load_rooms(a.floor)
    src = json.load(open(Path(a.rough_from) / "SONUC.json", encoding="utf-8"))["rooms"]; rows = []; t_model = 0.0; tok = []; agree = tot = 0
    for k, r in enumerate(src):
        g = gts[r["room"]]; box = T.crop_box(g["poly"], W, H, k); P = OP.clean(r["rough"]) if len(r.get("rough") or []) >= 3 else None
        if P is None: rows.append({"room": r["room"], "name": g["name"], "boundary": []}); continue
        fs = CY.faces(CH.lines_in(enc, box, min_len=8), box, min_area=1.0, long_reach=230, max_gap=230); bord = KP.border_info(fs, enc, gray)
        zs = zone_list(fs, bord); zp = [unary_union([fs[i] for i in z]).buffer(0.01) for z in zs]
        frac = [z.intersection(P).area / max(z.area, 1e-6) for z in zp]; seed = [i for i, z in enumerate(zp) if z.contains(Point(*g["xy"]))]
        inside = set(i for i, f in enumerate(frac) if f >= 0.5) | set(seed)
        U0 = unary_union([zp[i] for i in inside]).buffer(1.5) if inside else P; ring = U0.boundary.buffer(4.0)
        # doubtful: big enough, at the edge of the default selection (inside or outside it), never the labelled zone itself
        cand = [i for i, z in enumerate(zp) if z.area >= a.min_area and i not in seed and z.intersects(ring) and z.area < 0.8 * max(P.area, 1)]
        hull = unary_union([zp[i] for i in inside]).buffer(1.5).buffer(0) if inside else P
        hull = hull if hull.geom_type == "Polygon" else max(hull.geoms, key=lambda q: q.area)
        outer = hull.exterior.buffer(4.0)                                  # a region enclosed by sure floor is floor: not asked
        zstrip = [KB.strip_like(z) for z in zp]
        cand = [i for i in cand if zp[i].intersects(outer) and (zstrip[i] or zp[i].area < a.small * P.area)]   # wall-like or small: where the errors are
        cand = sorted(cand, key=lambda i: -zp[i].area)[: a.max_zones]
        sure = unary_union([zp[i] for i in inside if i not in cand]) if [i for i in inside if i not in cand] else None
        truth = {n + 1: ("room" if zp[i].intersection(g["poly"]).area >= 0.5 * zp[i].area else "notroom") for n, i in enumerate(cand)}
        pic = picture(im, box, a.side, g["xy"], sure, [zp[i] for i in cand]); pic.save(out / f"r{r['room']}_soru.png")
        if a.oracle: ans = {n: ("room" if v == "room" else "wall") for n, v in truth.items()}; raw = ""
        else:
            m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""; t = time.time()
            ans, use, raw = ask(f"http://127.0.0.1:{a.port}/v1/chat/completions", UZUN if a.prompt == "uzun" else KISA, pic, g["name"], m2, len(cand), a.temp, a.model_name)
            t_model += time.time() - t; tok.append(use.get("prompt_tokens", 0))
        keep = set(inside)
        for n, i in enumerate(cand):
            v = ans.get(n + 1)
            if v == "room": keep.add(i)
            elif v in ("wall", "other"): keep.discard(i)
            if v is not None: tot += 1; agree += (v == "room") == (truth[n + 1] == "room")
        U = unary_union([zp[i] for i in keep]).buffer(0.01).buffer(2.0, join_style=2).buffer(-2.0, join_style=2) if keep else None
        if U is not None and U.geom_type != "Polygon":
            hit = [q for q in U.geoms if q.contains(Point(*g["xy"]))]; U = hit[0] if hit else max(U.geoms, key=lambda q: q.area)
        rows.append({"room": r["room"], "name": g["name"], "boundary": [list(p) for p in U.simplify(0.4).exterior.coords[:-1]] if U is not None else [], "answers": ans, "truth": truth, "raw": raw})
        print(f"r{r['room']} {g['name']}: zones {len(zp)}, asked {len(cand)}, answers {ans}", flush=True)
    json.dump({"rooms": rows}, open(out / "SONUC.json", "w", encoding="utf-8"), ensure_ascii=False)
    SS.score_run(out, a.floor, 3.0, quiet=False)
    print("DECISIONS right", agree, "/", tot, "| seconds per room", round(t_model / max(1, len(rows)), 1), "| prompt tokens", int(np.mean(tok)) if tok else 0)
