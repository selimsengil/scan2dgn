"""Second stage on the encoded view (dev floors, never 015): the room found so far is shown in blue on the plan, the
groups of pieces touching it are shown coloured and numbered, and the model says which of them are also this room's
floor. The model picks among exact pieces; it writes no coordinates.
usage: python oda_duzelt.py --from 019_p3 --tag d1 [--rooms 21,24] [--think 0]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys, time, urllib.request, re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import v0_watershed as V0  # noqa: E402
import tek_oda_test as T  # noqa: E402
import qwen_ornekli as QO  # noqa: E402
import cizgi_kod as CK  # noqa: E402
import cizgi_hat as CH  # noqa: E402
import cizgi_yuz as CY  # noqa: E402
import oda_parca as OP  # noqa: E402

COLORS = [(255, 140, 0), (0, 170, 80), (200, 0, 200), (0, 160, 200), (150, 110, 0), (230, 60, 60), (110, 60, 220), (0, 130, 130), (170, 170, 0), (90, 90, 90)]
PROMPT = """This floor plan part shows one room being outlined. The BLUE area is the part of the room found so far; the red dot is the room's label ("{name}"{m2}).
Around the blue area there are coloured regions, each with a number in a white box: {nums}.

For each numbered region decide: is it floor of the SAME room as the blue area?
- YES if it is the same space continuing: nothing but a thin drawn line (floor tiles, a dimension line, a fixture, a shower tray, a counter, a door swing arc) lies between it and the blue area.
- NO if a wall (a thick double-line band), a door or a window separates it from the blue area, or if it is a column, a wall, a shaft, a balcony or another room (another room has its own label).

Answer with JSON only: {{"same_room": [numbers that are YES]}}"""


def thin(f):
    return 2.0 * f.area / max(f.exterior.length, 1e-6)


def strip_like(f, width=21.0):
    """A wall band or a window band: long and narrow."""
    r = f.minimum_rotated_rectangle
    if r.geom_type != "Polygon": return True
    c = np.array(r.exterior.coords); e = sorted((float(np.hypot(*(c[1] - c[0]))), float(np.hypot(*(c[2] - c[1])))))
    return thin(f) < 6.0 or (e[0] <= width and e[1] >= 2.5 * e[0])


def groups(fs, region_idx, small=700.0, min_area=150.0, max_groups=9):
    """What touches the region: every unselected piece next to it is one candidate. Only small pieces (tile-sized) pull
    in their small neighbours, so a tiled floor is one candidate instead of forty. Wall-like strips are never candidates."""
    R = unary_union([fs[i] for i in region_idx]).buffer(1.2)
    free = [i for i in range(len(fs)) if i not in region_idx and not strip_like(fs[i])]
    touch = [i for i in free if fs[i].intersects(R) and fs[i].intersection(R).area >= 8.0]
    smalls = [i for i in free if fs[i].area < small]; buf = {i: fs[i].buffer(1.2) for i in smalls}
    seen = set(); out = []
    for s in sorted(touch, key=lambda i: -fs[i].area):
        if s in seen: continue
        grp = [s]; seen.add(s)
        if fs[s].area < small:
            q = [s]
            while q:
                c = q.pop()
                for j in smalls:
                    if j not in seen and buf[c].intersects(fs[j]) and buf[c].intersection(fs[j]).area >= 4.0: seen.add(j); grp.append(j); q.append(j)
        if sum(fs[i].area for i in grp) >= min_area: out.append(grp)
    out.sort(key=lambda g: -sum(fs[i].area for i in g))
    return out[:max_groups]


def picture(im, box, region, grps, fs, xy, side=896):
    c = im.crop(box).convert("RGBA"); s = side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    lay = Image.new("RGBA", c.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    f = lambda poly: [((x - box[0]) * s, (y - box[1]) * s) for x, y in poly.exterior.coords]
    for part in (region.geoms if region.geom_type != "Polygon" else [region]):
        d.polygon(f(part), fill=(40, 90, 255, 95), outline=(20, 50, 255, 255))
    for k, g in enumerate(grps):
        col = COLORS[k % len(COLORS)]
        for i in g: d.polygon(f(fs[i]), fill=col + (120,), outline=col + (255,))
    c = Image.alpha_composite(c, lay).convert("RGB"); d = ImageDraw.Draw(c)
    try: font = ImageFont.truetype("arialbd.ttf", 22)
    except Exception: font = ImageFont.load_default()
    for k, g in enumerate(grps):
        p = max((fs[i] for i in g), key=lambda q: q.area).representative_point(); x, y = (p.x - box[0]) * s, (p.y - box[1]) * s
        t = str(k + 1); w = 14 * len(t) + 8
        d.rectangle([x - w / 2, y - 14, x + w / 2, y + 14], fill="white", outline=COLORS[k % len(COLORS)], width=3); d.text((x - w / 2 + 4, y - 12), t, fill=(0, 0, 0), font=font)
    return QO.mark(c, box, xy, s)


def ask(img, prompt, think):
    body = {"messages": [{"role": "user", "content": [QO.img_part(img), {"type": "text", "text": prompt}]}], "temperature": 0.1, "max_tokens": 200 + think,
            "chat_template_kwargs": {"enable_thinking": think > 0}}
    if think > 0: body["thinking_budget_tokens"] = think
    r = json.load(urllib.request.urlopen(urllib.request.Request(QO.URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    txt = r["choices"][0]["message"].get("content") or ""
    m = re.search(r'"same_room"\s*:\s*\[([^\]]*)\]', txt)
    return [int(v) for v in re.findall(r"\d+", m.group(1))] if m else [], txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="019_p3"); ap.add_argument("--tag", default="d1"); ap.add_argument("--rooms", default="")
    ap.add_argument("--think", type=int, default=0); ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args(); floor = a.src.split("_")[0]; assert floor != "015"
    src = json.load(open(OP.OUT / a.src / "SONUC.json", encoding="utf-8"))["rooms"]
    out = OP.OUT / f"{a.src}_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    im = Image.open(V0.B / floor / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); W, H = im.size
    enc = CK.encode(gray); gts = V0.load_rooms(floor)
    want = [int(v) for v in a.rooms.split(",")] if a.rooms else None
    rows = []; tiles = []; t0 = time.time()
    for k, r in enumerate(src):
        i = r["room"]
        if want and i not in want: continue
        g = gts[i]; box = T.crop_box(g["poly"], W, H, k); fs = CY.faces(CH.lines_in(enc, box, min_len=8), box)
        P = OP.clean(r["boundary"]) if len(r["boundary"]) >= 3 else None
        region_idx = set(j for j, f in enumerate(fs) if P is not None and f.intersection(P).area >= 0.5 * f.area)
        region_idx |= set(j for j, f in enumerate(fs) if f.contains(Point(*g["xy"])))
        m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
        added = []; asked = 0
        for rnd in range(a.rounds):
            if not region_idx: break
            grps = groups(fs, region_idx)
            if not grps: break
            region = unary_union([fs[j] for j in region_idx]).buffer(0.01)
            pic = picture(im, box, region, grps, fs, g["xy"])
            if rnd == 0: pic.save(out / f"r{i}_soru.png")
            yes, txt = ask(pic, PROMPT.format(name=g["name"], m2=m2, nums=", ".join(str(n + 1) for n in range(len(grps)))), a.think); asked += len(grps)
            yes = [n for n in yes if 1 <= n <= len(grps)]
            if not yes: break
            for n in yes: region_idx |= set(grps[n - 1]); added.append(n)
        if region_idx:
            U = unary_union([fs[j] for j in region_idx]).buffer(0.01)
            if U.geom_type != "Polygon":
                hit = [q for q in U.geoms if q.contains(Point(*g["xy"]))]; U = hit[0] if hit else max(U.geoms, key=lambda q: q.area)
            final = [list(p) for p in U.exterior.coords[:-1]]
        else: final = r["boundary"]
        sc = T.score(final, g["poly"]) if len(final) >= 3 else {"iou": 0.0}
        rows.append({"room": i, "name": g["name"], "iou_before": r["iou"], "iou": sc["iou"], "corners": sc.get("corners_hit6", 0), "gt_corners": r["gt_corners"], "asked": asked, "added": added, "boundary": final})
        print({k2: v for k2, v in rows[-1].items() if k2 != "boundary"}, flush=True)
        tiles.append(T.tile(im, box, g["poly"], final if len(final) >= 3 else None, f"r{i} {g['name']} IoU {sc['iou']} (once {r['iou']})", size=420))
    b = np.array([x["iou_before"] for x in rows]); n = np.array([x["iou"] for x in rows])
    summ = {"rooms": len(rows), "mean_iou_before": round(float(b.mean()), 3), "mean_iou": round(float(n.mean()), 3), "iou>=0.90": int((n >= 0.9).sum()), "iou>=0.95": int((n >= 0.95).sum()), "iou<0.5": int((n < 0.5).sum()),
            "better": int((n > b + 0.02).sum()), "worse": int((n < b - 0.02).sum()), "corners_hit": int(sum(x["corners"] for x in rows)), "gt_corners": int(sum(x["gt_corners"] for x in rows)), "seconds": round(time.time() - t0, 1)}
    print(summ)
    json.dump({"summary": summ, "rooms": rows}, open(out / "SONUC.json", "w", encoding="utf-8"), ensure_ascii=False)
    cols = 6; nr = (len(tiles) + cols - 1) // cols; cw = max(t.width for t in tiles) + 6; ch = max(t.height for t in tiles) + 6
    sh = Image.new("RGB", (cw * cols, ch * nr), "white")
    for j, t in enumerate(tiles): sh.paste(t, ((j % cols) * cw, (j // cols) * ch))
    sh.save(out / "sayfa.png")
