"""Room boundaries for a whole dev floor (never 015): Qwen draws each room roughly (taught by solved examples, no
thinking), the line encoding cuts the plan into exact pieces, and the pieces lying inside Qwen's rough drawing become
the room. Corners come from the plan's own strokes.
usage: python oda_parca.py --floor 019 --tag p1 [--rooms 11,18] [--samples 1] [--temp 0.2] [--examples 017:18,013:0,017:15]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

import v0_watershed as V0  # noqa: E402
import tek_oda_test as T  # noqa: E402
import qwen_ornekli as QO  # noqa: E402
import cizgi_kod as CK  # noqa: E402
import cizgi_hat as CH  # noqa: E402
import cizgi_yuz as CY  # noqa: E402

OUT = config.KT / "oda_parca"


def rough(msgs, temp, think=0):
    body = {"messages": msgs, "temperature": temp, "max_tokens": 900 + think, "chat_template_kwargs": {"enable_thinking": think > 0}}
    if think > 0: body["thinking_budget_tokens"] = think
    r = json.load(urllib.request.urlopen(urllib.request.Request(QO.URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return T.parse_points(r["choices"][0]["message"].get("content") or "")[:60]


def clean(pts):
    try:
        P = Polygon(pts).buffer(0)
        if P.is_empty: return None
        return P if P.geom_type == "Polygon" else max(P.geoms, key=lambda q: q.area)
    except Exception:
        return None


def vote(polys, fs, thr=0.5, seed_xy=None):
    """Pieces whose area lies inside the rough drawings (average over the samples) by at least `thr`. The piece under the
    room's label always belongs to the room, and only pieces connected to it are kept."""
    from shapely.geometry import Point
    polys = [p for p in polys if p is not None]
    seed = [f for f in fs if seed_xy is not None and f.contains(Point(*seed_xy))]
    sel = [f for f in fs if polys and np.mean([f.intersection(p).area for p in polys]) >= thr * f.area]
    for f in seed:
        if f not in sel and (not polys or f.area <= 3.0 * max(p.area for p in polys)): sel.append(f)
    if not sel: return None, []
    U = unary_union(sel).buffer(0.01).buffer(2.0, join_style=2).buffer(-2.0, join_style=2)
    if U.geom_type != "Polygon":
        parts = list(U.geoms)
        hit = [q for q in parts if seed_xy is not None and q.contains(Point(*seed_xy))]
        U = hit[0] if hit else max(parts, key=lambda q: q.area)
    return U, sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="019"); ap.add_argument("--tag", default="p1"); ap.add_argument("--rooms", default="")
    ap.add_argument("--samples", type=int, default=1); ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--examples", default="017:18,013:0,017:15"); ap.add_argument("--side", type=int, default=896)
    a = ap.parse_args(); assert a.floor != "015"
    out = OUT / f"{a.floor}_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); gray = np.array(im.convert("L")); W, H = im.size
    t0 = time.time(); enc = CK.encode(gray); t_enc = time.time() - t0
    gts = V0.load_rooms(a.floor)
    ids = [int(v) for v in a.rooms.split(",")] if a.rooms else [i for i, g in enumerate(gts) if g["poly"].geom_type == "Polygon"]
    ex = QO.example_msgs(a.examples, 640); rows = []; tiles = []; t_model = t_geo = 0.0
    for k, i in enumerate(ids):
        g = gts[i]; box = T.crop_box(g["poly"], W, H, k); bw, bh = box[2] - box[0], box[3] - box[1]
        c = im.crop(box); s = a.side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
        c = QO.mark(c, box, g["xy"], s)
        m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
        msgs = [{"role": "system", "content": QO.SYS}] + ex + [{"role": "user", "content": [QO.img_part(c), {"type": "text", "text": "Now the new room. " + QO.Q.format(name=g["name"], m2=m2)}]}]
        t = time.time(); raws = []
        from shapely.geometry import Point
        tries = 0; rejected = 0
        while len(raws) < a.samples and tries < a.samples + 3:      # a drawing that does not contain the room's label is thrown away
            tries += 1
            pts = rough(msgs, a.temp if tries == 1 else max(a.temp, 0.6))
            fl = [[box[0] + x / 1000 * bw, box[1] + y / 1000 * bh] for x, y in pts] if len(pts) >= 3 else None
            P = clean(fl) if fl else None
            if P is not None and P.buffer(6).contains(Point(*g["xy"])): raws.append(fl)
            else: rejected += 1
        t_model += time.time() - t; t = time.time()
        fs = CY.faces(CH.lines_in(enc, box, min_len=8), box, min_area=1.0)
        U, sel = vote([clean(r) for r in raws if r], fs, seed_xy=g["xy"])
        t_geo += time.time() - t
        first = next((r for r in raws if r), None)
        final = [list(p) for p in U.exterior.coords[:-1]] if U is not None else (first or [])
        s0 = T.score(first, g["poly"]) if first else {"iou": 0.0}; s1 = T.score(final, g["poly"]) if len(final) >= 3 else {"iou": 0.0}
        rows.append({"room": i, "name": g["name"], "iou_rough": s0["iou"], "iou": s1["iou"], "corners": s1.get("corners_hit6", 0), "gt_corners": s1.get("gt_corners", len(g["poly"].exterior.coords) - 1),
                     "bnd_mean": s1.get("bnd_mean"), "pieces": len(sel), "rejected": rejected, "boundary": final})
        print({k2: v for k2, v in rows[-1].items() if k2 != "boundary"}, flush=True)
        tiles.append(T.tile(im, box, g["poly"], final if len(final) >= 3 else None, f"r{i} {g['name']} IoU {s1['iou']} (kaba {s0['iou']}) kose {s1.get('corners_hit6')}/{s1.get('gt_corners')}", size=420))
    iou = np.array([r["iou"] for r in rows]); ir = np.array([r["iou_rough"] for r in rows])
    summ = {"floor": a.floor, "rooms": len(rows), "mean_iou": round(float(iou.mean()), 3), "mean_iou_rough": round(float(ir.mean()), 3), "iou>=0.90": int((iou >= 0.9).sum()), "iou>=0.95": int((iou >= 0.95).sum()),
            "iou<0.5": int((iou < 0.5).sum()), "corners_hit": int(sum(r["corners"] for r in rows)), "gt_corners": int(sum(r["gt_corners"] for r in rows)),
            "encode_s": round(t_enc, 1), "model_s": round(t_model, 1), "geometry_s": round(t_geo, 1), "samples": a.samples}
    print(summ)
    json.dump({"summary": summ, "rooms": rows}, open(out / "SONUC.json", "w", encoding="utf-8"), ensure_ascii=False)
    cols = 6; n = (len(tiles) + cols - 1) // cols; cw = max(t.width for t in tiles) + 6; ch = max(t.height for t in tiles) + 6
    sh = Image.new("RGB", (cw * cols, ch * n), "white")
    for j, t in enumerate(tiles): sh.paste(t, ((j % cols) * cw, (j // cols) * ch))
    sh.save(out / "sayfa.png")
