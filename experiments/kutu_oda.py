"""Room boundary from Qwen boxes (never 015 for development). Qwen only detects, in its native bbox_2d format, without
thinking: (1) rectangles covering the room floor, (2) column boxes, (3) end points of slanted walls. The tool builds
the polygon and snaps its edges onto the plan's own ink lines.
usage: python kutu_oda.py --rooms 11,18,7,21,14 --tag k1"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, re, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon, box as sbox
from shapely.ops import unary_union

import v0_watershed as V0  # noqa: E402
import tek_oda_test as T  # noqa: E402

T.SCHEMA = False
WHO = 'the room that contains the label "{name}"{m2}'
P_FLOOR = ("Locate the floor area of " + WHO + ": the white floor inside its walls, from inner wall face to inner wall face. "
           'Output JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "room"}}]')
P_COL = ("Detect every structural column (a small hatched or grey filled rectangle that is part of a wall) on the walls of " + WHO +
         '. Output JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "column"}}, ...] or [] if there is none.')
P_SLANT = ("Look at the walls of " + WHO + ". Most walls are horizontal or vertical. Detect only the walls of this room that are clearly "
           "SLANTED (diagonal). For each one give the box that tightly encloses that slanted wall from end to end. "
           'Output JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "slanted wall"}}, ...] or [] if no wall is slanted.')
P_MISS = ("The blue shaded outline marks the part of the floor of " + WHO + " that has been found so far. "
          "Detect every part of the floor of this same room that is NOT covered by the blue shading: white floor that continues "
          "past the blue outline without a wall in between (an extension, a niche, an L-shaped part). Do not report other rooms, "
          "balconies, wall thickness, or anything behind a wall or a door. "
          'Output JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "missing floor"}}, ...] or [] if the shading already covers the whole room.')


def boxes(txt):
    return [[float(v) for v in m] for m in re.findall(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', txt)]


def slants(txt):
    out = []
    for m in re.finditer(r"\{[^{}]*\}", txt):
        try:
            d = json.loads(m.group(0)); out.append((str(d.get("side", "")), [float(v) for v in d["p1"]], [float(v) for v in d["p2"]]))
        except Exception:
            pass
    return out


# ---------- geometry ----------
def presence(ink, a, b):
    """Fraction of the inner 80% of segment a-b that lies on ink (within 1 px across)."""
    a = np.asarray(a, float); b = np.asarray(b, float); L = np.hypot(*(b - a))
    if L < 2: return 0.0
    t = np.linspace(0.1, 0.9, max(4, int(L * 0.8)))
    P = a[None] + t[:, None] * (b - a)[None]; n = np.array([-(b - a)[1], (b - a)[0]]) / L
    hit = np.zeros(len(t), bool); H, W = ink.shape
    for o in (-1, 0, 1):
        x = np.clip(np.round(P[:, 0] + o * n[0]).astype(int), 0, W - 1); y = np.clip(np.round(P[:, 1] + o * n[1]).astype(int), 0, H - 1)
        hit |= ink[y, x]
    return float(hit.mean())


def snap(poly, ink, inside=5, outside=16, min_len=8):
    """Move every edge onto the first strong ink line met when going from a little inside the room outward."""
    c = np.array(poly.exterior.coords)[:-1]
    n = len(c); lines = []
    for i in range(n):
        a, b = c[i], c[(i + 1) % n]; d = b - a; L = np.hypot(*d)
        if L < min_len: lines.append((a, b)); continue
        nr = np.array([-d[1], d[0]]) / L
        if poly.contains(Point(*((a + b) / 2 + nr * 1.5))): nr = -nr          # nr now points out of the room
        offs = np.arange(-inside, outside + 1)
        if min(abs(d[0]), abs(d[1])) / L < 0.05:
            pr = np.array([presence(ink, a + o * nr, b + o * nr) for o in offs])
            thr = max(0.5, 0.8 * pr.max()); best = 0
            for k, o in enumerate(offs):
                if pr[k] >= thr and pr[k] >= pr[max(0, k - 1)] and pr[k] >= pr[min(len(offs) - 1, k + 1)]:
                    best = o; break
            lines.append((a + best * nr, b + best * nr))
        else:   # slanted: both ends move on their own
            so = np.arange(-outside, outside + 1); grid = np.array([[presence(ink, a + oa * nr, b + ob * nr) for ob in so] for oa in so])
            if grid.max() < 0.45: lines.append((a, b)); continue
            ia, ib = np.where(grid >= 0.85 * grid.max()); k = int(np.argmin(so[ia] + so[ib]))      # most inward of the strong lines
            lines.append((a + so[ia[k]] * nr, b + so[ib[k]] * nr))
    out = []
    for i in range(n):
        (p1, p2), (p3, p4) = lines[i - 1], lines[i]
        d1, d2 = p2 - p1, p4 - p3; cr = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cr) < 0.05 * np.hypot(*d1) * np.hypot(*d2) + 1e-9: out.append((p2 + p3) / 2); continue
        t = ((p3 - p1)[0] * d2[1] - (p3 - p1)[1] * d2[0]) / cr
        out.append(p1 + t * d1)
    P = Polygon(out)
    return P if P.is_valid and P.area > 0.5 * poly.area else poly


def best_presence(ink, p1, p2, r=14):
    """Best ink support of a line near p1-p2 (both ends may shift across by up to r px)."""
    p1 = np.array(p1, float); p2 = np.array(p2, float); d = p2 - p1; n = np.array([-d[1], d[0]]) / max(1e-6, np.hypot(*d))
    return max(presence(ink, p1 + oa * n, p2 + ob * n) for oa in range(-r, r + 1, 2) for ob in range(-r, r + 1, 2))


def align(bx, tol=9):
    """Box sides that nearly agree are made equal, so the union has no tiny steps."""
    bx = [list(b) for b in bx]
    for idx in ((0, 2), (1, 3)):
        vals = sorted((b[j], k, j) for k, b in enumerate(bx) for j in idx); grp = [vals[0]]
        for v in vals[1:] + [(1e9, -1, -1)]:
            if v[0] - grp[-1][0] <= tol: grp.append(v); continue
            m = float(np.mean([g[0] for g in grp]))
            for _, k, j in grp: bx[k][j] = m
            grp = [v]
    return bx


def refine_column(ink, b, pad=6):
    """A hatched column is a dense ink patch: its box is the dense area around Qwen's box. None if nothing dense is there."""
    H, W = ink.shape; x0, y0, x1, y1 = max(0, int(b[0] - pad)), max(0, int(b[1] - pad)), min(W, int(b[2] + pad)), min(H, int(b[3] + pad))
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    den = cv2.blur(ink[y0:y1, x0:x1].astype(np.float32), (7, 7)) >= 0.22
    nl, lab, st, _ = cv2.connectedComponentsWithStats(den.astype(np.uint8))
    if nl < 2: return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, w, h, ar = st[k]
    if ar < 40 or min(w, h) < 6: return None
    return [int(x0 + x), int(y0 + y), int(x0 + x + w), int(y0 + y + h)]


def one(g):
    return g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)


def assemble(fl, cols, sl, ink):
    U = one(unary_union([sbox(*b) for b in align(fl)]).buffer(0))
    cx, cy = U.centroid.x, U.centroid.y; far = 4000
    for side, p1, p2 in sl:
        p1, p2 = np.array(p1), np.array(p2); d = p2 - p1
        if np.hypot(*d) < 25 or min(abs(d[0]), abs(d[1])) / np.hypot(*d) < 0.06: continue    # not really slanted
        bx0, by0, bx1, by1 = U.bounds; ax = 1 if abs(d[1]) > abs(d[0]) else 0          # keep only the part of the wall beside the room
        lo, hi = (by0, by1) if ax else (bx0, bx1)
        ta = (lo - p1[ax]) / d[ax]; tb = (hi - p1[ax]) / d[ax]; ta, tb = max(0.0, min(ta, tb)), min(1.0, max(ta, tb))
        if tb - ta < 0.05: continue
        p1, p2 = p1 + ta * d, p1 + tb * d; d = p2 - p1
        if abs(d[1]) > abs(d[0]):
            out = -1 if (p1[0] + p2[0]) / 2 < cx else 1
            inner = Polygon([p1, p2, (cx, p2[1]), (cx, p1[1])]); outer = Polygon([p1, p2, (p2[0] + out * far, p2[1]), (p1[0] + out * far, p1[1])])
        else:
            out = -1 if (p1[1] + p2[1]) / 2 < cy else 1
            inner = Polygon([p1, p2, (p2[0], cy), (p1[0], cy)]); outer = Polygon([p1, p2, (p2[0], p2[1] + out * far), (p1[0], p1[1] + out * far)])
        U = one(U.union(inner.buffer(0)).difference(outer.buffer(0)).buffer(0))
    raw = U.simplify(1.5)
    S = snap(raw, ink)
    used = []
    for b in cols:
        r = refine_column(ink, b)
        if r is None: continue
        cb = sbox(*r); it = cb.intersection(S).area
        if it < 12 or it > 0.5 * S.area: continue
        S = one(S.difference(cb).buffer(0)); used.append(r)
    return raw, S, used


def run_room(im, ink, g, box, side=896, review=True):
    bw, bh = box[2] - box[0], box[3] - box[1]
    c = im.crop(box); s = side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
    m2 = f' (area text "A:{g["area_m2"]} m2")' if g.get("area_m2") and g["area_m2"] > 0 else ""
    fx = lambda x: box[0] + x / 1000 * bw; fy = lambda y: box[1] + y / 1000 * bh
    t = time.time(); txt = {}
    txt["floor"] = T.ask(c, P_FLOOR.format(name=g["name"], m2=m2), 0)[0]
    txt["col"] = T.ask(c, P_COL.format(name=g["name"], m2=m2), 0)[0]
    txt["slant"] = T.ask(c, P_SLANT.format(name=g["name"], m2=m2), 0)[0]
    sec = time.time() - t
    fl = [[fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])] for b in boxes(txt["floor"])][:4]
    cols = [[fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])] for b in boxes(txt["col"])][:12]
    sl = []
    for b in boxes(txt["slant"])[:4]:
        q = [fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])]
        d1 = ([q[0], q[1]], [q[2], q[3]]); d2 = ([q[0], q[3]], [q[2], q[1]])
        sl.append(("", *max((d1, d2), key=lambda d: best_presence(ink, *d))))
    if not fl: return None
    raw, S, used = assemble(fl, cols, sl, ink)
    first = S; miss = []
    if review:
        t = time.time()
        ov = c.convert("RGBA"); lay = Image.new("RGBA", ov.size, (0, 0, 0, 0)); dd = ImageDraw.Draw(lay)
        pp = [((x - box[0]) * s, (y - box[1]) * s) for x, y in S.exterior.coords]
        dd.polygon(pp, fill=(40, 90, 255, 70)); dd.line(pp, fill=(20, 50, 255, 255), width=3)
        txt["miss"] = T.ask(Image.alpha_composite(ov, lay).convert("RGB"), P_MISS.format(name=g["name"], m2=m2), 0)[0]
        sec += time.time() - t
        for b in boxes(txt["miss"])[:4]:
            q = [fx(b[0]), fy(b[1]), fx(b[2]), fy(b[3])]
            if sbox(*q).difference(S).area >= 150: miss.append(q)
        if miss:
            raw, S, used = assemble(fl + miss, cols, sl, ink)
    return {"sec": sec, "miss": miss, "first": first, "floor": fl, "cols": cols, "cols_used": used, "slant": sl, "raw": raw, "final": S, "txt": txt}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="019"); ap.add_argument("--rooms", default="11,18,7,21,14"); ap.add_argument("--tag", default="k1")
    a = ap.parse_args(); assert a.floor != "015"
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); W, H = im.size
    ink = np.array(im.convert("L")) < 150
    gts = V0.load_rooms(a.floor); tiles = []; rows = []
    for k, i in enumerate(int(v) for v in a.rooms.split(",")):
        g = gts[i]; box = T.crop_box(g["poly"], W, H, k)
        r = run_room(im, ink, g, box)
        if r is None: print(f"r{i}: no floor box"); continue
        s0 = T.score([list(p) for p in r["raw"].exterior.coords[:-1]], g["poly"]); s1 = T.score([list(p) for p in r["final"].exterior.coords[:-1]], g["poly"])
        rows.append({"room": i, "sec": round(r["sec"], 1), "floor_boxes": len(r["floor"]), "cols": f'{len(r["cols_used"])}/{len(r["cols"])}', "slants": len(r["slant"]), "missing": len(r["miss"]), "iou_first": T.score([list(p) for p in r["first"].exterior.coords[:-1]], g["poly"])["iou"],
                     "iou_raw": s0["iou"], "iou": s1["iou"], "corners": f'{s1["corners_hit6"]}/{s1["gt_corners"]}', "pred_corners": s1["pred_corners"],
                     "bnd_mean": s1["bnd_mean"], "bnd_p90": s1["bnd_p90"], "bnd_max": s1["bnd_max"]})
        print(rows[-1], flush=True)
        c = im.crop(box); s = 760 / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS); d = ImageDraw.Draw(c)
        f = lambda pts: [((x - box[0]) * s, (y - box[1]) * s) for x, y in pts]
        d.line(f(g["poly"].exterior.coords), fill=(0, 170, 0), width=4)
        for b in r["floor"]: d.rectangle(f([(b[0], b[1]), (b[2], b[3])]), outline=(255, 150, 0), width=1)
        for b in r["cols"]: d.rectangle(f([(b[0], b[1]), (b[2], b[3])]), outline=(255, 0, 0), width=1)
        for _, p1, p2 in r["slant"]: d.line(f([p1, p2]), fill=(255, 0, 255), width=1)
        for b in r["miss"]: d.rectangle(f([(b[0], b[1]), (b[2], b[3])]), outline=(0, 200, 200), width=2)
        d.line(f(r["final"].exterior.coords), fill=(30, 60, 255), width=2)
        d.rectangle([0, 0, c.width, 14], fill="white"); d.text((3, 1), f"r{i} {g['name']} IoU {s1['iou']} (ham {s0['iou']}) kose {s1['corners_hit6']}/{s1['gt_corners']} {r['sec']:.0f}s", fill=(200, 0, 0))
        tiles.append(c)
        json.dump({k2: (v if k2 not in ("raw", "final", "first") else list(v.exterior.coords)) for k2, v in r.items()}, open(T.OUT / f"kutu_{a.tag}_r{i}.json", "w", encoding="utf-8"), ensure_ascii=False, default=float)
    n = (len(tiles) + 2) // 3; cw = max(t.width for t in tiles) + 10; ch = max(t.height for t in tiles) + 10
    sh = Image.new("RGB", (cw * 3, ch * n), "white")
    for j, t in enumerate(tiles): sh.paste(t, ((j % 3) * cw, (j // 3) * ch))
    sh.save(T.OUT / f"sayfa_kutu_{a.tag}.png")
    print("mean IoU", round(float(np.mean([r["iou"] for r in rows])), 3), "raw", round(float(np.mean([r["iou_raw"] for r in rows])), 3), "| mean sec", round(float(np.mean([r["sec"] for r in rows])), 1))
