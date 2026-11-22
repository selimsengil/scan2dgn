"""Loss measurement of the line encoder against the image and against GT rooms (dev floors, never 015).
usage: python cizgi_olc.py --floors 019,017 [--show 019:11]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import v0_watershed as V0  # noqa: E402
import cizgi_kod as CK  # noqa: E402

OUT = config.KT / "cizgi_kod"


def measure(floor, enc, gray):
    ink = enc["ink"]; rd = CK.redraw(enc, gray.shape)
    k = np.ones((3, 3), np.uint8)
    recall = (ink & cv2.dilate(rd.astype(np.uint8), k).astype(bool)).sum() / ink.sum()
    prec = (rd & cv2.dilate(ink.astype(np.uint8), k).astype(bool)).sum() / max(1, rd.sum())
    line1 = CK.redraw(enc, gray.shape, width=1)
    dline = cv2.distanceTransform((~line1).astype(np.uint8), cv2.DIST_L2, 3)
    tree = cKDTree(CK.corner_points(enc))
    gts = [g for g in V0.load_rooms(floor) if g["poly"].geom_type == "Polygon"]
    cd, ed, rows = [], [], []
    for i, g in enumerate(gts):
        c = np.array(g["poly"].exterior.coords)[:-1]
        d = tree.query(c)[0]; cd.extend(d.tolist())
        L = g["poly"].exterior.length
        P = np.array([g["poly"].exterior.interpolate(s).coords[0] for s in np.arange(0, L, 2.0)])
        e = dline[np.clip(P[:, 1].round().astype(int), 0, gray.shape[0] - 1), np.clip(P[:, 0].round().astype(int), 0, gray.shape[1] - 1)]
        ed.extend(e.tolist()); rows.append((i, g["name"], len(c), int((d <= 2).sum()), int((d <= 4).sum()), float(d.max()), float((e <= 2).mean())))
    cd = np.array(cd); ed = np.array(ed)
    return {"floor": floor, "segments": len(enc["seg"]), "nodes": len(enc["node"]), "ink_recall": round(float(recall), 4), "redraw_precision": round(float(prec), 4),
            "gt_corners": len(cd), "corner<=2px": round(float((cd <= 2).mean()), 3), "corner<=3px": round(float((cd <= 3).mean()), 3), "corner<=5px": round(float((cd <= 5).mean()), 3),
            "edge_on_line<=2px": round(float((ed <= 2).mean()), 3)}, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floors", default="019,017"); ap.add_argument("--show", default="019:11"); ap.add_argument("--dark", type=int, default=150); ap.add_argument("--faint", type=int, default=200)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    for fl in a.floors.split(","):
        assert fl != "015"
        gray = np.array(Image.open(V0.B / fl / "clean.png").convert("L")); t = time.time()
        enc = CK.encode(gray, dark=a.dark, faint=a.faint); sec = time.time() - t
        m, rows = measure(fl, enc, gray); m["encode_sec"] = round(sec, 1); print(m, flush=True)
        worst = sorted(rows, key=lambda r: r[3] / r[2])[:6]
        for r in worst: print(f"   r{r[0]} {r[1]}: corners {r[3]}/{r[2]} within 2px ({r[4]} within 4px), worst {r[5]:.1f}px, edge on line {r[6]:.2f}")
        if a.show.split(":")[0] == fl:
            g = [q for q in V0.load_rooms(fl) if q["poly"].geom_type == "Polygon"][int(a.show.split(":")[1])]
            x0, y0, x1, y1 = [int(v) for v in g["poly"].bounds]; x0, y0, x1, y1 = max(0, x0 - 40), max(0, y0 - 40), x1 + 40, y1 + 40; s = 3
            base = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR); base = cv2.resize(base, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            base = (base * 0.35 + 255 * 0.65).astype(np.uint8)
            rng = np.random.default_rng(0)
            for (sx1, sy1, sx2, sy2) in enc["seg"]:
                if max(sx1, sx2) < x0 or min(sx1, sx2) > x1 or max(sy1, sy2) < y0 or min(sy1, sy2) > y1: continue
                col = tuple(int(v) for v in rng.integers(0, 200, 3))
                cv2.line(base, (int((sx1 - x0 + .5) * s), int((sy1 - y0 + .5) * s)), (int((sx2 - x0 + .5) * s), int((sy2 - y0 + .5) * s)), col, 2, cv2.LINE_AA)
            for cx, cy in np.array(g["poly"].exterior.coords)[:-1]:
                cv2.circle(base, (int((cx - x0 + .5) * s), int((cy - y0 + .5) * s)), 9, (0, 170, 0), 2)
            cv2.imwrite(str(OUT / f"kod_{fl}_r{a.show.split(':')[1]}.png"), base)
