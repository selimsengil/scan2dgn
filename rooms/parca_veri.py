"""Pieces of a labelled floor as a training table (no model here). Floors scanned finer than 130 px/m are first reduced
to about 100 px/m, so one metre looks alike everywhere; every length feature is in metres.
Writes kat_test/parca_veri/<floor>.npz (features X, class y, room index, representative points, borders) and
<floor>.pkl (the piece polygons, in the reduced image's pixels) + meta.
classes: 0 unlabelled, 1 room floor, 2 wall, 3 column, 4 door opening, 5 window.
usage: python parca_veri.py --floor 013 [--scale 155]   (--scale px/m is needed only for floors without room areas)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, pickle, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import shape

import v0_watershed as V0, cizgi_kod as CK, cizgi_yuz as CY, kod_paket as KP  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = config.KT / "parca_veri"
KIND = {"wall": 2, "column": 3, "door": 4, "window": 5}
NAMES = ["area_m2", "perim_m", "thick_m", "rect_short_m", "rect_long_m", "aspect", "convexity", "n_vert", "gray_mean", "ink_frac", "dark_frac",
         "bord_len_m", "drawn_share", "drawn_gray", "cut_share", "n_nb", "nb_max_area_m2", "nb_thick_m", "nb_area_m2",
         "ink_r015", "ink_r04", "ink_r1", "ink_r25", "gray_r04", "gray_r1", "dist_ink_m"]


def floor_scale(floor):
    r = [g for g in V0.load_rooms(floor) if g["area_m2"] and g["area_m2"] > 0 and g["poly"].geom_type == "Polygon"]
    return float(np.median([(g["poly"].area / g["area_m2"]) ** 0.5 for g in r])) if r else None


def ring(P, r=1.0): return np.round(np.array(P.exterior.coords) * r).astype(np.int32)


def extract(floor, scale=None, target=100.0):
    t0 = time.time(); s = scale or floor_scale(floor); assert s, "give --scale for a floor without room areas"
    r = target / s if s > 130 else 1.0; se = s * r
    im = Image.open(V0.B / floor / "clean.png").convert("L"); g0 = np.array(im)
    gray = cv2.resize(g0, (int(round(g0.shape[1] * r)), int(round(g0.shape[0] * r))), interpolation=cv2.INTER_AREA) if r != 1.0 else g0
    H, W = gray.shape; reach = 2.85 * se
    enc, lines = KP.build(gray); fs = CY.faces(lines, (0, 0, W, H), min_area=1.0, long_reach=reach, max_gap=reach)
    fs = [f for f in fs if f.bounds[0] > 0.5 and f.bounds[1] > 0.5 and f.bounds[2] < W - 0.5 and f.bounds[3] < H - 0.5]
    bord = KP.border_info(fs, enc, gray); n = len(fs); print(f"{floor}: scale {s:.1f} -> {se:.1f} px/m, {W}x{H}, pieces {n}, borders {len(bord)}, {time.time() - t0:.0f}s", flush=True)
    # id map (big pieces first, so pieces lying in a hole of another are drawn over it)
    idm = np.zeros((H, W), np.int32)
    for i in sorted(range(n), key=lambda i: -fs[i].area): cv2.fillPoly(idm, [ring(fs[i])], i + 1)
    L = json.load(open(V0.B / floor / "LABELS_COMMON.json", encoding="utf-8")); cls = np.zeros((H, W), np.uint8); rid = np.zeros((H, W), np.int32)
    rooms = V0.load_rooms(floor)
    for k, g in enumerate(rooms):
        P = affinity.scale(g["poly"], r, r, origin=(0, 0))
        for Q in ([P] if P.geom_type == "Polygon" else list(P.geoms)): cv2.fillPoly(cls, [ring(Q)], 1); cv2.fillPoly(rid, [ring(Q)], k + 1)
    for kind in ("wall", "column", "door", "window"):
        for st in L["structures"]:
            if st["kind"] != kind: continue
            P = affinity.scale(shape(st["geometry"]).buffer(0), r, r, origin=(0, 0))
            for Q in ([P] if P.geom_type == "Polygon" else list(getattr(P, "geoms", []))):
                if Q.geom_type == "Polygon" and not Q.is_empty: cv2.fillPoly(cls, [ring(Q)], KIND[kind]); cv2.fillPoly(rid, [ring(Q)], 0)
    flat = idm.ravel(); npx = np.bincount(flat, minlength=n + 1).astype(np.float64)
    ink = enc["ink"].astype(np.float32); gsum = np.bincount(flat, weights=gray.ravel(), minlength=n + 1); isum = np.bincount(flat, weights=ink.ravel(), minlength=n + 1)
    dsum = np.bincount(flat, weights=(gray.ravel() < 100), minlength=n + 1)
    cc = np.bincount(flat.astype(np.int64) * 6 + cls.ravel(), minlength=(n + 1) * 6).reshape(n + 1, 6)
    rep = np.array([f.representative_point().coords[0] for f in fs]); rx = np.clip(rep[:, 0].round().astype(int), 0, W - 1); ry = np.clip(rep[:, 1].round().astype(int), 0, H - 1)
    y = np.zeros(n, np.int8); room = np.full(n, -1, np.int32)
    for i in range(n):
        c = cc[i + 1]; tot = c.sum()
        if tot < 1: y[i] = cls[ry[i], rx[i]]
        else:
            k = int(np.argmax(c[1:])) + 1; y[i] = k if c[k] >= 0.5 * tot else 0
        if y[i] == 1: room[i] = rid[ry[i], rx[i]] - 1
    X = np.zeros((n, len(NAMES)), np.float32); m = 1.0 / se
    for i, f in enumerate(fs):
        rr = f.minimum_rotated_rectangle; e = (0.0, 0.0)
        if rr.geom_type == "Polygon":
            c = np.array(rr.exterior.coords); e = sorted((float(np.hypot(*(c[1] - c[0]))), float(np.hypot(*(c[2] - c[1])))))
        px = max(npx[i + 1], 1.0)
        X[i, :11] = [f.area * m * m, f.exterior.length * m, 2 * f.area / max(f.exterior.length, 1e-6) * m, e[0] * m, e[1] * m, e[1] / max(e[0], 1e-3), f.area / max(f.convex_hull.area, 1e-6),
                     len(f.exterior.coords) - 1, gsum[i + 1] / px if npx[i + 1] else gray[ry[i], rx[i]], isum[i + 1] / px, dsum[i + 1] / px]
    B = np.array(bord, np.float64).reshape(-1, 5); area = np.array([f.area for f in fs]); thick = X[:, 2].astype(np.float64)
    bl = np.zeros(n); dr = np.zeros(n); dg = np.zeros(n); cut = np.zeros(n); nn = np.zeros(n); nmax = np.zeros(n); nth = np.zeros(n); nar = np.zeros(n)
    for a_, b_ in ((0, 1), (1, 0)):
        i = B[:, a_].astype(int); j = B[:, b_].astype(int); Ln = B[:, 2]; inkv = B[:, 3]; gv = B[:, 4]
        np.add.at(bl, i, Ln); np.add.at(dr, i, Ln * inkv); np.add.at(dg, i, Ln * inkv * gv); np.add.at(cut, i, Ln * (inkv < 0.25)); np.add.at(nn, i, 1)
        np.maximum.at(nmax, i, area[j]); np.add.at(nth, i, Ln * thick[j]); np.add.at(nar, i, Ln * np.log1p(area[j] * m * m))
    bz = np.maximum(bl, 1e-6); X[:, 11] = bl * m; X[:, 12] = dr / bz; X[:, 13] = np.where(dr > 0, dg / np.maximum(dr, 1e-6), 255); X[:, 14] = cut / bz; X[:, 15] = nn
    X[:, 16] = nmax * m * m; X[:, 17] = nth / bz; X[:, 18] = nar / bz
    for col, rad in ((19, 0.15), (20, 0.4), (21, 1.0), (22, 2.5)):
        k = max(3, int(rad * se) * 2 + 1); X[:, col] = cv2.blur(ink, (k, k))[ry, rx]
    for col, rad in ((23, 0.4), (24, 1.0)):
        k = max(3, int(rad * se) * 2 + 1); X[:, col] = cv2.blur(gray.astype(np.float32), (k, k))[ry, rx]
    X[:, 25] = cv2.distanceTransform((~enc["ink"]).astype(np.uint8), cv2.DIST_L2, 3)[ry, rx] * m
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{floor}.npz", X=X, y=y, room=room, rep=rep.astype(np.float32), bord=B.astype(np.float32), area=area.astype(np.float32))
    pickle.dump({"wkb": [f.wkb for f in fs], "meta": {"floor": floor, "scale": s, "resize": r, "px_per_m": se, "size": [W, H], "names": NAMES}}, open(OUT / f"{floor}.pkl", "wb"))
    cv2.imwrite(str(OUT / f"{floor}.png"), gray)
    print(f"{floor}: classes {dict(zip(*np.unique(y, return_counts=True)))} | total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floor", required=True); ap.add_argument("--scale", type=float, default=None); a = ap.parse_args()
    assert a.floor != "015", "015 is the blind test; extract it only for the final run"
    extract(a.floor, a.scale)
