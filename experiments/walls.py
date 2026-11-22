"""Wall mask v2: a wall is two parallel faces 8-30 cm apart running together >= 0.4 m.

Axis-aligned faces first (most plans): face pixels = ink kept by an opening with a long
1-D kernel. For each spacing d in the wall range, a column (row) where a horizontal
(vertical) face has a partner face d pixels away gets the band between them filled, if
the band itself is mostly not ink-free text... (the band may be empty or hatched).
Columns: hatched blocks, i.e. dense ink. Scored against the user's wall+column labels.

usage: python walls.py 015
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import v0_watershed as V0


def ink_mask(img: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink.astype(np.uint8)


def faces(ink: np.ndarray, s: float, min_len_m: float, horizontal: bool) -> np.ndarray:
    L = max(5, int(min_len_m * s))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if horizontal else (1, L))
    # bridge 1-2 px breaks in scanned lines first
    kb = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1) if horizontal else (1, 3))
    return cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kb), cv2.MORPH_OPEN, k)


def bands(face: np.ndarray, s: float, dmin_m: float, dmax_m: float, min_len_m: float, horizontal: bool) -> np.ndarray:
    """Fill between two parallel faces dmin..dmax apart that overlap >= min_len."""
    f = face if horizontal else face.T
    f = f.astype(bool)
    H, W = f.shape
    out = np.zeros_like(f)
    dmin, dmax = max(2, int(dmin_m * s)), max(3, int(dmax_m * s))
    L = max(5, int(min_len_m * s))
    kL = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    for d in range(dmin, dmax + 1):
        pair = f[:-d] & f[d:]                    # a face at row y and its partner at y+d
        pair = cv2.morphologyEx(pair.astype(np.uint8), cv2.MORPH_OPEN, kL).astype(bool)   # together >= L
        if not pair.any():
            continue
        ys, xs = np.nonzero(pair)
        # fill rows y..y+d at those columns (cumulative trick per d)
        acc = np.zeros((H + 1, W), np.int16)
        np.add.at(acc, (ys, xs), 1)
        np.add.at(acc, (ys + d + 1, xs), -1)
        out |= np.cumsum(acc, 0)[:H] > 0
    return out if horizontal else out.T


def columns(ink: np.ndarray, s: float, size_m: float = 0.18, density: float = 0.45) -> np.ndarray:
    k = max(3, int(size_m * s)) | 1
    dens = cv2.blur(ink.astype(np.float32), (k, k))
    m = (dens > density).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k // 2 | 1,) * 2)).astype(bool)


def wall_mask_v2(img: np.ndarray, s: float, dmin_m=0.07, dmax_m=0.32, min_len_m=0.4, erase=()) -> np.ndarray:
    ink = ink_mask(img)
    pad = max(1, int(0.03 * s))
    for (x0, y0, x1, y1) in erase:
        ink[max(0, int(y0) - pad):int(y1) + pad + 1, max(0, int(x0) - pad):int(x1) + pad + 1] = 0
    m = np.zeros(ink.shape, bool)
    for hz in (True, False):
        fc = faces(ink, s, min_len_m, hz)
        m |= bands(fc, s, dmin_m, dmax_m, min_len_m, hz)
    m |= columns(ink, s)
    # faces themselves belong to the wall where they bound a band
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))).astype(bool)
    return m


def grid_lines(ink: np.ndarray, span: float = 0.9, thin_px: int = 4) -> np.ndarray:
    """Axis and dimension lines: thin straight lines running across (almost) the whole
    sheet. Wall faces stop at the building."""
    H, W = ink.shape
    out = np.zeros(ink.shape, bool)
    for hz in (True, False):
        L = int((W if hz else H) * span)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if hz else (1, L))
        kb = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1) if hz else (1, 5))
        ln = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kb), cv2.MORPH_OPEN, k)
        # keep only thin ones: a line, not a band
        kt = cv2.getStructuringElement(cv2.MORPH_RECT, (1, thin_px + 1) if hz else (thin_px + 1, 1))
        thick = cv2.morphologyEx(ln, cv2.MORPH_OPEN, kt)
        out |= (ln > 0) & ~(cv2.dilate(thick, kt) > 0)
    return cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def crossing_lines(ink: np.ndarray, s: float, dmin_m=0.07, dmax_m=0.32, min_len_m=0.4,
                   long_m: float = 2.0, gap_m: float = 0.3, thin_px: int = 4) -> np.ndarray:
    """Axis lines that the sheet-span test misses (broken by text): long thin lines that
    pass THROUGH at least two perpendicular walls. A wall face stops at the wall it meets."""
    out = np.zeros(ink.shape, bool)
    vb = bands(faces(ink, s, min_len_m, False), s, dmin_m, dmax_m, min_len_m, False)
    hb = bands(faces(ink, s, min_len_m, True), s, dmin_m, dmax_m, min_len_m, True)
    for hz, perp in ((True, vb), (False, hb)):
        g = max(3, int(gap_m * s)); L = max(5, int(long_m * s))
        kb = cv2.getStructuringElement(cv2.MORPH_RECT, (g, 1) if hz else (1, g))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if hz else (1, L))
        ln = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kb), cv2.MORPH_OPEN, k)
        kt = cv2.getStructuringElement(cv2.MORPH_RECT, (1, thin_px + 1) if hz else (thin_px + 1, 1))
        thick = cv2.dilate(cv2.morphologyEx(ln, cv2.MORPH_OPEN, kt), kt) > 0
        ln = (ln > 0) & ~thick
        # the perpendicular walls' insides (their faces trimmed off)
        ke = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1) if hz else (1, 5))
        inside = cv2.erode(perp.astype(np.uint8), ke) > 0
        n, cc = cv2.connectedComponents(ln.astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        hit = cc[ln & inside]
        # count separate crossings per line: components of (line & inside)
        m2, cc2 = cv2.connectedComponents((ln & inside).astype(np.uint8), connectivity=8)
        owner = np.zeros(m2, np.int32)
        ys, xs = np.nonzero(cc2)
        owner[cc2[ys, xs]] = cc[ys, xs]
        cnt = np.bincount(owner[1:], minlength=n)
        # and it runs out of the building: past the outermost walls by >= out_m
        allw = (vb | hb)
        ys_, xs_ = np.nonzero(allw)
        bx0, bx1, by0, by1 = np.percentile(xs_, 0.5), np.percentile(xs_, 99.5), np.percentile(ys_, 0.5), np.percentile(ys_, 99.5)
        o = 0.3 * s
        _, _, st, _ = cv2.connectedComponentsWithStats(ln.astype(np.uint8), 8)
        axis = np.zeros(n, bool)
        for j in range(1, n):
            x, y, w, h, _a = st[j]
            out_of = (x < bx0 - o or x + w > bx1 + o) if hz else (y < by0 - o or y + h > by1 + o)
            axis[j] = cnt[j] >= 2 and out_of
        out |= axis[cc]
    return cv2.dilate(out.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def wall_pieces(img: np.ndarray, s: float, dmin_m=0.07, dmax_m=0.32, min_len_m=0.4, erase=(),
                lone_max_m=1.0, drop_grid=True) -> tuple[np.ndarray, list]:
    """The wall as numbered pieces: each straight horizontal band, each vertical band and
    each column block is its own piece (they overlap at junctions). Returns (pid, info):
    pid[y, x] = piece id of one piece covering the pixel (0 none), info = per piece dict.
    Pieces touching no other piece and shorter than lone_max_m are dropped (chairs, lone
    furniture). Overlaps keep every piece: masks_of() rebuilds a piece exactly."""
    ink = ink_mask(img)
    pad = max(1, int(0.03 * s))
    for (x0, y0, x1, y1) in erase:
        ink[max(0, int(y0) - pad):int(y1) + pad + 1, max(0, int(x0) - pad):int(x1) + pad + 1] = 0
    if drop_grid:
        ink[grid_lines(ink)] = 0
        ink[crossing_lines(ink, s, dmin_m, dmax_m, min_len_m)] = 0
    layers = []
    for hz in (True, False):
        layers.append(("h" if hz else "v", bands(faces(ink, s, min_len_m, hz), s, dmin_m, dmax_m, min_len_m, hz)))
    layers.append(("c", columns(ink, s)))
    comps, info = [], []
    for kind, m in layers:
        n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        comps.append((kind, cc, n, st))
    union = np.zeros(ink.shape, np.int32)   # how many pieces cover a pixel
    for kind, cc, n, st in comps:
        union += cc > 0
    pid = np.zeros(ink.shape, np.int32)
    k = 0
    keep_cc = []
    for kind, cc, n, st in comps:
        lut = np.zeros(n, np.int32)
        for j in range(1, n):
            x, y, w, h, a = st[j]
            ext = max(w, h) / s
            sub = cc[y:y + h, x:x + w] == j
            touches = (union[y:y + h, x:x + w][sub] > 1).any()
            if not touches and ext < lone_max_m:
                continue
            k += 1; lut[j] = k
            info.append({"id": k, "kind": kind, "bbox": [int(x), int(y), int(x + w), int(y + h)], "len_m": round(float(ext), 2),
                         "layer": len(keep_cc), "j": int(j)})
        keep_cc.append((lut, cc))
        pid = np.where((pid == 0) & (lut[cc] > 0), lut[cc], pid)
    return pid, info, keep_cc


def mask_without(keep_cc, deleted: set) -> np.ndarray:
    m = None
    for lut, cc in keep_cc:
        l2 = lut.copy()
        if deleted:
            l2[np.isin(l2, list(deleted))] = 0
        part = l2[cc] > 0
        m = part if m is None else (m | part)
    return m


def gt_struct(floor: str, shape) -> np.ndarray:
    from shapely.geometry import shape as shp
    L = json.load(open(V0.B / floor / "LABELS_COMMON.json", encoding="utf-8"))
    m = np.zeros(shape, bool)
    for st in L["structures"]:
        if st.get("kind") in ("wall", "column"):
            try:
                m |= V0.poly_mask(shp(st["geometry"]), shape)
            except Exception:
                pass
    return m


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else "015"
    img = cv2.imread(str(V0.B / f / "clean.png"), cv2.IMREAD_GRAYSCALE)
    s = V0.oracle_scale(V0.load_rooms(f)) if len(sys.argv) < 3 else float(sys.argv[2])
    out = Path(__file__).with_name("out_walls"); out.mkdir(exist_ok=True)
    m = wall_mask_v2(img, s)
    gt = gt_struct(f, img.shape)
    # the labels cover the building only: score inside the labels' bounding box
    ys, xs = np.nonzero(gt); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    roi = np.zeros_like(gt); roi[y0:y1, x0:x1] = True
    tp = (m & gt & roi).sum(); fp = (m & ~gt & roi).sum(); fn = (~m & gt & roi).sum()
    print(f, "scale", round(s, 1), "wall precision %.3f recall %.3f IoU %.3f" % (tp / (tp + fp), tp / (tp + fn), tp / (tp + fp + fn)))
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) // 2 + 127
    vis[m & gt] = (60, 160, 60); vis[m & ~gt] = (40, 40, 220); vis[~m & gt] = (220, 120, 40)
    sc = min(1.0, 2400 / max(img.shape))
    cv2.imwrite(str(out / f"{f}_walls.png"), cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main()
