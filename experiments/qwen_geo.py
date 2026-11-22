"""Geometry core for Qwen's tools: wall mask, seeded room cells, polygon fitting, checks.

Wall model (scanned Turkish municipal plans): walls are drawn as two parallel dark lines (faces) with a light core, or as
solid/hatched bands; furniture, text, dimension and axis lines are single thin strokes.
  1. ink        : Otsu threshold
  2. long lines : morphological opening with long horizontal / vertical / diagonal line elements (drops text, furniture)
  3. bands      : closing across the line direction fills the core between two faces -> one solid band per wall
  4. wall mask  : bands thicker than ~4 px (single thin lines vanish) + solid dark blobs (columns, hatched walls)
Room cells: watershed on the free space's distance transform, one marker per room seed (+ 'outside' markers);
door gaps become the narrow necks where neighbouring basins meet, so a door is closed near the jamb line.
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import math

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid


def load_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im


def ink_mask(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = min(max(t, 120), 200)
    return (gray < t).astype(np.uint8) * 255


def wall_mask(gray, px_per_cm=0.8, close_m=0.24, open_m=0.07, text_m=0.45, minpiece_m=1.2):
    """v0 rule (v0_watershed.py): drop text-sized ink pieces, close across wall cores (h and v), open away thin lines,
    keep pieces at least minpiece long."""
    s = px_per_cm * 100.0   # px per metre
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n0, cc0, st0, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    small = np.maximum(st0[:, cv2.CC_STAT_WIDTH], st0[:, cv2.CC_STAT_HEIGHT]) < text_m * s
    small[0] = False
    ink[small[cc0]] = 0
    kc = max(3, int(round(close_m * s))) | 1
    ko = max(3, int(round(open_m * s))) | 1
    cv_ = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kc)))
    ch_ = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kc, 1)))
    walls = cv2.morphologyEx(np.maximum(cv_, ch_), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko)))
    n, cc, st, _ = cv2.connectedComponentsWithStats((walls > 0).astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= minpiece_m * s
    return np.where(keep[cc], 255, 0).astype(np.uint8)


def wall_mask_lines(gray, px_per_cm=0.8, min_len_cm=40, max_core_cm=30):
    """Experimental line-based variant (worse on 019 so far); kept for comparison."""
    ink = ink_mask(gray)
    L = max(15, int(min_len_cm * px_per_cm))
    core = max(7, int(max_core_cm * px_per_cm))
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    # diagonal lines (oblique facades): rotate, open, rotate back
    dl = np.zeros_like(ink)
    h, w = ink.shape
    for ang in (-60, -45, -30, -15, 15, 30, 45, 60):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        r = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST)
        r = cv2.morphologyEx(r, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, core)))
        Mi = cv2.getRotationMatrix2D((w / 2, h / 2), -ang, 1.0)
        dl = cv2.bitwise_or(dl, cv2.warpAffine(r, Mi, (w, h), flags=cv2.INTER_NEAREST))
    hb = cv2.morphologyEx(hl, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, core)))
    vb = cv2.morphologyEx(vl, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (core, 1)))
    band = cv2.bitwise_or(cv2.bitwise_or(hb, vb), dl)
    thick = max(3, int(5 * px_per_cm))   # ~5 cm: a single line is thinner, a wall band is thicker
    wall = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (thick, thick)))
    # solid dark blobs (columns, hatched/filled walls): ink that survives a thick opening
    solid = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), cv2.MORPH_OPEN,
                             np.ones((thick + 2, thick + 2), np.uint8))
    wall = cv2.bitwise_or(wall, solid)
    # drop small isolated pieces (chairs, text clusters): walls form a connected network
    n, lab, st, _ = cv2.connectedComponentsWithStats(wall, 8)
    keep = np.zeros_like(wall)
    minarea = int((60 * px_per_cm) ** 2 * 0.15)
    for i in range(1, n):
        if st[i, 4] >= minarea or max(st[i, 2], st[i, 3]) >= 3 * L:
            keep[lab == i] = 255
    return keep


def cells(wall, seeds, outside=(), px_per_cm=0.8):
    """seeds: list of (id, [(x, y), ...]) -> {id: mask}. Watershed of the free space, door necks split basins."""
    free = (wall == 0).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    markers = np.zeros(wall.shape, np.int32)
    ids = []
    r = max(2, int(3 * px_per_cm))
    for k, (rid, pts) in enumerate(seeds, start=1):
        ids.append(rid)
        for (x, y) in pts:
            x, y = _nearest_free(free, int(x), int(y))
            cv2.circle(markers, (x, y), r, k, -1)
    out_lab = len(seeds) + 1
    for (x, y) in outside:
        x, y = _nearest_free(free, int(x), int(y))
        cv2.circle(markers, (x, y), r, out_lab, -1)
    # border of the image is outside too
    markers[0, :] = markers[-1, :] = out_lab
    markers[:, 0] = markers[:, -1] = out_lab
    markers[wall > 0] = 0
    from skimage.segmentation import watershed
    ws = watershed(-dist, markers, mask=free > 0)
    res = {}
    for k, rid in enumerate(ids, start=1):
        m = (ws == k).astype(np.uint8)
        if m.any():   # holes inside the room (text, furniture blobs) belong to it
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            m = np.zeros_like(m); cv2.drawContours(m, cnts, -1, 1, -1)
        res[rid] = m * 255
    return res


def _nearest_free(free, x, y):
    h, w = free.shape
    x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    if free[y, x]:
        return x, y
    for rad in range(1, 40):
        y0, y1, x0, x1 = max(0, y - rad), min(h, y + rad + 1), max(0, x - rad), min(w, x + rad + 1)
        ys, xs = np.nonzero(free[y0:y1, x0:x1])
        if len(xs):
            i = np.argmin((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2)
            return int(xs[i] + x0), int(ys[i] + y0)
    return x, y


def mask_to_polygon(m, eps=2.0):
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    a = cv2.approxPolyDP(c, eps, True)[:, 0, :]
    pts = [[int(x), int(y)] for x, y in a]
    pts = orthogonalize(pts)
    if len(pts) < 3:
        return None
    P = Polygon(pts)
    if not P.is_valid:
        P = make_valid(P)
        if P.geom_type != "Polygon":
            P = max(getattr(P, "geoms", [P]), key=lambda g: g.area)
        pts = [[int(round(x)), int(round(y))] for x, y in list(P.exterior.coords)[:-1]]
    if Polygon(pts).exterior.is_ccw:   # image y is down: clockwise on screen == ccw in math coords
        pass
    else:
        pts = pts[::-1]
    return pts


def orthogonalize(pts, tol_deg=8):
    """Snap near-horizontal/vertical edges exactly (shared coordinate); oblique edges stay."""
    n = len(pts)
    if n < 4:
        return pts
    P = [list(p) for p in pts]
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ang = abs(math.degrees(math.atan2(dy, dx))) % 180
        if min(ang, 180 - ang) < tol_deg:
            y = int(round((a[1] + b[1]) / 2)); a[1] = b[1] = y
        elif abs(ang - 90) < tol_deg:
            x = int(round((a[0] + b[0]) / 2)); a[0] = b[0] = x
    # remove duplicates / collinear points
    out = []
    for p in P:
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    res = []
    m = len(out)
    for i in range(m):
        a, b, c = out[i - 1], out[i], out[(i + 1) % m]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            res.append(b)
    return res if len(res) >= 3 else out


def area_m2(pts, px_per_cm):
    return Polygon(pts).area / (px_per_cm ** 2) / 10000.0


def wall_stickiness(pts, wall, tol=3):
    """Share of boundary length within tol px of the wall mask."""
    d = cv2.distanceTransform((wall == 0).astype(np.uint8), cv2.DIST_L2, 3)
    n, near = 0, 0
    for i in range(len(pts)):
        a, b = np.array(pts[i], float), np.array(pts[(i + 1) % len(pts)], float)
        L = int(np.hypot(*(b - a)))
        for t in np.linspace(0, 1, max(2, L // 2)):
            x, y = (a + t * (b - a)).astype(int)
            if 0 <= y < d.shape[0] and 0 <= x < d.shape[1]:
                n += 1; near += d[y, x] <= tol
    return near / max(1, n)
