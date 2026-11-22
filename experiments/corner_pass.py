"""Corner pass: column shoulders, wall steps and door recesses the coarse drawing cut across.

For each corner (corners closer than `merge` px share one window) a square window of half-size R
around it, upscaled to ~900 px; the current boundary drawn thin, the edge coming IN green, the edge
going OUT red, the room inside tinted blue. Qwen lists the turning points of the room's inner face
in that window, in order from the green side to the red side (0-1000 of the window). The windows'
points, in order, become the new polygon; then edges of at least `snap_min` px are snapped again.
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import math

import numpy as np
from PIL import Image, ImageDraw

import qwen_room as QR
import qwen_test as QT
import snap

Q = ("Görüntü bir kat planında '{target}' odasının bir köşe bölgesinin büyütülmüş yakın görünümü. Odanın içi AÇIK MAVİ "
     "boyalı (yaklaşık). İnce çizgi şimdiki taslak sınır: YEŞİL kısım sınırın bu bölgeye GELDİĞİ kenar, KIRMIZI kısım "
     "ÇIKTIĞI kenar. TURUNCU kalın çizgi varsa odanın giriş kapısı açıklığıdır.\n" + QT.RULES +
     " Odaya bakan kolon yüzleri ve duvar basamakları sınıra dahildir; kapı ya da balkon kapısı önündeki duvar girintisi "
     "duvar iç yüzü hizasında kapatılır.\nBu bölgede odanın GERÇEK iç yüz sınırının köşe/dönüş noktalarını, yeşil taraftan "
     "kırmızı tarafa giderken sırayla ver (taslak yanlışsa düzelt; kolon omzu veya basamak varsa her dönüşü ayrı nokta "
     "olarak ver). Koordinatlar bu görüntünün 0-1000 normalize koordinatları. Yalnız JSON: {{\"points\": [[x, y], ...]}}")


def _simple(poly) -> bool:
    from shapely.geometry import Polygon
    try:
        p = Polygon(poly)
        return p.is_valid and p.area > 10
    except Exception:
        return False


def _ink_ok(crop, pts, need=0.45, slack=2):
    """Every edge between consecutive proposed points lies on ink (within +-slack px)."""
    g = np.asarray(crop.convert("L"))
    for a, b in zip(pts, pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        if L < 6:
            continue
        u = (b - a) / L; nrm = np.array([-u[1], u[0]])
        best = max(snap._line_dark(g, a + nrm * o, b + nrm * o, frac=0.8) for o in range(-slack, slack + 1))
        if best < need:
            return False
    return True


def _clusters(poly, merge):
    n = len(poly); groups = [[0]]
    for i in range(1, n):
        if math.dist(poly[i], poly[groups[-1][-1]]) < merge:
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1 and math.dist(poly[groups[0][0]], poly[groups[-1][-1]]) < merge:
        groups[0] = groups[-1] + groups[0]; groups.pop()
    return groups


def corner_pass(crop: Image.Image, target, hint, poly, R=90, merge=60, side_px=900, snap_min=25, out=None, tag=""):
    W, H = crop.size
    n = len(poly)
    P = [np.asarray(p, float) for p in poly]
    new = []
    log = []
    for gi, g in enumerate(_clusters(poly, merge)):
        pts = np.asarray([P[i] for i in g])
        c = pts.mean(axis=0)
        r = max(R, int(np.max(np.abs(pts - c))) + R // 2)
        x0, y0 = int(c[0] - r), int(c[1] - r); side = 2 * r
        reg = Image.new("RGB", (side, side), "white")
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
        reg.paste(crop.convert("RGB").crop((sx0, sy0, sx1, sy1)), (sx0 - x0, sy0 - y0))
        s = side_px / side
        big = reg.resize((side_px, side_px), Image.Resampling.NEAREST)
        tint = Image.new("L", big.size, 0)
        ImageDraw.Draw(tint).polygon([((x - x0) * s, (y - y0) * s) for x, y in poly], fill=60)
        big = Image.composite(Image.new("RGB", big.size, (90, 160, 255)), big, tint)
        d = ImageDraw.Draw(big)
        T = lambda p: ((p[0] - x0) * s, (p[1] - y0) * s)
        ja, jb = hint.jambs
        d.line([T(ja), T(jb)], fill="#ef5000", width=4)
        first, last = g[0], g[-1]
        prev = P[first - 1]; nxt = P[(last + 1) % n]
        d.line([T(prev), T(P[first])], fill=(0, 170, 0), width=3)               # coming in
        for a, b in zip(g, g[1:]):
            d.line([T(P[a]), T(P[b])], fill=(255, 140, 0), width=2)
        d.line([T(P[last]), T(nxt)], fill=(220, 0, 0), width=3)                 # going out
        ans_pts, raw = QR.ask_points(Q.format(target=target), big)
        got = [[x0 + (x * side_px / 1000.0) / s, y0 + (y * side_px / 1000.0) / s] for x, y in ans_pts]
        # points on the window's border are where the boundary enters / leaves, not turns
        m = 0.06 * side
        got = [p for p in got if x0 + m < p[0] < x0 + side - m and y0 + m < p[1] < y0 + side - m]
        why = "accepted"
        if not got:
            got, why = [P[i].tolist() for i in g], "no inner point"
        elif not _ink_ok(crop, got):
            got, why = [P[i].tolist() for i in g], "a new edge has no ink line under it"
        else:
            trial = [q for k, q in enumerate(new)] + got + [P[i].tolist() for gg in _clusters(poly, merge)[gi + 1:] for i in gg]
            if len(trial) >= 3 and not _simple(trial):
                got, why = [P[i].tolist() for i in g], "self-crossing"
        new.extend(got)
        log.append({"corners": g, "window": [x0, y0, side], "points": got, "why": why, "raw": raw[-200:]})
        if out is not None:
            big.save(out / f"{tag}corner{gi:02d}.png")
    # tidy: drop near-duplicates, then snap edges long enough to measure
    clean = []
    for p in new:
        if clean and math.dist(p, clean[-1]) < 4:
            continue
        clean.append(p)
    if len(clean) > 3 and math.dist(clean[0], clean[-1]) < 4:
        clean.pop()
    snapped = snap_long(crop, clean, snap_min)
    return snapped, clean, log


def snap_long(crop, poly, min_len):
    """snap.snap_polygon, but edges shorter than min_len keep their place (a column shoulder is short)."""
    full = snap.snap_polygon(crop, poly, max_px=6)
    n = len(poly)
    out = []
    for i in range(n):
        lin = math.dist(poly[i - 1], poly[i]) >= min_len
        lout = math.dist(poly[i], poly[(i + 1) % n]) >= min_len
        out.append(full[i] if (lin and lout) else poly[i])
    return out
