"""Snap a shifted DGN opening box onto the scan's jambs: translate it (dx, dy within +-R px) to the position where
both short sides (jambs, across the wall) lie on ink.  Accepted only when the best jamb share >= min_j and it is
clearly better than the original.   snap(P, dist, tol, R) -> (P2, jamb_before, jamb_after, (dx, dy))"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import numpy as np
from shapely.affinity import translate

from aciklik_denetim import sides


def _pts(seg, n=None):
    a, b = np.asarray(seg[0], float), np.asarray(seg[1], float); n = n or max(4, int(np.hypot(*(b - a))))
    t = np.linspace(0.1, 0.9, n)[:, None]; return a + t * (b - a)


def wall_pts(P, s):
    """Points on the two wall faces (long sides of the box) continued 3..12 cm beyond each jamb: the wall goes on there."""
    s1, s2, _, _ = sides(P); a, b = (s1[0] + s1[1]) / 2, (s2[0] + s2[1]) / 2; u = (b - a) / max(np.hypot(*(b - a)), 1e-6)
    out = []
    for c in (s1[0], s1[1]):                                   # the two faces through the corners of side 1
        f = c - a
        for e in np.linspace(0.03 * s, 0.12 * s, 6): out += [a + f - u * e, b + f + u * e]
    return np.array(out)


def snap(P, dist, tol, R, min_j=0.9, s=None, min_w=0.6):
    s1, s2, _, _ = sides(P); p = np.vstack([_pts(s1), _pts(s2)]); n1 = len(_pts(s1)); H, W = dist.shape
    d = np.arange(-R, R + 1); DX, DY = np.meshgrid(d, d); DX = DX.ravel(); DY = DY.ravel()
    xi = np.clip(np.round(p[None, :, 0] + DX[:, None]).astype(int), 0, W - 1); yi = np.clip(np.round(p[None, :, 1] + DY[:, None]).astype(int), 0, H - 1)
    ok = dist[yi, xi] <= tol; j = np.minimum(ok[:, :n1].mean(1), ok[:, n1:].mean(1))
    j0 = float(j[(DX == 0) & (DY == 0)][0]); best = j.max()
    cand = np.where(j >= best - 1e-6)[0]; k = cand[np.argmin(DX[cand] ** 2 + DY[cand] ** 2)]   # nearest of the best
    if best < min_j or best <= j0: return P, j0, j0, (0, 0)
    if s is not None:                                           # the wall must go on beyond both jambs
        w = wall_pts(P, s); wx = np.clip(np.round(w[None, :, 0] + DX[cand, None]).astype(int), 0, W - 1); wy = np.clip(np.round(w[None, :, 1] + DY[cand, None]).astype(int), 0, H - 1)
        wo = (dist[wy, wx] <= tol).mean(1); good = cand[wo >= min_w]
        if not len(good): return P, j0, j0, (0, 0)
        k = good[np.argmin(DX[good] ** 2 + DY[good] ** 2)]
    return translate(P, int(DX[k]), int(DY[k])), j0, float(best), (int(DX[k]), int(DY[k]))
