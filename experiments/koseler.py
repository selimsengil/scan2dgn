"""Corner candidates from the plan's own ink: straight segments (any angle) -> junction points where two segments of
different direction meet. Qwen later picks numbered candidates instead of writing coordinates."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import cv2
import numpy as np


def segments(gray, min_len=18):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    L = lsd.detect(gray)[0]
    if L is None: return np.zeros((0, 4))
    L = L.reshape(-1, 4)
    return L[np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]) >= min_len]


def junctions(segs, reach=7, min_angle=20, merge=5):
    """Intersections of segment pairs whose supporting lines cross within `reach` px of both segments."""
    n = len(segs); P = segs[:, :2]; Q = segs[:, 2:]; D = Q - P; Ln = np.hypot(D[:, 0], D[:, 1]); U = D / Ln[:, None]
    pts = []
    # coarse neighbour search on a grid of segment boxes
    x0 = np.minimum(P[:, 0], Q[:, 0]) - reach; x1 = np.maximum(P[:, 0], Q[:, 0]) + reach
    y0 = np.minimum(P[:, 1], Q[:, 1]) - reach; y1 = np.maximum(P[:, 1], Q[:, 1]) + reach
    for i in range(n):
        cand = np.where((x0 < x1[i]) & (x1 > x0[i]) & (y0 < y1[i]) & (y1 > y0[i]))[0]
        for j in cand[cand > i]:
            cr = U[i, 0] * U[j, 1] - U[i, 1] * U[j, 0]
            if abs(cr) < np.sin(np.radians(min_angle)): continue
            w = P[j] - P[i]
            t = (w[0] * U[j, 1] - w[1] * U[j, 0]) / cr; s = (w[0] * U[i, 1] - w[1] * U[i, 0]) / cr
            if -reach <= t <= Ln[i] + reach and -reach <= s <= Ln[j] + reach:
                pts.append(P[i] + t * U[i])
    if not pts: return np.zeros((0, 2))
    pts = np.array(pts); keep = []; used = np.zeros(len(pts), bool)
    for k in range(len(pts)):      # greedy merge of near-duplicates
        if used[k]: continue
        m = np.hypot(*(pts - pts[k]).T) <= merge
        keep.append(pts[m].mean(0)); used |= m
    return np.array(keep)
