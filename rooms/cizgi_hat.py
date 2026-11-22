"""From stroke records to LINES: collinear strokes that continue each other (broken only by a crossing stroke or a
small gap) become one line with an id, two end points and a thickness. Still no model and no meaning attached:
a line may be a wall face, a column side, a dimension line or furniture. The model decides that.

lines_in(enc, box) -> list of {"id", "p1", "p2", "len", "thick", "ang"} for the lines touching the box."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import numpy as np


def lines_in(enc, box, gap=6.0, off_tol=1.6, ang_tol=2.0, min_len=10.0, pad=30):
    x0, y0, x1, y1 = box
    s = enc["seg"]; th = enc["thick"]; gr = enc["gray"] if "gray" in enc else np.zeros(len(enc["seg"]))
    m = (np.maximum(s[:, 0], s[:, 2]) >= x0 - pad) & (np.minimum(s[:, 0], s[:, 2]) <= x1 + pad) & \
        (np.maximum(s[:, 1], s[:, 3]) >= y0 - pad) & (np.minimum(s[:, 1], s[:, 3]) <= y1 + pad)
    if "glyph" in enc: m &= ~enc["glyph"]            # letters and digits are not lines
    s = s[m].astype(float); th = th[m]; gr = gr[m]
    d = s[:, 2:] - s[:, :2]; L = np.hypot(d[:, 0], d[:, 1]); keep = L >= 2.5
    s, th, d, L, gr = s[keep], th[keep], d[keep], L[keep], gr[keep]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    order = np.argsort(-L); used = np.zeros(len(s), bool); out = []
    for i in order:
        if used[i]: continue
        # grow a line from the longest unused stroke: take strokes on the same line that touch its span
        members = [i]; used[i] = True
        u = d[i] / L[i]; n = np.array([-u[1], u[0]]); o = s[i, :2]
        t0, t1 = 0.0, L[i]
        changed = True
        while changed:
            changed = False
            cand = np.where(~used)[0]
            if not len(cand): break
            da = np.abs(ang[cand] - ang[i]); da = np.minimum(da, 180 - da)
            p = s[cand, :2] - o; q = s[cand, 2:] - o
            on = (da <= ang_tol + 40.0 / np.maximum(L[cand], 4)) & (np.abs(p @ n) <= off_tol) & (np.abs(q @ n) <= off_tol)
            a0 = np.minimum(p @ u, q @ u); a1 = np.maximum(p @ u, q @ u)
            touch = on & (a1 >= t0 - gap) & (a0 <= t1 + gap)
            for j, lo, hi in zip(cand[touch], a0[touch], a1[touch]):
                used[j] = True; members.append(j); t0 = min(t0, lo); t1 = max(t1, hi); changed = True
            if changed and len(members) >= 2:      # refit the line through all member end points
                P = np.vstack([s[members, :2], s[members, 2:]]); w = np.repeat(L[members], 1)
                c = P.mean(0); _, _, vt = np.linalg.svd(P - c); u2 = vt[0]
                if u2 @ u < 0: u2 = -u2
                tt = (P - c) @ u2; o = c + u2 * tt.min(); u = u2; n = np.array([-u[1], u[0]]); t0, t1 = 0.0, float(tt.max() - tt.min())
        if t1 - t0 < min_len: continue
        p1 = o + u * t0; p2 = o + u * t1
        if max(p1[0], p2[0]) < x0 or min(p1[0], p2[0]) > x1 or max(p1[1], p2[1]) < y0 or min(p1[1], p2[1]) > y1: continue
        out.append({"p1": p1, "p2": p2, "len": float(t1 - t0), "thick": float(np.average(th[members], weights=L[members])),
                    "ang": float(np.degrees(np.arctan2(u[1], u[0])) % 180.0), "ink": float(L[members].sum()),
                    "gray": float(np.average(gr[members], weights=L[members]))})
    out.sort(key=lambda r: -r["len"])
    for k, r in enumerate(out): r["id"] = k + 1
    return out
