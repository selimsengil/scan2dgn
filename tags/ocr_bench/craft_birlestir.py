"""CRAFT word boxes -> text-line boxes: keep boxes of text height (12..70 px at 315 px/m), join neighbours on one line
(same direction, gap < 1.2 x height, offset across < 0.5 x height).  python craft_birlestir.py IN_okumalar.json OUT_bolgeler.json"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys
import numpy as np
def frame(b):
    c = np.asarray(b, float); e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e)); u = (c[(k + 1) % 4] - c[k]) / max(e[k], 1e-6)
    if u[0] < 0 or (abs(u[0]) < 1e-6 and u[1] < 0): u = -u
    return c.mean(0), u, max(e), min(e)
def lines(boxes, hmin=12, hmax=70):
    B = [b for b in boxes if hmin <= frame(b)[3] <= hmax]; used = [False] * len(B); out = []
    for i in range(len(B)):
        if used[i]: continue
        grp = [i]; used[i] = True; changed = True
        while changed:
            changed = False
            for j in range(len(B)):
                if used[j]: continue
                cj, uj, Lj, hj = frame(B[j])
                for k in grp:
                    ck, uk, Lk, hk = frame(B[k]); h = max(hj, hk)
                    if abs(abs(uj @ uk) - 1) > 0.05: continue
                    d = cj - ck; along = abs(d @ uk); across = abs(d @ np.array([-uk[1], uk[0]]))
                    if across < 0.5 * h and along - (Lj + Lk) / 2 < 1.2 * h: grp.append(j); used[j] = True; changed = True; break
        pts = np.vstack([np.asarray(B[k], float) for k in grp]); c0, u, _, _ = frame(B[grp[0]]); n = np.array([-u[1], u[0]])
        t = pts @ u; q = pts @ n; box = [u * a + n * b for a, b in ((t.min(), q.min()), (t.max(), q.min()), (t.max(), q.max()), (t.min(), q.max()))]
        out.append({"box": np.round(box, 1).tolist(), "n_kelime": len(grp)})
    return out
if __name__ == "__main__":
    d = json.load(open(sys.argv[1], encoding="utf-8")); L = lines(d["kutular"]); json.dump(L, open(sys.argv[2], "w", encoding="utf-8")); print(len(d["kutular"]), "kutu ->", len(L), "satir")
