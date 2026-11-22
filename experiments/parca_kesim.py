"""Choosing pieces from a ROUGH room polygon so that the room's outline prefers DRAWN lines.

Plain voting takes every piece lying mostly inside the rough polygon. With many small pieces the outline then simply
follows the rough polygon along helper cuts (lines that are not drawn). Here the choice is one minimum cut:
  - leaving a piece on the wrong side of the rough polygon costs its misplaced area (px2),
  - separating two touching pieces costs  lam * (length of their border that is NOT drawn ink)  + eps * drawn length.
So the outline runs along drawn lines for free and crosses open floor only where the rough polygon insists on it
(a doorway). The piece under the room's label is always inside.

select(fs, borders, rough_polygon, seed_xy, lam) -> list of piece indices."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from shapely.geometry import Point


def select(fs, borders, P, seed_xy=None, lam=20.0, eps=0.15, reach=60.0):
    """fs: pieces; borders: [i, j, length, ink share, gray]; P: rough polygon (shapely). Only pieces near P take part."""
    near = P.buffer(reach); idx = [i for i, f in enumerate(fs) if f.intersects(near)]
    if not idx: return []
    pos = {i: k for k, i in enumerate(idx)}; n = len(idx); S, T = n, n + 1
    rows, cols, caps = [], [], []
    def add(u, v, c):
        c = int(round(c * 10))
        if c > 0: rows.append(u); cols.append(v); caps.append(c)
    BIG = 10 ** 9
    for i in idx:
        f = fs[i]; a_in = f.intersection(P).area; a_out = f.area - a_in; k = pos[i]
        if seed_xy is not None and f.contains(Point(*seed_xy)): rows.append(S); cols.append(k); caps.append(BIG); continue
        add(S, k, a_in); add(k, T, a_out)
    for i, j, L, ink, g in borders:
        if i in pos and j in pos:
            w = lam * L * (1.0 - ink) + eps * L * ink
            add(pos[i], pos[j], w); add(pos[j], pos[i], w)
    G = csr_matrix((np.array(caps, np.int64), (rows, cols)), shape=(n + 2, n + 2))
    G.sum_duplicates()
    res = maximum_flow(G.astype(np.int32) if G.max() < 2 ** 31 else G, S, T)
    resid = (G - res.flow).tocsr()
    seen = np.zeros(n + 2, bool); seen[S] = True; stack = [S]                    # nodes still reachable from the source = the room
    while stack:
        u = stack.pop()
        for v, c in zip(resid.indices[resid.indptr[u]:resid.indptr[u + 1]], resid.data[resid.indptr[u]:resid.indptr[u + 1]]):
            if c > 0 and not seen[v]: seen[v] = True; stack.append(v)
    return [i for i in idx if seen[pos[i]]]
