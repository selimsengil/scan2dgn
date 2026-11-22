"""Line encoder: every ink stroke of a scanned plan becomes a record (two end points, thickness), with the graph of
where strokes meet. No model. The encoding is meant to lose nothing a room boundary needs: corners come from stroke
ends and meeting points, at pixel precision.

encode(gray) -> {"seg": (N,4) float x1,y1,x2,y2, "thick": (N,), "chain": (N,) id of the stroke it came from,
                 "node": (M,2) points where strokes end or meet, "seg_node": (N,2) node index at each end (-1 = bend)}
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import cv2
import numpy as np
from skimage.morphology import skeletonize

NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def ink_mask(gray, dark=150, faint=200, contrast=25):
    """Ink = dark pixels, plus FAINT thin lines (window lines, hatched column edges): a pixel under `faint` that is
    darker by `contrast` than the paper 2-3 px away on both sides, in any of the four directions."""
    g = gray.astype(np.int16); ink = g < dark
    if faint <= dark: return ink
    H, W = g.shape; pad = np.pad(g, 3, mode="edge"); ridge = np.zeros_like(ink)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        def sh(k): return pad[3 + k * dy:3 + k * dy + H, 3 + k * dx:3 + k * dx + W]
        a = np.maximum(sh(2), sh(3)); b = np.maximum(sh(-2), sh(-3))
        ridge |= (a - g >= contrast) & (b - g >= contrast)
    return ink | ((g < faint) & ridge)


def _chains(sk):
    """Skeleton -> ordered pixel chains between nodes (ends and meeting points)."""
    nb = cv2.filter2D(sk.astype(np.uint8), -1, NB, borderType=cv2.BORDER_CONSTANT)
    node = sk & (nb != 2)
    branch = sk & ~node
    n_lab, lab = cv2.connectedComponents(branch.astype(np.uint8), connectivity=8)
    n_nodes, nlab = cv2.connectedComponents(node.astype(np.uint8), connectivity=8)   # touching node pixels = one node
    ys, xs = np.nonzero(branch)
    order = np.argsort(lab[ys, xs], kind="stable"); ys, xs = ys[order], xs[order]; labs = lab[ys, xs]
    starts = np.searchsorted(labs, np.arange(1, n_lab)); ends = np.append(starts[1:], len(labs))
    H, W = sk.shape; chains = []
    for s, e in zip(starts, ends):
        pix = set(zip(ys[s:e].tolist(), xs[s:e].tolist()))
        # an end pixel of the chain has at most one neighbour inside the chain
        first = None
        for p in pix:
            if sum(((p[0] + dy, p[1] + dx) in pix) for dy, dx in OFFS) <= 1: first = p; break
        if first is None: first = next(iter(pix))          # a closed loop
        path = [first]; seen = {first}; cur = first
        while True:
            nxt = None
            for dy, dx in OFFS:                              # 4-neighbours come first in a good walk; any order works on a thin chain
                q = (cur[0] + dy, cur[1] + dx)
                if q in pix and q not in seen: nxt = q; break
            if nxt is None: break
            path.append(nxt); seen.add(nxt); cur = nxt
        ends_node = []
        for tip, toward in ((path[0], 0), (path[-1], -1)):   # attach the node pixel each tip touches
            hit = -1
            for dy, dx in OFFS:
                y, x = tip[0] + dy, tip[1] + dx
                if 0 <= y < H and 0 <= x < W and node[y, x]:
                    hit = nlab[y, x]
                    if toward == 0: path.insert(0, (y, x))
                    else: path.append((y, x))
                    break
            ends_node.append(hit)
        chains.append((np.array(path, np.float32)[:, ::-1], ends_node))    # as x, y
    # node positions: centroid of each node blob
    ny, nx = np.nonzero(node); nl = nlab[ny, nx]
    cnt = np.bincount(nl, minlength=n_nodes); cx = np.bincount(nl, nx, n_nodes) / np.maximum(cnt, 1); cy = np.bincount(nl, ny, n_nodes) / np.maximum(cnt, 1)
    return chains, np.stack([cx, cy], 1)


def encode(gray, dark=150, eps=1.2, min_chain=3, glyph=46, faint=200):
    ink = ink_mask(gray, dark, faint)
    sk = skeletonize(ink)
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    chains, nodes = _chains(sk)
    seg, thick, chain_id, seg_node, sgray = [], [], [], [], []
    for ci, (pts, en) in enumerate(chains):
        if len(pts) < min_chain: continue
        ap = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, False).reshape(-1, 2)
        # index of each kept vertex along the chain
        idx = [int(np.argmin(np.abs(pts - v).sum(1))) for v in ap]
        for k in range(len(ap) - 1):
            i0, i1 = sorted((idx[k], idx[k + 1])); part = pts[i0:i1 + 1]
            a, b = ap[k].astype(float), ap[k + 1].astype(float)
            if len(part) >= 6:                                # least-squares line through the stroke's pixels
                vx, vy, x0, y0 = cv2.fitLine(part, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
                d = np.array([vx, vy]); o = np.array([x0, y0])
                a = o + d * np.dot(a - o, d); b = o + d * np.dot(b - o, d)
            xi = np.clip(part[:, 0].astype(int), 0, gray.shape[1] - 1); yi = np.clip(part[:, 1].astype(int), 0, gray.shape[0] - 1)
            seg.append([a[0], a[1], b[0], b[1]]); thick.append(2.0 * float(np.median(dist[yi, xi]))); chain_id.append(ci)
            sgray.append(float(np.median(gray[yi, xi])))
            seg_node.append([en[0] if k == 0 else -1, en[1] if k == len(ap) - 2 else -1])
    seg = np.array(seg, np.float32).reshape(-1, 4)
    # strokes of small isolated marks (letters, digits, symbols): ink islands smaller than `glyph` px that are not a filled block
    n_cc, cc, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    w, h, area = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    small = (np.maximum(w, h) <= glyph) & ~((np.minimum(w, h) >= 14) & (area >= 0.45 * w * h))
    small[0] = False
    if len(seg):
        mx = np.clip(((seg[:, 0] + seg[:, 2]) / 2).round().astype(int), 0, gray.shape[1] - 1); my = np.clip(((seg[:, 1] + seg[:, 3]) / 2).round().astype(int), 0, gray.shape[0] - 1)
        lab = cc[my, mx]
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1)):        # a fitted mid point can fall just off the ink
            miss = lab == 0
            if not miss.any(): break
            lab[miss] = cc[np.clip(my[miss] + dy, 0, gray.shape[0] - 1), np.clip(mx[miss] + dx, 0, gray.shape[1] - 1)]
        is_glyph = small[lab]
    else:
        is_glyph = np.zeros(0, bool)
    return {"seg": seg, "glyph": is_glyph, "gray": np.array(sgray, np.float32), "thick": np.array(thick, np.float32), "chain": np.array(chain_id, np.int32),
            "node": nodes.astype(np.float32), "seg_node": np.array(seg_node, np.int32).reshape(-1, 2), "ink": ink, "skeleton": sk}


def redraw(enc, shape, width=None):
    """The plan drawn again from the records alone (uint8 mask)."""
    out = np.zeros(shape, np.uint8)
    for (x1, y1, x2, y2), t in zip(enc["seg"], enc["thick"]):
        cv2.line(out, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, width or max(1, int(round(t))))
    return out > 0


def corner_points(enc):
    """Every point a room corner can sit on: stroke ends, bends and meeting points."""
    s = enc["seg"]
    return np.vstack([s[:, :2], s[:, 2:], enc["node"]])
