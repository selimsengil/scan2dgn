"""From lines to FACES: the encoded lines cut the plan into closed pieces. A room is a set of pieces (its floor, cut
up by furniture or dimension lines); walls, columns and other rooms are other pieces. The model only says which pieces
belong to the room; the room boundary is then the outline of their union, exactly on the plan's own strokes.

Light geometry only: line ends are stretched a few px to meet the line they point at, and collinear lines across an
opening get a closing piece (so a doorway does not glue two rooms together). Extra cuts cost nothing: pieces can be
merged back by the model.

faces(lines, box) -> list of shapely Polygons (in floor pixels)."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import numpy as np
from shapely.geometry import LineString, box as sbox
from shapely.ops import polygonize, unary_union


def _extend(lines, reach, long_reach=150.0, long_len=30.0, through=True):
    """Each line end grows until it meets another line: up to `reach` px for a short line, up to `long_reach` px for a
    long one (a wall face that stops at a doorway carries on to the next wall, which closes the opening)."""
    P1 = np.array([l["p1"] for l in lines]); P2 = np.array([l["p2"] for l in lines]); D = P2 - P1
    out = []
    for i, l in enumerate(lines):
        u = (l["p2"] - l["p1"]) / l["len"]; ends = []
        for p, r in ((l["p1"], -u), (l["p2"], u)):
            den = r[0] * D[:, 1] - r[1] * D[:, 0]; w = P1 - p
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (w[:, 0] * D[:, 1] - w[:, 1] * D[:, 0]) / den; s = (w[:, 0] * r[1] - w[:, 1] * r[0]) / den
            lim = long_reach if l["len"] >= long_len else reach
            ok = (np.abs(den) > 1e-9) & (t > -1.0) & (t <= lim) & (s >= -0.02) & (s <= 1.02); ok[i] = False
            # a short line stops at the first line it meets; a long one runs on to the LAST line within reach, because a
            # door leaf or a swing arc standing in the opening must not stop a wall face from closing the doorway
            pick = t[ok].max() if (ok.any() and through and l["len"] >= long_len) else (t[ok].min() if ok.any() else None)
            grow = float(pick) + 0.6 if pick is not None else 1.5      # a hair past the meeting point so the noding closes
            ends.append(p + r * max(grow, 0.0))
        out.append((ends[0], ends[1]))
    return out


def _closings(lines, min_gap=14.0, max_gap=150.0, off=2.0, ang=2.0):
    """Collinear lines separated by an opening: the straight piece across it."""
    out = []; n = len(lines)
    for i in range(n):
        a = lines[i]
        if a["len"] < 12: continue
        u = (a["p2"] - a["p1"]) / a["len"]; nn = np.array([-u[1], u[0]])
        for j in range(i + 1, n):
            b = lines[j]
            if b["len"] < 12: continue
            da = abs(a["ang"] - b["ang"]); da = min(da, 180 - da)
            if da > ang: continue
            if abs((b["p1"] - a["p1"]) @ nn) > off or abs((b["p2"] - a["p1"]) @ nn) > off: continue
            t = sorted(((b["p1"] - a["p1"]) @ u, (b["p2"] - a["p1"]) @ u))
            if t[0] >= a["len"]: gap = t[0] - a["len"]; q1, q2 = a["p2"], (b["p1"] if (b["p1"] - a["p1"]) @ u < (b["p2"] - a["p1"]) @ u else b["p2"])
            elif t[1] <= 0: gap = -t[1]; q1, q2 = a["p1"], (b["p2"] if (b["p1"] - a["p1"]) @ u < (b["p2"] - a["p1"]) @ u else b["p1"])
            else: continue
            if min_gap <= gap <= max_gap: out.append((q1, q2))
    return out


def faces(lines, box, reach=10.0, min_area=25.0, closings=True, through=True, long_reach=150.0, long_len=30.0, max_gap=150.0):
    segs = _extend(lines, reach, long_reach=long_reach, long_len=long_len, through=through)
    if closings: segs += _closings(lines, max_gap=max_gap)
    x0, y0, x1, y1 = box
    geoms = [LineString([tuple(a), tuple(b)]) for a, b in segs if np.hypot(*(np.asarray(b) - np.asarray(a))) > 0.5]
    geoms.append(sbox(x0, y0, x1, y1).exterior)                 # the crop frame closes the pieces at the edge
    net = unary_union(geoms)
    return [p for p in polygonize(net) if p.area >= min_area]
