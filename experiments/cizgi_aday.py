"""Which lines to put in front of the model for one room: the lines seen from the room's label point (first few hits
of rays in all directions) plus the lines that touch those. Only a shortlist; the model decides what is a wall.

candidates(lines, pt) -> ids (list), in no particular order."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import numpy as np


def _ray_hits(lines, pt, n_rays=64, max_dist=900.0):
    P1 = np.array([l["p1"] for l in lines]); P2 = np.array([l["p2"] for l in lines]); D = P2 - P1
    hits = []
    for a in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        r = np.array([np.cos(a), np.sin(a)])
        den = r[0] * D[:, 1] - r[1] * D[:, 0]
        w = P1 - pt
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[:, 0] * D[:, 1] - w[:, 1] * D[:, 0]) / den      # along the ray
            s = (w[:, 0] * r[1] - w[:, 1] * r[0]) / den            # along the line, 0..1
        ok = (np.abs(den) > 1e-9) & (t > 1.0) & (t < max_dist) & (s >= -0.01) & (s <= 1.01)
        idx = np.where(ok)[0]
        hits.append([(float(t[i]), int(i)) for i in idx[np.argsort(t[idx])]])
    return hits


def _pt_seg(p, a, b):
    d = b - a; L2 = d @ d
    t = 0.0 if L2 == 0 else np.clip((p - a) @ d / L2, 0, 1)
    return float(np.hypot(*(p - (a + t * d))))


def candidates(lines, pt, first=3, min_hit_len=12.0, touch=6.0, rounds=1):
    pt = np.asarray(pt, float)
    big = [k for k, l in enumerate(lines) if l["len"] >= min_hit_len]
    sub = [lines[k] for k in big]
    chosen = set()
    for ray in _ray_hits(sub, pt):
        for _, i in ray[:first]: chosen.add(big[i])
    for _ in range(rounds):                       # neighbours: a line with an end on a chosen line, or a chosen line's end on it
        add = set()
        for k, l in enumerate(lines):
            if k in chosen: continue
            for c in chosen:
                m = lines[c]
                if min(_pt_seg(l["p1"], m["p1"], m["p2"]), _pt_seg(l["p2"], m["p1"], m["p2"]),
                       _pt_seg(m["p1"], l["p1"], l["p2"]), _pt_seg(m["p2"], l["p1"], l["p2"])) <= touch:
                    add.add(k); break
        chosen |= add
    return sorted(chosen)
