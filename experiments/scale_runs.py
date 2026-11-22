"""Scale from wall thickness: histogram of white runs bounded by ink on both sides.

A wall drawn as two parallel faces gives the same short white run in every row (or
column) along its length, so wall interiors form sharp peaks; text and hatching spread.
Prints the strongest peaks per floor next to the oracle 10/20 cm widths.
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import sys

import cv2
import numpy as np

import v0_watershed as V0


def white_runs(ink: np.ndarray, maxlen: int) -> np.ndarray:
    """Counts of white run lengths 1..maxlen between two ink pixels, along rows."""
    h = np.zeros(maxlen + 1, np.int64)
    for row in ink:
        idx = np.flatnonzero(np.diff(row.astype(np.int8)))   # transitions
        if len(idx) < 2:
            continue
        # white run = from an ink->white transition to the next white->ink
        starts = idx[:-1][row[idx[:-1]] == 1]
        for s0 in starts:
            pass
        r = row.astype(np.int8)
        d = np.diff(r)
        down = np.flatnonzero(d == -1)   # ink -> white at down+1
        up = np.flatnonzero(d == 1)      # white -> ink at up+1
        if not len(down) or not len(up):
            continue
        j = np.searchsorted(up, down)
        ok = j < len(up)
        L = up[j[ok]] - down[ok]
        L = L[(L >= 1) & (L <= maxlen)]
        h += np.bincount(L, minlength=maxlen + 1)[: maxlen + 1]
    return h


def ink_of(img: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink.astype(np.uint8)


def main():
    floors = sys.argv[1].split(",") if len(sys.argv) > 1 else V0.FLOORS
    for f in floors:
        img = cv2.imread(str(V0.B / f / "clean.png"), cv2.IMREAD_GRAYSCALE)
        s = V0.oracle_scale(V0.load_rooms(f))
        ink = ink_of(img)
        maxlen = 120
        h = white_runs(ink, maxlen) + white_runs(ink.T.copy(), maxlen)
        hs = np.convolve(h, [1, 2, 1], "same").astype(float)
        hs[:3] = 0
        # peaks relative to a smooth background
        bg = np.convolve(hs, np.ones(15) / 15, "same") + 1
        score = hs / bg
        top = np.argsort(-score)[:8]
        print(f, "oracle 10cm=%.1f 20cm=%.1f px" % (0.10 * s, 0.20 * s), "peaks(len:score):",
              ", ".join(f"{t}:{score[t]:.1f}" for t in sorted(top)))


if __name__ == "__main__":
    main()
