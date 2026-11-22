"""Label-free rooms: the boundary-line network (cizgi_ag c1) + code.  No DGN, no room labels are read.
1. line map p (probability of a room boundary line) at ~100 px/m
2. seeds = centres of open floor (h-maxima of the distance to the nearest line, at least --min-r m from a line)
3. watershed of p from the seeds (+ the page border as outside): every region ends on a line ridge
4. neighbouring regions are merged when the border between them runs without a line for more than --open m
   (an open passage, an over-split L-shaped hall); a door (~0.8-1.1 m) stays a border
5. regions touching the page border, smaller than --min-a m2 or thinner than 0.5 m are dropped
Output kat_test/oda_kur/<tag>/<floor>.png (+ .json with polygons in clean.png pixels) and SURE.json (seconds per step).
Run with .venv-spec.   usage: python oda_kur.py --floors A,B --tag k1"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

import bolut_egit as BE  # noqa: E402
import cizgi_ag as CA  # noqa: E402

OUT = config.KT / "oda_kur"


def rooms(p, s, open_m=1.3, min_r=0.3, min_a=0.8):
    H, W = p.shape; free = (p < 0.5).astype(np.uint8); D = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    pk = h_maxima(np.round(D).astype(np.int32), max(2, int(0.15 * s))) & (D >= min_r * s)
    n, mk = cv2.connectedComponents(pk.astype(np.uint8)); mk = mk.astype(np.int32) + 1; mk[mk == 1] = 0
    mk[0, :] = mk[-1, :] = mk[:, 0] = mk[:, -1] = 1
    lab = watershed(p, mk)
    # borders between labels: pixel pairs with different labels (right and down neighbours)
    low = p < 0.3; L = {}
    for a, b, q in ((lab[:, :-1], lab[:, 1:], low[:, :-1] & low[:, 1:]), (lab[:-1, :], lab[1:, :], low[:-1, :] & low[1:, :])):
        m = (a != b) & q
        for x, y in zip(a[m].ravel(), b[m].ravel()):
            k = (min(x, y), max(x, y)); L[k] = L.get(k, 0) + 1
    par = list(range(lab.max() + 1))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for (x, y), c in L.items():
        if x > 1 and c > open_m * s: par[f(y)] = f(x)
    root = np.array([f(i) for i in range(len(par))]); lab = root[lab]
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist()) | {1}
    for i in range(len(root)):           # anything merged with outside is outside
        if root[i] in border: pass
    out = []
    for r in np.unique(lab):
        if r in border: continue
        m = (lab == r).astype(np.uint8); area = m.sum() / s / s
        if area < min_a or cv2.distanceTransform(m, cv2.DIST_L2, 5).max() < 0.25 * s: continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) + 0.5
        out.append({"poly": c.round(1).tolist(), "m2": round(float(area), 2)})
    return out


def draw(g, rs, s):
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); v = (v * 0.6 + 100).astype(np.uint8); rng = np.random.default_rng(3); fill = v.copy()
    for r in rs:
        c = np.array(r["poly"], np.int32); col = [int(x) for x in rng.integers(60, 230, 3)]; cv2.fillPoly(fill, [c], col)
    v = cv2.addWeighted(fill, 0.35, v, 0.65, 0)
    for k, r in enumerate(rs):
        c = np.array(r["poly"], np.int32); cv2.polylines(v, [c], True, (0, 0, 210), max(2, int(s / 50)))
        m = cv2.moments(c); cx, cy = int(m["m10"] / max(m["m00"], 1)), int(m["m01"] / max(m["m00"], 1))
        cv2.putText(v, f"{k + 1}: {r['m2']:.2f}", (cx - int(s * 0.4), cy), cv2.FONT_HERSHEY_SIMPLEX, s / 130, (0, 0, 0), max(1, int(s / 60)))
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--floors", required=True); ap.add_argument("--tag", default="k1"); ap.add_argument("--net", default="c1")
    ap.add_argument("--open", type=float, default=1.3); a = ap.parse_args(); out = OUT / a.tag; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); dev = torch.device("cuda:0"); net = BE.UNet(n=2).to(dev).to(memory_format=torch.channels_last); net.load_state_dict(torch.load(CA.OUT / a.net / "unet.pt", map_location=dev))
    times = {"model_yukleme_s": round(time.time() - t0, 2)}
    for f in a.floors.split(","):
        L = json.load(open(CA.R / f / "LABELS_COMMON.json", encoding="utf-8")); s0 = L.get("px_per_m_from_room_areas") or L["px_per_m"]; r = 100.0 / s0 if s0 > 130 else 1.0; s = s0 * r
        t1 = time.time(); g = CA.load(f, 0.95)[0]; pr = CA.predict(net, g, dev); p = pr[1] / np.maximum(pr[0] + pr[1], 1e-6); t2 = time.time()
        rs = rooms(p, s, a.open); t3 = time.time()
        cv2.imwrite(str(out / f"{f}.png"), draw(g, rs, s))
        json.dump({"floor": f, "px_per_m": s, "scale_from_clean": r, "rooms": rs}, open(out / f"{f}.json", "w"), indent=0)
        times[f] = {"cizgi_agi_s": round(t2 - t1, 2), "oda_kurma_s": round(t3 - t2, 2), "oda": len(rs)}; print(f, times[f], flush=True)
    json.dump(times, open(out / "SURE.json", "w"), indent=1)
