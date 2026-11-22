"""Trains the piece classifier on labelled floors and tests it on another floor (never 015 unless --final).
  stage 1  gradient-boosted trees on each piece's own features
  stage 2  the same, plus what stage 1 thinks of the piece's neighbours (and their neighbours)
  rooms    from every room label point, spread over pieces called "room floor"; where two rooms meet through a mistake,
           the piece goes to the room that reaches it across less drawn ink. Outline = the pieces' own outline.
Reports: piece classes on the test floor, door / window openings found, and the boundary score of the rooms.
usage: python parca_egit.py --train 001,002,010,013,016 --test 019 --tag a1"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, heapq, json, pickle, sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from shapely import affinity, wkb
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

import v0_watershed as V0, sinir_skor as SS  # noqa: E402

D = config.KT / "parca_veri"; OUT = config.KT / "parca_egit"
CLS = ["etiketsiz", "oda", "duvar", "kolon", "kapi", "pencere"]


def load(f):
    z = np.load(D / f"{f}.npz"); return {k: z[k] for k in z.files}


def nb_mean(P, bord, n, hops=1):
    """Border-length weighted mean of the neighbours' class probabilities."""
    i = bord[:, 0].astype(int); j = bord[:, 1].astype(int); L = bord[:, 2:3]
    out = P
    for _ in range(hops):
        S = np.zeros_like(P); W = np.zeros((n, 1))
        np.add.at(S, i, out[j] * L); np.add.at(S, j, out[i] * L); np.add.at(W, i, L); np.add.at(W, j, L)
        out = S / np.maximum(W, 1e-6)
    return out


def fit(X, y, w, seed=0):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1, n_jobs=8)
    m.fit(X, y, sample_weight=w); return m


def weights(d):
    y = d["y"]; a = np.clip(np.sqrt(d["X"][:, 0]), 0.03, 1.5)        # big pieces matter more, tiny ones still count
    w = a.copy()
    for c in range(1, 6): w[y == c] *= 1.0 / max(a[y == c].sum(), 1e-6)
    return w * 1e4


def proba(m, X):
    p = np.zeros((len(X), 6), np.float32); p[:, m.classes_] = m.predict_proba(X); return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", default=""); ap.add_argument("--test", required=True); ap.add_argument("--tag", default="a1"); ap.add_argument("--final", action="store_true"); ap.add_argument("--proba", default=""); ap.add_argument("--thin", type=float, default=0.0); ap.add_argument("--seed-r", type=float, default=0.3); ap.add_argument("--gap", type=float, default=1.5)
    a = ap.parse_args(); tr = [f for f in a.train.split(",") if f]; te = a.test; assert a.final or "015" not in tr + [te]
    out = OUT / f"{te}_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    data = {f: load(f) for f in tr + [te]}
    if a.proba:                                   # piece opinions from elsewhere (the segmentation network): no training here
        P2 = np.load(a.proba); P1 = {te: P2}; tr = []
    # stage 1: out-of-floor predictions for the training floors, so stage 2 learns from honest neighbour opinions
    P1 = {te: P2} if a.proba else {}
    for f in tr:
        rest = [g for g in tr if g != f]
        if not rest: P1[f] = np.full((len(data[f]["y"]), 6), 1 / 6, np.float32); continue
        X = np.vstack([data[g]["X"][data[g]["y"] > 0] for g in rest]); y = np.concatenate([data[g]["y"][data[g]["y"] > 0] for g in rest]); w = np.concatenate([weights(data[g])[data[g]["y"] > 0] for g in rest])
        P1[f] = proba(fit(X, y, w), data[f]["X"])
    if not a.proba: Xa = np.vstack([data[g]["X"][data[g]["y"] > 0] for g in tr]); ya = np.concatenate([data[g]["y"][data[g]["y"] > 0] for g in tr]); wa = np.concatenate([weights(data[g])[data[g]["y"] > 0] for g in tr])
    if not a.proba:
        m1 = fit(Xa, ya, wa); P1[te] = proba(m1, data[te]["X"])
        ctx = lambda f: np.hstack([data[f]["X"], P1[f], nb_mean(P1[f], data[f]["bord"], len(P1[f]), 1), nb_mean(P1[f], data[f]["bord"], len(P1[f]), 2)])
        X2 = np.vstack([ctx(g)[data[g]["y"] > 0] for g in tr]); m2 = fit(X2, ya, wa, seed=1); P2 = proba(m2, ctx(te))
    d = data[te]; y = d["y"]; area = d["X"][:, 0]; lab = y > 0
    for name, P in (("stage 1", P1[te]), ("stage 2", P2)):
        pr = P[:, 1:].argmax(1) + 1
        print(f"== {name}: piece classes on {te} (by area, labelled pieces only): right {area[lab][pr[lab] == y[lab]].sum() / area[lab].sum():.3f}")
        for c in range(1, 6):
            rec = area[(y == c) & (pr == c)].sum() / max(area[y == c].sum(), 1e-9); prec = area[lab & (pr == c) & (y == c)].sum() / max(area[lab & (pr == c)].sum(), 1e-9)
            print(f"   {CLS[c]:8} found {rec:.3f}  correct among called {prec:.3f}  (labelled area {area[y == c].sum():.0f} m2)")
    pred = P2[:, 1:].argmax(1) + 1; np.save(out / "pred.npy", pred); np.save(out / "proba.npy", P2)
    # ---- rooms
    pk = pickle.load(open(D / f"{te}.pkl", "rb")); fs = [wkb.loads(b) for b in pk["wkb"]]; r = pk["meta"]["resize"]; se = pk["meta"]["px_per_m"]; n = len(fs)
    gts = V0.load_rooms(te); tree = STRtree(fs); seeds = {}
    for k, g in enumerate(gts):
        if g["poly"].geom_type != "Polygon" or not g.get("xy"): continue
        p = Point(g["xy"][0] * r, g["xy"][1] * r); hit = [int(i) for i in tree.query(p) if fs[int(i)].contains(p)]
        if hit: seeds[k] = min(hit, key=lambda i: fs[i].area)
    adj = [[] for _ in range(n)]
    for i, j, L, ink, gv in d["bord"]:
        c = (L * ink + 0.02 * L) / se; adj[int(i)].append((int(j), c)); adj[int(j)].append((int(i), c))
    # pieces facing each other across a drawn line whose own hair-thin body was dropped share no border on file: link them
    gp = D / f"{te}_gap.npy"
    if gp.exists(): gap = np.load(gp)
    else:
        have = set((int(min(i, j)), int(max(i, j))) for i, j in d["bord"][:, :2]); gl = []
        for i, f in enumerate(fs):
            fb = f.buffer(a.gap)
            for j in tree.query(fb):
                j = int(j)
                if j <= i or (i, j) in have: continue
                Ln = fs[j].exterior.intersection(fb).length
                if Ln >= 3.0: gl.append((i, j, Ln))
        gap = np.array(gl, np.float32).reshape(-1, 3); np.save(gp, gap)
    for i, j, Ln in gap:
        c = (Ln * 1.0 + 0.02 * Ln) / se; adj[int(i)].append((int(j), c)); adj[int(j)].append((int(i), c))
    print(f"== links: {len(d['bord'])} borders + {len(gap)} across dropped line bodies")
    floor_ok = pred == 1
    if a.thin > 0: floor_ok |= d["X"][:, 2] < a.thin          # a sliver thinner than any wall (a drawn line's own body) never stops a room
    for k, i in seeds.items(): floor_ok[i] = True
    dist = np.full(n, np.inf); own = np.full(n, -1); pq = []
    for k, i in seeds.items(): dist[i] = 0.0; own[i] = k; heapq.heappush(pq, (0.0, i, k))
    if a.seed_r > 0:                                          # the label point may sit inside a letter: start from the floor around it too
        rep = d["rep"]
        for k, i in seeds.items():
            g = gts[k]; near = np.where(floor_ok & (np.hypot(rep[:, 0] - g["xy"][0] * r, rep[:, 1] - g["xy"][1] * r) < a.seed_r * se))[0]
            for j in near:
                if dist[j] > 0: dist[j] = 0.0; own[j] = k; heapq.heappush(pq, (0.0, int(j), k))
    while pq:
        dd, i, k = heapq.heappop(pq)
        if dd > dist[i] or own[i] != k: continue
        for j, c in adj[i]:
            if floor_ok[j] and dd + c < dist[j]: dist[j] = dd + c; own[j] = k; heapq.heappush(pq, (dist[j], j, k))
    rows = []; np.save(out / "own.npy", own)
    for k, g in enumerate(gts):
        if g["poly"].geom_type != "Polygon": continue
        sel =[fs[i] for i in np.where(own == k)[0]]
        if not sel: rows.append({"room": k, "name": g["name"], "boundary": []}); continue
        U = unary_union(sel).buffer(0.01).buffer(2.0, join_style=2).buffer(-2.0, join_style=2)
        if U.geom_type != "Polygon":
            p = Point(g["xy"][0] * r, g["xy"][1] * r) if g.get("xy") else None; hit = [q for q in U.geoms if p is not None and q.contains(p)]; U = hit[0] if hit else max(U.geoms, key=lambda q: q.area)
        U = affinity.scale(type(U)(U.exterior), 1 / r, 1 / r, origin=(0, 0))          # holes filled, back to the scan's own pixels
        rows.append({"room": k, "name": g["name"], "boundary": [list(p) for p in U.simplify(0.4).exterior.coords[:-1]]})
    json.dump({"rooms": rows}, open(out / "SONUC.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"== rooms on {te}: {len(seeds)} label points placed"); S, _ = SS.score_run(out, te, 3.0, quiet=True)
    # ---- door / window openings
    L = json.load(open(V0.B / te / "LABELS_COMMON.json", encoding="utf-8")); res = {}
    for kind, c in (("door", 4), ("window", 5)):
        G = [affinity.scale(shape(s["geometry"]).buffer(0), r, r, origin=(0, 0)) for s in L["structures"] if s["kind"] == kind]
        if not G: continue
        U = unary_union([fs[i] for i in np.where(pred == c)[0]]).buffer(0.5) if (pred == c).any() else None
        found = sum(U is not None and g.intersection(U).area >= 0.5 * g.area for g in G); GU = unary_union(G).buffer(1.0)
        comps = [q for q in (list(U.geoms) if U is not None and U.geom_type != "Polygon" else ([U] if U is not None else [])) if q.area >= 0.03 * se * se]
        good = sum(q.intersection(GU).area >= 0.5 * q.area for q in comps); res[kind] = {"labelled": len(G), "found": int(found), "called": len(comps), "called_right": int(good)}
        print(f"== {kind}: {found}/{len(G)} labelled openings found | {len(comps)} regions called, {good} of them on a labelled opening")
    json.dump({"train": tr, "test": te, "openings": res, "boundary": S}, open(out / "OZET.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
