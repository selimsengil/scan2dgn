"""Opening net: scan + room outlines in, door / window boxes out.
Input  2 channels at 100 px/m: the scan (gray) and the reliable room outlines (aciklik_veri.py), 2 px lines.  During
       training each room outline is shifted a few px, rooms are dropped now and then (production outlines are imperfect
       and incomplete).  Where an opening is, comes from the scan; the outlines only help the box ends sit on the faces.
Target per pixel: 1 background, 2 door box, 3 window box, 0 ignored (unreliable DGN openings, widened 15 cm).
Measure on the test floors: reliable labelled openings found (aciklik_olc._match, jamb error), predictions that hit no
labelled opening at all (false), predictions on unreliable labels are not counted.   Run with .venv-spec.
usage: python aciklik_ag.py --test <floors> --tag a1 [--steps 12000]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.affinity import scale as sscale
from shapely.geometry import Point, shape

import bolut_egit as BE  # noqa: E402

R = config.KT / "dgn_katlar"; V = config.KT / "aciklik_ag" / "veri"; OUT = config.KT / "aciklik_ag"
KIND = {"door": 2, "window": 3}
GAP = True; GROW = 2.0; ROT = 0.3        # a2: seams between boxes, boxes grown back by GROW px, free rotation share


class Net(BE.UNet):
    def __init__(s, c=32):
        super().__init__(c=c, n=3); s.d[0] = BE.block(2, c)


def rings(P):
    P = P.buffer(0)
    return [np.asarray(p.exterior.coords) for p in ([P] if P.geom_type == "Polygon" else list(getattr(P, "geoms", [])))]


def load(fid, target=100.0):
    d = json.load(open(V / f"{fid}.json", encoding="utf-8")); s = d["px_per_m"]; r = target / s if s > 130 else 1.0
    g = cv2.imdecode(np.fromfile(str(R / fid / "clean.png"), np.uint8), 0)
    if r != 1.0: g = cv2.resize(g, (int(round(g.shape[1] * r)), int(round(g.shape[0] * r))), interpolation=cv2.INTER_AREA)
    lab = np.ones(g.shape, np.uint8); sc = lambda P: sscale(shape(P), r, r, origin=(0, 0))
    for b in d["bad"]:
        for c in rings(sc(b["poly"]).buffer(0.15 * s * r)): cv2.fillPoly(lab, [np.round(c).astype(np.int32)], 0)
    ops = []
    for o in d["openings"]:
        P = sc(o["poly"]); ops.append({"kind": o["kind"], "poly": P})
        for c in rings(P): cv2.fillPoly(lab, [np.round(c).astype(np.int32)], KIND[o["kind"]])
    if GAP:                                                                  # thin background seam round every box, so
        for o in ops:                                                        # touching openings come out as separate pieces
            for c in rings(o["poly"]): cv2.polylines(lab, [np.round(c).astype(np.int32)], True, 1, 2)
    rooms = [rings(sc(x["poly"]))[0] for x in d["rooms"]]
    bad = [{"kind": b["kind"], "poly": sc(b["poly"])} for b in d["bad"]]
    return g, lab, rooms, ops, bad


def room_channel(shape_hw, rooms, rng=None, off=(0, 0)):
    ch = np.zeros(shape_hw, np.uint8)
    drop_all = rng is not None and rng.random() < 0.1
    for c in rooms:
        if rng is not None and (drop_all or rng.random() < 0.15): continue
        q = c - np.array(off, float)
        if rng is not None: q = q + rng.normal(0, 1.5, 2) + rng.normal(0, 0.7, q.shape)       # whole-room shift + vertex wobble (px at 1 cm)
        cv2.polylines(ch, [np.round(q).astype(np.int32)], True, 1, 2)
    return ch


def sample(F_, rng, size=384):
    k = rng.integers(len(F_)); g, lab, rooms, ops, _ = F_[k]; z = rng.uniform(0.75, 1.35); s = int(size / z); H, W = g.shape
    if ops and rng.random() < 0.7:                                           # most crops around an opening
        c = ops[rng.integers(len(ops))]["poly"].centroid; y = int(c.y - rng.integers(s // 6, 5 * s // 6)); x = int(c.x - rng.integers(s // 6, 5 * s // 6))
    else: y = rng.integers(0, max(1, H - s)); x = rng.integers(0, max(1, W - s))
    y = int(np.clip(y, 0, max(0, H - s))); x = int(np.clip(x, 0, max(0, W - s)))
    a = g[y:y + s, x:x + s]; l = lab[y:y + s, x:x + s]; rc = room_channel(a.shape, [c for c in rooms if c[:, 0].max() >= x and c[:, 0].min() < x + s and c[:, 1].max() >= y and c[:, 1].min() < y + s], rng, (x, y))
    if a.shape != (s, s): return sample(F_, rng, size)
    if rng.random() < 0.5: ker = np.ones((2, 2), np.uint8); a = cv2.erode(a, ker) if rng.random() < 0.5 else cv2.dilate(a, ker)
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR); l = cv2.resize(l, (size, size), interpolation=cv2.INTER_NEAREST)
    rc = (cv2.resize(rc.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.2).astype(np.uint8)
    if rng.random() < ROT:                                                   # slanted walls: free rotation
        M = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-45, 45), 1.0)
        a = cv2.warpAffine(a, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=255); l = cv2.warpAffine(l, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
        rc = cv2.warpAffine(rc, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    r = rng.integers(4); a, l, rc = np.rot90(a, r), np.rot90(l, r), np.rot90(rc, r)
    if rng.random() < 0.5: a, l, rc = a[:, ::-1], l[:, ::-1], rc[:, ::-1]
    a = a.astype(np.float32) / 255.0
    if rng.random() < 0.5: a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.4, 1.1))
    gm = rng.uniform(0.6, 1.6); a = 1.0 - np.clip((1.0 - a) * rng.uniform(0.6, 1.5), 0, 1) ** gm
    a = np.clip(a * rng.uniform(0.85, 1.0) + rng.normal(0, rng.uniform(0, 0.04), a.shape), 0, 1)
    return np.ascontiguousarray(np.stack([a, rc.astype(np.float32)]), np.float32), np.ascontiguousarray(l, np.int64)


@torch.no_grad()
def predict(net, x2, dev, tile=768, pad=96):
    net.eval(); _, H, W = x2.shape; a = np.pad(x2, ((0, 0), (pad, pad + tile), (pad, pad + tile))); a[0, :pad] = a[0, :, :pad] = 1.0
    a[0, pad + H:] = 1.0; a[0, :, pad + W:] = 1.0; out = np.zeros((3, H, W), np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            t = torch.from_numpy(a[:, y:y + tile + 2 * pad, x:x + tile + 2 * pad].copy())[None].to(dev)
            with torch.autocast("cuda", dtype=torch.float16): q = F.softmax(net(t).float(), 1) + torch.flip(F.softmax(net(torch.flip(t, [3])).float(), 1), [3])
            q = (q[0] / 2)[:, pad:pad + tile, pad:pad + tile].cpu().numpy(); h = min(tile, H - y); w = min(tile, W - x); out[:, y:y + h, x:x + w] = q[:, :h, :w]
    return out


def boxes(prob, min_area=60, thr=0.5):
    """Connected pieces of each class -> minimum rotated rectangle, its long axis as jambs p1, p2."""
    out = []; cls = prob.argmax(0)
    for kind, k in (("door", 1), ("window", 2)):
        m = ((cls == k) & (prob[k] > thr)).astype(np.uint8); n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if st[i, 4] < min_area: continue
            ys, xs = np.where(lb == i); rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            rect = (rect[0], (rect[1][0] + GROW, rect[1][1] + GROW), rect[2]); c = cv2.boxPoints(rect)
            e = [np.hypot(*(c[(j + 1) % 4] - c[j])) for j in range(4)]; j = int(np.argmin(e))
            p1 = (c[j] + c[(j + 1) % 4]) / 2; p2 = (c[(j + 2) % 4] + c[(j + 3) % 4]) / 2
            out.append({"kind": kind, "p1": p1.tolist(), "p2": p2.tolist(), "box": c.tolist(), "score": float(prob[k][lb == i].mean())})
    return out


def axis(P):
    r = P.minimum_rotated_rectangle; c = np.array(r.exterior.coords)[:4]; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmin(e))
    a = (c[k] + c[(k + 1) % 4]) / 2; b = (c[(k + 2) % 4] + c[(k + 3) % 4]) / 2; return a, b


def measure(pred, ops, bad, tol_jamb=5.0):
    import aciklik_olc as AO
    G = [{"kind": o["kind"], "poly": o["poly"], "p1": axis(o["poly"])[0], "p2": axis(o["poly"])[1]} for o in ops]
    for g in G: g["width"] = float(np.hypot(*(g["p2"] - g["p1"])))
    res = {}
    for kind in ("door", "window"):
        Gk = [g for g in G if g["kind"] == kind]; P = [p for p in pred if p["kind"] == kind]; used = set(); je = []
        for g in Gk:
            best = None
            for i, p in enumerate(P):
                if i in used: continue
                m = AO._match((p["p1"], p["p2"]), g, tol=10.0)
                if m and (best is None or m["jamb_err"] < best[1]["jamb_err"]): best = (i, m)
            if best: used.add(best[0]); je.append(best[1]["jamb_err"])
        # a prediction is false only when it hits no labelled opening at all (reliable or not, any kind)
        anyl = [o["poly"].buffer(10) for o in ops + bad]
        false = [i for i, p in enumerate(P) if i not in used and not any(L.contains(Point(*((np.array(p["p1"]) + np.array(p["p2"])) / 2))) for L in anyl)]
        res[kind] = {"etiket": len(Gk), "bulunan": len(je), "recall": round(len(je) / max(1, len(Gk)), 3), "tahmin": len(P), "yanlis": len(false),
                     "kasa_hata_med_px": round(float(np.median(je)), 1) if je else None, "kasa_5px_ici": int(sum(v <= tol_jamb for v in je))}
    return res


def sheet(g, rooms, pred, ops, bad, path):
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); v = (v * 0.55 + 115).astype(np.uint8)
    for c in rooms: cv2.polylines(v, [np.round(c).astype(np.int32)], True, (230, 160, 60), 1)
    for o in bad: cv2.polylines(v, [np.round(rings(o["poly"])[0]).astype(np.int32)], True, (160, 160, 160), 1)
    for o in ops: cv2.polylines(v, [np.round(rings(o["poly"])[0]).astype(np.int32)], True, (0, 150, 0), 1)
    for p in pred: cv2.polylines(v, [np.round(np.array(p["box"])).astype(np.int32)], True, (0, 0, 230) if p["kind"] == "door" else (200, 0, 200), 2)
    cv2.imwrite(str(path), v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip", default="Kisla"); ap.add_argument("--test", required=True); ap.add_argument("--tag", default="a1")
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--batch", type=int, default=12)
    a = ap.parse_args(); skip = [s for s in a.skip.split(",") if s]; out = OUT / a.tag; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda:0")
    net = Net().to(dev).to(memory_format=torch.channels_last)
    if a.steps > 0:
        idx = [p.stem for p in sorted(V.glob("*.json")) if not any(s.lower() in p.stem.lower() for s in skip)]
        Fl = [load(f) for f in idx]; Fl = [f for f in Fl if f[3]]
        print("training floors", len(Fl), "| openings", sum(len(f[3]) for f in Fl), "| door/window px share", round(float(np.mean([(f[1] >= 2).mean() for f in Fl])), 4), flush=True)
        json.dump({"egitim_katlari": idx}, open(out / "EGITIM_KATLARI.json", "w", encoding="utf-8"), indent=1)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.steps, pct_start=0.05)
        sc = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(0); cw = torch.tensor([1.0, 3.0, 3.0], device=dev); t0 = time.time(); run = 0.0
        for step in range(a.steps):
            net.train(); b = [sample(Fl, rng) for _ in range(a.batch)]
            x = torch.from_numpy(np.stack([q[0] for q in b])).to(dev); y = torch.from_numpy(np.stack([q[1] for q in b])).to(dev) - 1; y[y < 0] = 255
            with torch.autocast("cuda", dtype=torch.float16): lo = net(x)
            loss = F.cross_entropy(lo.float(), y, weight=cw, ignore_index=255)
            opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            if step % 500 == 0 or step == a.steps - 1: print(f"step {step} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
        torch.save(net.state_dict(), out / "unet.pt")
    else: net.load_state_dict(torch.load(out / "unet.pt", map_location=dev))
    res = {}
    for f in [t for t in a.test.split(",") if t]:
        g, lab, rooms, ops, bad = load(f)
        for mode in ("oda_var", "oda_yok"):
            rc = room_channel(g.shape, rooms if mode == "oda_var" else [])
            pr = predict(net, np.stack([g.astype(np.float32) / 255.0, rc.astype(np.float32)]), dev); pred = boxes(pr)
            res[f"{f}|{mode}"] = measure(pred, ops, bad); print(f, mode, json.dumps(res[f"{f}|{mode}"]), flush=True)
            if mode == "oda_var": sheet(g, rooms, pred, ops, bad, out / f"{f}_aciklik.png"); json.dump(pred, open(out / f"{f}_tahmin.json", "w"))
    json.dump(res, open(out / "OLCUM.json", "w"), indent=1)
