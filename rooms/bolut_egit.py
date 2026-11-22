"""Small segmentation network: reads the scan (about 100 px/m, grey) and says for every pixel
room floor / wall / column / door opening / window. Unlabelled pixels are ignored in training (unknown, not "nothing").
Its opinion is then averaged over each encoded piece (piece_proba), so outlines still come from the plan's own lines.
Run with .venv-spec (torch + cuda).  Needs parca_veri.py output for every floor used.  Never 015 unless --final.
usage: python bolut_egit.py --train 001,002,010,013,016 --test 019 --tag u1 [--steps 6000]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, pickle, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely import affinity, wkb
from shapely.geometry import shape

import v0_watershed as V0  # noqa: E402

D = config.KT / "parca_veri"; OUT = config.KT / "bolut"
KIND = {"wall": 2, "column": 3, "door": 4, "window": 5}


def ring(P): return np.round(np.array(P.exterior.coords)).astype(np.int32)


def label_raster(floor):
    """Class picture in the reduced image's pixels: 0 unlabelled, 1 room, 2 wall, 3 column, 4 door, 5 window."""
    meta = pickle.load(open(D / f"{floor}.pkl", "rb"))["meta"]; r = meta["resize"]; W, H = meta["size"]; cls = np.zeros((H, W), np.uint8)
    for g in V0.load_rooms(floor):
        P = affinity.scale(g["poly"], r, r, origin=(0, 0))
        for Q in ([P] if P.geom_type == "Polygon" else list(P.geoms)): cv2.fillPoly(cls, [ring(Q)], 1)
    L = json.load(open(V0.B / floor / "LABELS_COMMON.json", encoding="utf-8"))
    for kind in ("wall", "column", "door", "window"):
        for st in L["structures"]:
            if st["kind"] != kind: continue
            P = affinity.scale(shape(st["geometry"]).buffer(0), r, r, origin=(0, 0))
            for Q in ([P] if P.geom_type == "Polygon" else list(getattr(P, "geoms", []))):
                if Q.geom_type == "Polygon" and not Q.is_empty: cv2.fillPoly(cls, [ring(Q)], KIND[kind])
    return cls


def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(s, c=32, n=5):
        super().__init__(); s.d = nn.ModuleList([block(1, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c), block(8 * c, 8 * c)])
        s.u = nn.ModuleList([block(16 * c, 4 * c), block(8 * c, 2 * c), block(4 * c, c), block(2 * c, c)]); s.o = nn.Conv2d(c, n, 1)

    def forward(s, x):
        sk = []
        for i, b in enumerate(s.d):
            x = b(x)
            if i < 4: sk.append(x); x = F.max_pool2d(x, 2)
        for b in s.u:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False); x = b(torch.cat([x, sk.pop()], 1))
        return s.o(x)


def sample(imgs, labs, rng, size=384):
    """One training crop with the variation a new office's scan may bring: size, rotation, mirror, contrast, line weight, blur, noise."""
    k = rng.integers(len(imgs)); im, lb = imgs[k], labs[k]; z = rng.uniform(0.7, 1.4); s = int(size / z)
    for _ in range(20):
        y = rng.integers(0, max(1, im.shape[0] - s)); x = rng.integers(0, max(1, im.shape[1] - s)); l = lb[y:y + s, x:x + s]
        if (l > 0).mean() > 0.25: break
    a = im[y:y + s, x:x + s]
    if rng.random() < 0.5:                                               # thinner / heavier lines
        ker = np.ones((2, 2), np.uint8); a = cv2.erode(a, ker) if rng.random() < 0.5 else cv2.dilate(a, ker)
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR); l = cv2.resize(l, (size, size), interpolation=cv2.INTER_NEAREST)
    if a.shape != (size, size) or l.shape != (size, size): return sample(imgs, labs, rng, size)
    r = rng.integers(4); a = np.rot90(a, r); l = np.rot90(l, r)
    if rng.random() < 0.5: a = a[:, ::-1]; l = l[:, ::-1]
    a = a.astype(np.float32) / 255.0
    if rng.random() < 0.5: a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.4, 1.1))
    g = rng.uniform(0.6, 1.6); a = 1.0 - np.clip((1.0 - a) * rng.uniform(0.6, 1.5), 0, 1) ** g          # ink strength
    a = np.clip(a * rng.uniform(0.85, 1.0) + rng.normal(0, rng.uniform(0, 0.04), a.shape), 0, 1)
    return np.ascontiguousarray(a, np.float32), np.ascontiguousarray(l, np.int64)


@torch.no_grad()
def predict(net, gray, dev, tile=768, pad=96):
    """Class probabilities for a whole picture (H, W, 5), sliding window, mirrored average."""
    net.eval(); H, W = gray.shape; a = np.pad(gray.astype(np.float32) / 255.0, ((pad, pad + tile), (pad, pad + tile)), mode="constant", constant_values=1.0)
    out = np.zeros((5, H, W), np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            t = torch.from_numpy(a[y:y + tile + 2 * pad, x:x + tile + 2 * pad].copy())[None, None].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                p = F.softmax(net(t).float(), 1) + torch.flip(F.softmax(net(torch.flip(t, [3])).float(), 1), [3])
            p = (p[0] / 2)[:, pad:pad + tile, pad:pad + tile].cpu().numpy(); h = min(tile, H - y); w = min(tile, W - x); out[:, y:y + h, x:x + w] = p[:, :h, :w]
    return out


def piece_proba(floor, prob):
    """Mean class probability over each encoded piece -> (pieces, 6), column 0 unused."""
    pk = pickle.load(open(D / f"{floor}.pkl", "rb")); fs = [wkb.loads(b) for b in pk["wkb"]]; W, H = pk["meta"]["size"]; n = len(fs); idm = np.zeros((H, W), np.int32)
    for i in sorted(range(n), key=lambda i: -fs[i].area): cv2.fillPoly(idm, [ring(fs[i])], i + 1)
    flat = idm.ravel(); cnt = np.bincount(flat, minlength=n + 1).astype(np.float64); P = np.zeros((n, 6), np.float32)
    for c in range(5): P[:, c + 1] = (np.bincount(flat, weights=prob[c].ravel(), minlength=n + 1) / np.maximum(cnt, 1))[1:]
    rep = np.array([f.representative_point().coords[0] for f in fs]); rx = np.clip(rep[:, 0].round().astype(int), 0, W - 1); ry = np.clip(rep[:, 1].round().astype(int), 0, H - 1)
    z = cnt[1:] < 1; P[z, 1:] = prob[:, ry[z], rx[z]].T
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", required=True); ap.add_argument("--tag", default="u1")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=12); ap.add_argument("--final", action="store_true"); ap.add_argument("--target", type=float, default=100.0)
    a = ap.parse_args(); tr = a.train.split(","); tests = a.test.split(","); assert a.final or "015" not in tr + tests
    out = OUT / a.tag; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda:0"); print(torch.cuda.get_device_name(0), flush=True)
    imgs = [cv2.imread(str(D / f"{f}.png"), 0) for f in tr]; labs = [label_raster(f) for f in tr]
    freq = np.bincount(np.concatenate([l.ravel() for l in labs]), minlength=6)[1:].astype(np.float64); w = (freq.sum() / freq) ** 0.5; w = w / w.min(); print("class weights", np.round(w, 2), flush=True)
    net = UNet().to(dev).to(memory_format=torch.channels_last); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.steps, pct_start=0.05); sc = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(0)
    cw = torch.tensor(w, dtype=torch.float32, device=dev); t0 = time.time(); run = 0.0
    for step in range(a.steps):
        net.train(); b = [sample(imgs, labs, rng) for _ in range(a.batch)]
        x = torch.from_numpy(np.stack([q[0] for q in b]))[:, None].to(dev); y = torch.from_numpy(np.stack([q[1] for q in b])).to(dev) - 1; y[y < 0] = 255
        with torch.autocast("cuda", dtype=torch.float16): lo = net(x)
        loss = F.cross_entropy(lo.float(), y, weight=cw, ignore_index=255)
        opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 500 == 0 or step == a.steps - 1: print(f"step {step} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), out / "unet.pt")
    for f in tests:
        meta = pickle.load(open(D / f"{f}.pkl", "rb"))["meta"]; g = cv2.imread(str(D / f"{f}.png"), 0); z = a.target / meta["px_per_m"]
        gz = cv2.resize(g, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR) if abs(z - 1) > 0.05 else g
        pr = predict(net, gz, dev); pr = np.stack([cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR) for p in pr]) if gz.shape != g.shape else pr
        lb = label_raster(f); m = lb > 0; pred = pr.argmax(0) + 1; print(f"== {f}: pixels right (labelled only) {(pred[m] == lb[m]).mean():.3f}")
        for c, nm in enumerate(["oda", "duvar", "kolon", "kapi", "pencere"], 1):
            print(f"   {nm:8} found {((pred == c) & (lb == c)).sum() / max((lb == c).sum(), 1):.3f}  correct among called {((pred == c) & (lb == c)).sum() / max(((pred == c) & m).sum(), 1):.3f}")
        np.save(out / f"{f}_piece_proba.npy", piece_proba(f, pr)); cv2.imwrite(str(out / f"{f}_sinif.png"), (pred * 50).astype(np.uint8))
