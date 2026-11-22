"""Segmentation network trained on the floors read from the user's own DGN drawings (dgn_kat.py).
The drawing of a floor is complete, so every pixel has a class:
  1 room floor (OdaAlani)   2 other: walls, columns, shafts, outside   3 balcony / common area   4 door opening   5 window
(the order the room builder parca_egit.py expects: column 1 = room).
Tested on floors prepared by parca_veri.py (their pieces), e.g. the old 019 / 017 with printed label points.
Run with .venv-spec.  The blind floor is never in kat_test/dgn_katlar unless it was written on purpose.
usage: python bolut_dgn.py --skip Kisla --test 019,017 --tag d1 [--steps 8000]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, pickle, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import bolut_egit as BE  # noqa: E402

R = config.KT / "dgn_katlar"; OUT = config.KT / "bolut"
CLS = {"room": 1, "balcony": 3, "common": 3, "door": 4, "window": 5}; BAND = 0.0


def load_floor(fid, target=100.0):
    L = json.load(open(R / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]; r = target / s if s > 130 else 1.0
    g = cv2.imdecode(np.fromfile(str(R / fid / "clean.png"), np.uint8), 0)
    if r != 1.0: g = cv2.resize(g, (int(round(g.shape[1] * r)), int(round(g.shape[0] * r))), interpolation=cv2.INTER_AREA)
    lab = np.full(g.shape, 2 if BAND <= 0 else 0, np.uint8)
    for kind in ("room", "common", "balcony", "door", "window"):
        for o in (L["rooms"] if kind == "room" else [x for x in L["structures"] if x["kind"] == kind]):
            c = (o.get("interior_polygon") or o["geometry"])["coordinates"][0]; cv2.fillPoly(lab, [np.round(np.array(c) * r).astype(np.int32)], CLS[kind])
    if BAND > 0:      # "other" only close to what was drawn (walls, columns, shafts between rooms); the paper further out stays unknown
        k = int(BAND * (s * r)) * 2 + 1; near = cv2.dilate((lab > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        lab[(near > 0) & (lab == 0)] = 2
    return g, lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip", default="Kisla", help="projects left out of training (comma separated name parts)"); ap.add_argument("--test", default="019,017")
    ap.add_argument("--tag", default="d1"); ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--band", type=float, default=0.0, help="metres around the drawing that count as 'other'; 0 = everything not drawn")
    ap.add_argument("--merge", action="store_true", help="room, balcony and common area are one class 'floor' (they look the same; the name tells them apart)"); ap.add_argument("--edge", type=float, default=0.0, help="weight of pixels near the drawn boundary (0 = off)")
    a = ap.parse_args(); BAND = a.band
    if a.merge: CLS.update(balcony=1, common=1)
    skip = [s for s in a.skip.split(",") if s]; out = OUT / a.tag; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    if a.steps == 0:                       # only predict with the saved network
        net = BE.UNet().to(dev).to(memory_format=torch.channels_last); net.load_state_dict(torch.load(out / "unet.pt", map_location=dev))
    idx = [] if a.steps == 0 else [i for i in json.load(open(R / "INDEX.json", encoding="utf-8")) if i.get("room", 0) > 0 and not any(s.lower() in i["floor"].lower() for s in skip)]
    print("training floors", len(idx), "rooms", sum(i["room"] for i in idx), "| left out:", skip, flush=True)
    imgs, labs = [], []
    for i in idx:
        g, l = load_floor(i["floor"]); imgs.append(g); labs.append(l)
    if a.steps > 0: freq = np.bincount(np.concatenate([l.ravel()[::7] for l in labs]), minlength=6)[1:].astype(np.float64); w = np.where(freq > 0, (freq.sum() / np.maximum(freq, 1)) ** 0.5, 0.0); w = w / w[w > 0].min(); print("class weights (room, other, balcony/common, door, window)", np.round(w, 2), flush=True)
    if a.steps > 0:
        net = BE.UNet().to(dev).to(memory_format=torch.channels_last); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.steps, pct_start=0.05); sc = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(0)
        cw = torch.tensor(w, dtype=torch.float32, device=dev); t0 = time.time(); run = 0.0
    for step in range(a.steps):
        net.train(); b = [BE.sample(imgs, labs, rng) for _ in range(a.batch)]
        x = torch.from_numpy(np.stack([q[0] for q in b]))[:, None].to(dev); y = torch.from_numpy(np.stack([q[1] for q in b])).to(dev) - 1; y[y < 0] = 255
        with torch.autocast("cuda", dtype=torch.float16): lo = net(x)
        if a.edge > 0:      # pixels within 15 cm of a class change (the drawn boundary itself) count a.edge times more
            yf = y.float()[:, None]; pw = 1 + (a.edge - 1) * ((F.max_pool2d(yf, 31, 1, 15) != -F.max_pool2d(-yf, 31, 1, 15))[:, 0]).float()
            l = F.cross_entropy(lo.float(), y, weight=cw, ignore_index=255, reduction="none"); loss = (l * pw).sum() / (pw * (y != 255)).sum().clamp(min=1)
        else: loss = F.cross_entropy(lo.float(), y, weight=cw, ignore_index=255)
        opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 1000 == 0 or step == a.steps - 1: print(f"step {step} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
    if a.steps > 0: torch.save(net.state_dict(), out / "unet.pt")
    for f in [t for t in a.test.split(",") if t]:
        meta = pickle.load(open(BE.D / f"{f}.pkl", "rb"))["meta"]; g = cv2.imread(str(BE.D / f"{f}.png"), 0); z = 100.0 / meta["px_per_m"]
        gz = cv2.resize(g, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR) if abs(z - 1) > 0.05 else g
        pr = BE.predict(net, gz, dev); pr = np.stack([cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR) for p in pr]) if gz.shape != g.shape else pr
        np.save(out / f"{f}_piece_proba.npy", BE.piece_proba(f, pr)); cv2.imwrite(str(out / f"{f}_sinif.png"), ((pr.argmax(0) + 1) * 50).astype(np.uint8)); print("predicted", f, flush=True)
