"""Network that marks the room boundary LINE itself (the office's DGN room outline), not room areas.
Target per pixel: 2 = on a DGN room outline (3 cm wide, also across door openings), 1 = not, 0 = ignored
(outline of a room whose DGN does not sit on the scan's ink: gt_denetim oturma < --min-oturma).
Measures on the test floors: recall = share of the clean GT outline with a predicted line pixel within 3 px,
precision = share of predicted line pixels within 3 px of some GT outline.   Run with .venv-spec.
usage: python cizgi_ag.py --skip Kisla --test <floors> --tag c1 [--steps 10000]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from shapely.geometry import shape

import bolut_egit as BE  # noqa: E402

R = config.KT / "dgn_katlar"; OUT = config.KT / "cizgi_ag"
AUD = json.load(open(config.KT / "gt_denetim" / "DENETIM.json", encoding="utf-8"))


def load(fid, min_ot, target=100.0):
    L = json.load(open(R / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]; r = target / s if s > 130 else 1.0
    g = cv2.imdecode(np.fromfile(str(R / fid / "clean.png"), np.uint8), 0)
    if r != 1.0: g = cv2.resize(g, (int(round(g.shape[1] * r)), int(round(g.shape[0] * r))), interpolation=cv2.INTER_AREA)
    ot = {a["id"]: a["oturma"] for a in AUD[fid]["rooms"]}; lab = np.ones(g.shape, np.uint8); gt = np.zeros(g.shape, np.uint8); clean = np.zeros(g.shape, np.uint8)
    w = max(2, int(round(0.03 * s * r)))
    rings = []
    for rm in L["rooms"]:
        P = shape(rm["interior_polygon"]).buffer(0)
        if P.is_empty: continue
        for p in ([P] if P.geom_type == "Polygon" else list(P.geoms)): rings.append((rm["id"], np.round(np.asarray(p.exterior.coords) * r).astype(np.int32)))
    for rid, c in rings:
        if ot.get(rid, 0) < min_ot: cv2.polylines(lab, [c], True, 0, 3 * w)
    for rid, c in rings:
        cv2.polylines(gt, [c], True, 1, 1)
        if ot.get(rid, 0) >= min_ot: cv2.polylines(lab, [c], True, 2, w); cv2.polylines(clean, [c], True, 1, 1)
    return g, lab, gt, clean


@torch.no_grad()
def predict(net, gray, dev, tile=768, pad=96):
    net.eval(); H, W = gray.shape; a = np.pad(gray.astype(np.float32) / 255.0, ((pad, pad + tile), (pad, pad + tile)), mode="constant", constant_values=1.0); out = np.zeros((2, H, W), np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            t = torch.from_numpy(a[y:y + tile + 2 * pad, x:x + tile + 2 * pad].copy())[None, None].to(dev)
            with torch.autocast("cuda", dtype=torch.float16): q = F.softmax(net(t).float(), 1) + torch.flip(F.softmax(net(torch.flip(t, [3])).float(), 1), [3])
            q = (q[0] / 2)[:, pad:pad + tile, pad:pad + tile].cpu().numpy(); h = min(tile, H - y); w = min(tile, W - x); out[:, y:y + h, x:x + w] = q[:, :h, :w]
    return out


def measure(prob, gt, clean, tol=3.0, thr=0.5):
    pred = (prob > thr).astype(np.uint8)
    dp = cv2.distanceTransform(1 - pred, cv2.DIST_L2, 5); dg = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 5)
    rec = float((dp[clean > 0] <= tol).mean()); prec = float((dg[pred > 0] <= tol).mean()) if pred.any() else 0.0
    return rec, prec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip", default="Kisla"); ap.add_argument("--test", required=True); ap.add_argument("--tag", default="c1")
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--batch", type=int, default=12); ap.add_argument("--min-oturma", type=float, default=0.95); ap.add_argument("--posw", type=float, default=4.0)
    a = ap.parse_args(); skip = [s for s in a.skip.split(",") if s]; out = OUT / a.tag; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda:0")
    net = BE.UNet(n=2).to(dev).to(memory_format=torch.channels_last)
    if a.steps > 0:
        idx = [i["floor"] for i in json.load(open(R / "INDEX.json", encoding="utf-8")) if i.get("room", 0) > 0 and not any(s.lower() in i["floor"].lower() for s in skip)]
        imgs, labs = [], []
        for f in idx: g, l, _, _ = load(f, a.min_oturma); imgs.append(g); labs.append(l)
        print("training floors", len(idx), "| line pixels share", round(float(np.mean([(l == 2).mean() for l in labs])), 4), flush=True)
        opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.steps, pct_start=0.05)
        sc = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(0); cw = torch.tensor([1.0, a.posw], device=dev); t0 = time.time(); run = 0.0
        for step in range(a.steps):
            net.train(); b = [BE.sample(imgs, labs, rng) for _ in range(a.batch)]
            x = torch.from_numpy(np.stack([q[0] for q in b]))[:, None].to(dev); y = torch.from_numpy(np.stack([q[1] for q in b])).to(dev) - 1; y[y < 0] = 255
            with torch.autocast("cuda", dtype=torch.float16): lo = net(x)
            loss = F.cross_entropy(lo.float(), y, weight=cw, ignore_index=255)
            opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            if step % 1000 == 0 or step == a.steps - 1: print(f"step {step} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
        torch.save(net.state_dict(), out / "unet.pt")
    else: net.load_state_dict(torch.load(out / "unet.pt", map_location=dev))
    res = {}
    for f in [t for t in a.test.split(",") if t]:
        g, l, gt, clean = load(f, a.min_oturma); pr = predict(net, g, dev)
        p = pr[1] / np.maximum(pr[0] + pr[1], 1e-6); np.save(out / f"{f}_cizgi.npy", p.astype(np.float16))
        res[f] = {f"thr{t}": measure(p, gt, clean, thr=t) for t in (0.3, 0.5, 0.7)}
        v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); v = (v * 0.5 + 127).astype(np.uint8); pm = p > 0.5
        dg = cv2.distanceTransform(1 - gt, cv2.DIST_L2, 5); dp = cv2.distanceTransform(1 - pm.astype(np.uint8), cv2.DIST_L2, 5)
        v[(clean > 0) & (dp > 3)] = (0, 140, 255)            # GT outline missed: orange
        v[pm & (dg <= 3)] = (200, 90, 0)                     # predicted, right: blue
        v[pm & (dg > 3)] = (0, 0, 230)                       # predicted, not a GT outline: red
        cv2.imwrite(str(out / f"{f}_cizgi.png"), v); print(f, json.dumps(res[f]), flush=True)
    json.dump(res, open(out / "OLCUM.json", "w"), indent=1)
