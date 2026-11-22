"""Tag finder: marks the door/window tag text ('90/210', 'K2 90/220', 'P1 150/140') on the scan, no openings needed.
Target at 100 px/m: 2 = tag region (verified tags of kat_test/ocr_test/ETIKETLER.json: the read's box when the tag is
beside the opening, the opening box when it is written inside the gap), 1 = background (other texts included),
0 = ignored (1 m around openings whose tag was not verified: a tag is probably there but unread).
Then each found region is cut level (+ margin), read by GLM-OCR and PaddleOCR-VL (0/180 deg), numbers parsed.
usage: python etiket_bulucu.py egit [--steps 8000] | bul FLOOR      (.venv-spec, 5070 Ti only)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, os, re, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
from shapely.geometry import Polygon, shape
import bolut_egit as BE  # noqa: E402
K = config.KT; OUT = K / "etiket_bulucu"; TGT = 100.0


def scale_of(fid):
    L = json.load(open(K / "dgn_katlar" / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
    return s, (TGT / s if s > 130 else 1.0)


def gray(fid, r):
    g = cv2.imdecode(np.fromfile(str(K / "dgn_katlar" / fid / "clean.png"), np.uint8), 0)
    return cv2.resize(g, (int(round(g.shape[1] * r)), int(round(g.shape[0] * r))), interpolation=cv2.INTER_AREA) if r != 1.0 else g


def poly_px(p, r):
    P = shape(p) if isinstance(p, dict) else Polygon(p)
    return np.round(np.asarray(P.exterior.coords) * r).astype(np.int32)


def load(fid, E):
    s, r = scale_of(fid); g = gray(fid, r); lab = np.ones(g.shape, np.uint8)
    V = {o["id"]: o for o in json.load(open(K / "aciklik_ag" / "veri" / f"{fid}.json", encoding="utf-8"))["openings"]}
    for t in E[fid]:
        if t["derece"] is None and t["id"] in V: cv2.fillPoly(lab, [poly_px(shape(V[t["id"]]["poly"]).buffer(1.0 * s), r)], 0)
    n = 0
    for t in E[fid]:
        if t["derece"] is None: continue
        cv2.fillPoly(lab, [poly_px(t["box"], r)], 2); n += 1
    return g, lab, n


def sample(imgs, labs, rng, size=384):
    k = rng.integers(len(imgs)); im, lb = imgs[k], labs[k]; H, W = im.shape
    if rng.random() < 0.6:
        ys, xs = np.where(lb == 2)
        if len(ys): i = rng.integers(len(ys)); y = int(ys[i] - rng.integers(40, size - 40)); x = int(xs[i] - rng.integers(40, size - 40))
        else: y, x = rng.integers(0, H - size), rng.integers(0, W - size)
    else: y, x = rng.integers(0, max(1, H - size)), rng.integers(0, max(1, W - size))
    y = int(np.clip(y, 0, H - size)); x = int(np.clip(x, 0, W - size)); a = im[y:y + size, x:x + size]; l = lb[y:y + size, x:x + size]
    if rng.random() < 0.3:
        ker = np.ones((2, 2), np.uint8); a = cv2.erode(a, ker) if rng.random() < 0.5 else cv2.dilate(a, ker)
    r = rng.integers(4); a, l = np.rot90(a, r), np.rot90(l, r)
    if rng.random() < 0.5: a, l = a[:, ::-1], l[:, ::-1]
    a = a.astype(np.float32) / 255.0
    if rng.random() < 0.4: a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.4, 1.0))
    a = 1.0 - np.clip((1.0 - a) * rng.uniform(0.7, 1.4), 0, 1)
    return np.ascontiguousarray(a), np.ascontiguousarray(l, np.int64)


def egit(steps, tag):
    E = json.load(open(K / "ocr_test" / "ETIKETLER.json", encoding="utf-8")); imgs, labs = [], []; nt = 0
    for fid in E:
        g, l, n = load(fid, E)
        if n: imgs.append(g); labs.append(l); nt += n
    print("katlar", len(imgs), "etiket", nt, flush=True)
    dev = torch.device("cuda:0"); assert "5070" in torch.cuda.get_device_name(0)
    net = BE.UNet(n=2).to(dev).to(memory_format=torch.channels_last); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.05); sc = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(0)
    cw = torch.tensor([1.0, 3.0], device=dev); t0 = time.time(); run = 0.0
    for st in range(steps):
        net.train(); b = [sample(imgs, labs, rng) for _ in range(12)]
        x = torch.from_numpy(np.stack([q[0] for q in b]))[:, None].to(dev); y = torch.from_numpy(np.stack([q[1] for q in b])).to(dev) - 1; y[y < 0] = 255
        with torch.autocast("cuda", dtype=torch.float16): lo = net(x)
        loss = F.cross_entropy(lo.float(), y, weight=cw, ignore_index=255)
        opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); run = 0.98 * run + 0.02 * loss.item() if st else loss.item()
        if st % 500 == 0 or st == steps - 1: print(f"step {st} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
    (OUT / tag).mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), OUT / tag / "unet.pt")


@torch.no_grad()
def regions(fid, tag, thr=0.5, min_area=40):
    """Tag regions on the sheet (sheet px): [{box: 4 pts, score}]"""
    dev = torch.device("cuda:0"); net = BE.UNet(n=2).to(dev); net.load_state_dict(torch.load(OUT / tag / "unet.pt", map_location=dev)); net.eval()
    s, r = scale_of(fid); g = gray(fid, r); tile, pad = 768, 96; H, W = g.shape
    a = np.pad(g.astype(np.float32) / 255.0, ((pad, pad + tile), (pad, pad + tile)), constant_values=1.0); prob = np.zeros((H, W), np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            t = torch.from_numpy(a[y:y + tile + 2 * pad, x:x + tile + 2 * pad].copy())[None, None].to(dev)
            with torch.autocast("cuda", dtype=torch.float16): q = F.softmax(net(t).float(), 1)[0, 1] + torch.flip(F.softmax(net(torch.flip(t, [3])).float(), 1)[0, 1], [1])
            q = (q / 2)[pad:pad + tile, pad:pad + tile].cpu().numpy(); h = min(tile, H - y); w = min(tile, W - x); prob[y:y + h, x:x + w] = q[:h, :w]
    m = (prob > thr).astype(np.uint8); n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8); out = []
    for i in range(1, n):
        if st[i, 4] < min_area: continue
        ys, xs = np.where(lb == i); rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32)); box = cv2.boxPoints(rect) / r
        out.append({"box": box.tolist(), "score": float(prob[lb == i].mean())})
    return out, prob


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("floor", nargs="?"); ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--tag", default="e1"); a = ap.parse_args()
    if a.mode == "egit": egit(a.steps, a.tag)
    else:
        reg, prob = regions(a.floor, a.tag); (OUT / a.tag).mkdir(parents=True, exist_ok=True)
        json.dump(reg, open(OUT / a.tag / f"{a.floor}_bolgeler.json", "w"), indent=0); np.save(OUT / a.tag / f"{a.floor}_prob.npy", prob.astype(np.float16)); print(len(reg), "bolge")
