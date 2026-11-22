"""Opening net a3: scan + room outlines + TAG hints in, door / window boxes out.  The tags tell the net where an opening is
and how wide it is; the net's job is to draw the box right (jamb to jamb, face to face).
Input 4 channels at 100 px/m: gray scan, room outlines (2 px), tag mask (tag text box filled), tag width (first number / 300
inside the tag box).  Training tags: the verified tags of ocr_test/ETIKETLER.json.  A tag written inside the gap is given as a
text-sized box (not the opening box, or the net would copy it), every tag is jittered, 20 % are dropped, false tags are added
(production tags come from OCR + image matching and are imperfect).
Target: 1 background, 2 door, 3 window, 0 ignored: unreliable DGN openings (+15 cm) and a 2 px band on every box outline
(Citymax accepts a range of drawings; a few px either way is not an error).
usage: python aciklik_ag3.py egit --tag a3 [--steps 12000]  |  python aciklik_ag3.py tahmin --tag a3 --kat KEY [--etiketsiz]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
from shapely.affinity import scale as sscale
from shapely.geometry import Polygon, shape
import aciklik_ag as AA  # noqa: E402
import bolut_egit as BE  # noqa: E402
AA.GAP = False                                     # touching openings are one piece (user, 22.11)
K = config.KT; OUT = K / "aciklik_ag"
ET = json.load(open(K / "ocr_test" / "ETIKETLER.json", encoding="utf-8"))
SKIP = ("kisla", "malkoc_361_79__k2")
BOX3 = os.environ.get("A3_BOX3", "1") == "1"                    # new box extraction (union pieces, L split, no grow)
NIN = int(os.environ.get("A3_NIN", "4"))                         # input channels: 4 = scan, rooms, tag mask, tag width; 2 = scan, rooms
CW = float(os.environ.get("A3_CW", "3.0")); POS = float(os.environ.get("A3_POS", "0.75")); KOSE = os.environ.get("A3_KOSE", "0") == "1"
NOISE_L, DROP = float(os.environ.get("A3_NOISE", "1.5")), float(os.environ.get("A3_DROP", "0.2"))   # a3: ~1.5 false tags / crop, a4: 6 / 0.3


class Net3(BE.UNet):
    def __init__(s, c=32, nin=None):
        super().__init__(c=c, n=3); s.d[0] = BE.block(nin or NIN, c)


def frame(c):
    c = np.asarray(c, float)[:4]; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e))
    u = (c[(k + 1) % 4] - c[k]) / e[k]; return c.mean(0), u, max(e), min(e)


def rect(cen, u, L, T):
    n = np.array([-u[1], u[0]]); return np.array([cen - u * L / 2 - n * T / 2, cen + u * L / 2 - n * T / 2, cen + u * L / 2 + n * T / 2, cen - u * L / 2 + n * T / 2])


def load3(fid):
    g, lab, rooms, ops, bad = AA.load(fid); d = json.load(open(AA.V / f"{fid}.json", encoding="utf-8")); s = d["px_per_m"]; r = 100.0 / s if s > 130 else 1.0
    for o in ops:                                   # tolerance band on the outline
        for c in AA.rings(o["poly"]): cv2.polylines(lab, [np.round(c).astype(np.int32)], True, 0, 2)
    if KOSE:                                        # openings meeting at an angle are two openings: background seam at the joint
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                A_, B_ = ops[i]["poly"], ops[j]["poly"]
                if A_.distance(B_) > 3: continue
                ua = frame(np.asarray(A_.minimum_rotated_rectangle.exterior.coords))[1]; ub = frame(np.asarray(B_.minimum_rotated_rectangle.exterior.coords))[1]
                if abs(ua @ ub) > 0.5: continue                      # in line (collinear neighbours stay one piece)
                J = A_.buffer(2).intersection(B_.buffer(2))
                for c in AA.rings(J.buffer(1)): cv2.fillPoly(lab, [np.round(c).astype(np.int32)], 1)
    V = {o["id"]: o for o in d["openings"]}; tags = []
    for t in ET.get(fid, []):
        if t.get("derece") is None or t["id"] not in V: continue
        P = sscale(shape(V[t["id"]]["poly"]), r, r, origin=(0, 0)); w = t["sayilar"][0]
        if t["yer"] == "kutu_disi":
            b = np.asarray(t["box"], float) * r; tags.append({"tip": "kutu", "box": b, "w": w})
        else:
            cen, u, L, T = frame(np.asarray(P.minimum_rotated_rectangle.exterior.coords)); tags.append({"tip": "ici", "cen": cen, "u": u, "L": L, "T": T, "w": w})
    return g, lab, rooms, ops, bad, tags, r


def tag_channels(shape_hw, tags, rng=None, off=(0, 0), r=1.0):
    m = np.zeros(shape_hw, np.float32); wch = np.zeros(shape_hw, np.float32); ox, oy = off
    for t in tags:
        if rng is not None and rng.random() < DROP: continue
        if t["tip"] == "ici":                       # text-sized box inside the gap, never the opening box itself
            Lf = rng.uniform(0.45, 0.8) if rng is not None else 0.6; Tf = rng.uniform(8, 14) if rng is not None else 11
            cen = t["cen"] + (t["u"] * rng.normal(0, 3) if rng is not None else 0); b = rect(cen, t["u"], t["L"] * Lf, Tf)
        else:
            b = np.asarray(t["box"], float)
            if rng is not None: b = b + rng.normal(0, 2, 2)
        b = np.round(b - [ox, oy]).astype(np.int32); cv2.fillPoly(m, [b], 1.0); cv2.fillPoly(wch, [b], t["w"] / 300.0)
    if rng is not None:                             # false tags (OCR noise): number pairs that belong to no opening
        for _ in range(rng.poisson(NOISE_L)):
            cx, cy = rng.uniform(0, shape_hw[1]), rng.uniform(0, shape_hw[0]); ang = rng.choice([0, np.pi / 2]) + rng.normal(0, 0.05)
            u = np.array([np.cos(ang), np.sin(ang)]); b = np.round(rect(np.array([cx, cy]), u, rng.uniform(25, 70), rng.uniform(8, 14))).astype(np.int32)
            cv2.fillPoly(m, [b], 1.0); cv2.fillPoly(wch, [b], rng.choice([60, 80, 90, 100, 120, 150, 200]) / 300.0)
    return m, wch


def sample3(F_, rng, size=384):
    k = rng.integers(len(F_)); g, lab, rooms, ops, _, tags, r = F_[k]; z = rng.uniform(0.75, 1.35); s = int(size / z); H, W = g.shape
    if ops and rng.random() < POS:
        c = ops[rng.integers(len(ops))]["poly"].centroid; y = int(c.y - rng.integers(s // 6, 5 * s // 6)); x = int(c.x - rng.integers(s // 6, 5 * s // 6))
    else: y = rng.integers(0, max(1, H - s)); x = rng.integers(0, max(1, W - s))
    y = int(np.clip(y, 0, max(0, H - s))); x = int(np.clip(x, 0, max(0, W - s)))
    a = g[y:y + s, x:x + s]; l = lab[y:y + s, x:x + s]
    if a.shape != (s, s): return sample3(F_, rng, size)
    rc = AA.room_channel(a.shape, [c for c in rooms if c[:, 0].max() >= x and c[:, 0].min() < x + s and c[:, 1].max() >= y and c[:, 1].min() < y + s], rng, (x, y)).astype(np.float32)
    tm, tw = tag_channels(a.shape, tags, rng, (x, y))
    if rng.random() < 0.1: tm[:] = 0; tw[:] = 0                      # sometimes no tags at all
    if rng.random() < 0.5: ker = np.ones((2, 2), np.uint8); a = cv2.erode(a, ker) if rng.random() < 0.5 else cv2.dilate(a, ker)
    rs = lambda q, interp: cv2.resize(q, (size, size), interpolation=interp)
    a = rs(a, cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR); l = rs(l, cv2.INTER_NEAREST); rc = (rs(rc, cv2.INTER_AREA) > 0.2).astype(np.float32)
    tm = rs(tm, cv2.INTER_NEAREST); tw = rs(tw, cv2.INTER_NEAREST)
    if rng.random() < AA.ROT:
        M = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-45, 45), 1.0); wa = lambda q, i, b: cv2.warpAffine(q, M, (size, size), flags=i, borderValue=b)
        a = wa(a, cv2.INTER_LINEAR, 255); l = wa(l, cv2.INTER_NEAREST, 0); rc = wa(rc, cv2.INTER_NEAREST, 0); tm = wa(tm, cv2.INTER_NEAREST, 0); tw = wa(tw, cv2.INTER_NEAREST, 0)
    rr = rng.integers(4); a, l, rc, tm, tw = [np.rot90(q, rr) for q in (a, l, rc, tm, tw)]
    if rng.random() < 0.5: a, l, rc, tm, tw = [q[:, ::-1] for q in (a, l, rc, tm, tw)]
    a = a.astype(np.float32) / 255.0
    if rng.random() < 0.5: a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.4, 1.1))
    gm = rng.uniform(0.6, 1.6); a = 1.0 - np.clip((1.0 - a) * rng.uniform(0.6, 1.5), 0, 1) ** gm
    a = np.clip(a * rng.uniform(0.85, 1.0) + rng.normal(0, rng.uniform(0, 0.04), a.shape), 0, 1)
    X = np.stack([a, rc, tm, tw]) if NIN == 4 else np.stack([a, rc])
    return np.ascontiguousarray(X, np.float32), np.ascontiguousarray(l, np.int64)


def egit(tag, steps):
    dev = torch.device("cuda:0"); assert "5070" in torch.cuda.get_device_name(0)
    idx = [p.stem for p in sorted(AA.V.glob("*.json")) if not any(k in p.stem.lower() for k in SKIP)]
    Fl = [load3(f) for f in idx]; Fl = [f for f in Fl if f[3]]; (OUT / tag).mkdir(parents=True, exist_ok=True)
    json.dump({"egitim_katlari": idx, "not": "Malkoc K2 ve Kisla disarida; Malkoc K1/K3 ayni plan (sizinti)"}, open(OUT / tag / "EGITIM_KATLARI.json", "w", encoding="utf-8"), indent=1)
    print("katlar", len(Fl), "aciklik", sum(len(f[3]) for f in Fl), "etiket", sum(len(f[5]) for f in Fl), flush=True)
    net = Net3().to(dev).to(memory_format=torch.channels_last); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.05); sc = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(0)
    cw = torch.tensor([1.0, CW, CW], device=dev); t0 = time.time(); run = 0.0
    json.dump({"NIN": NIN, "CW": CW, "POS": POS, "KOSE": KOSE, "NOISE_L": NOISE_L, "DROP": DROP, "steps": steps}, open(OUT / tag / "AYAR.json", "w"), indent=1)
    for st in range(steps):
        net.train(); b = [sample3(Fl, rng) for _ in range(12)]
        x = torch.from_numpy(np.stack([q[0] for q in b])).to(dev); y = torch.from_numpy(np.stack([q[1] for q in b])).to(dev) - 1; y[y < 0] = 255
        with torch.autocast("cuda", dtype=torch.float16): lo = net(x)
        loss = F.cross_entropy(lo.float(), y, weight=cw, ignore_index=255)
        opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); run = 0.98 * run + 0.02 * loss.item() if st else loss.item()
        if st % 500 == 0 or st == steps - 1: print(f"step {st} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(net.state_dict(), OUT / tag / "unet.pt")


def test_tags(key, r):
    """Production-like tag hints for a test floor: number-pair reads of the chain (esle) + cheap OCRs, sheet px -> net px."""
    import re
    sys.path.insert(0, str(K / "ocr_test")); import test2
    out = []
    for name in ("esle", "rapidocr_4yon", "doctr_db50_parseq", "florence2_large"):
        p = K / "ocr_test" / "SONUC2" / f"{name}__{key}.json"
        if not p.exists(): continue
        for rd in json.load(open(p, encoding="utf-8"))["okumalar"]:
            ps = test2.pairs(rd["text"]); m = re.match(r"^(\d{2,3})/\?$", rd["text"])
            ps = [p for p in ps if 50 <= p[0] <= 300 and 140 <= p[1] <= 260]       # plausible width / height only
            w = ps[0][0] if ps else (int(m.group(1)) if m else None)
            if w is None or not 40 <= w <= 300: continue
            b = np.asarray(rd["box"], float) * r
            if any(np.hypot(*(b.mean(0) - q["box"].mean(0))) < 6 for q in out): continue
            out.append({"tip": "kutu", "box": b, "w": w})
    return out


def tahmin(tag, key, etiketsiz=False):
    sys.path.insert(0, str(K / "ocr_test")); import test2
    fid = test2.FLOORS[key]; dev = torch.device("cuda:0"); net = Net3().to(dev); net.load_state_dict(torch.load(OUT / tag / "unet.pt", map_location=dev))
    g, lab, rooms, ops, bad, _, r = load3(fid)
    if key == "malkoc_k2":                          # hand-label rooms (production: rooms come from the line net)
        E = json.load(open(K / "el_etiket" / "malkoc_k2" / "ETIKET.json", encoding="utf-8")); rooms = [np.asarray(x["poly"], float) * r for x in E["rooms"]]
    rc = AA.room_channel(g.shape, rooms).astype(np.float32); tags = [] if etiketsiz else test_tags(key, r)
    tm, tw = tag_channels(g.shape, tags)
    x_in = np.stack([g.astype(np.float32) / 255.0, rc, tm, tw]) if NIN == 4 else np.stack([g.astype(np.float32) / 255.0, rc])
    pr = AA.predict(net, x_in, dev); pred = boxes3(pr) if BOX3 else AA.boxes(pr)
    for p in pred: p["box"] = (np.asarray(p["box"]) / r).tolist(); p["p1"] = (np.asarray(p["p1"]) / r).tolist(); p["p2"] = (np.asarray(p["p2"]) / r).tolist()
    name = f"{tag}{'_etiketsiz' if etiketsiz else ''}{'_k3' if BOX3 else ''}{os.environ.get('A3_SUFFIX', '')}"
    json.dump({"tahmin": pred, "etiket_sayisi": len(tags)}, open(OUT / tag / f"tahmin_{name}__{key}.json", "w")); print(key, name, len(pred), "kutu,", len(tags), "etiket ipucu")
    return pred


# ---- box extraction v3 (no retraining): door+window pieces together, type by majority; L-shaped pieces split along the
# sheet's main wall directions; no growing (the net's overshoot is taken off later by aciklik_son.kisalt)
def _ana_aci(prob, thr=0.5):
    m = ((prob[1] + prob[2]) > thr).astype(np.uint8); n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8); angs, ws = [], []
    for i in range(1, n):
        if st[i, 4] < 60: continue
        ys, xs = np.where(lb == i); (cx, cy), (w, h), a = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        if w * h > 0 and st[i, 4] / (w * h) > 0.8: angs.append(a % 90); ws.append(st[i, 4])
    if not angs: return 0.0
    hist = np.bincount(np.round(np.asarray(angs)).astype(int) % 90, weights=ws, minlength=90); return float(np.argmax(hist))


def _rect(xs, ys):
    rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32)); c = cv2.boxPoints(rect)
    e = [np.hypot(*(c[(j + 1) % 4] - c[j])) for j in range(4)]; j = int(np.argmin(e))
    return c, (c[j] + c[(j + 1) % 4]) / 2, (c[(j + 2) % 4] + c[(j + 3) % 4]) / 2


def boxes3(prob, min_area=60, thr=0.5, fill_min=None, kol=35):
    fill_min = float(os.environ.get("A3_FILL", "0.72")) if fill_min is None else fill_min
    out = []; m = ((prob[1] + prob[2]) > thr).astype(np.uint8); ang = _ana_aci(prob, thr)
    n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        if st[i, 4] < min_area: continue
        ys, xs = np.where(lb == i)
        dmask = prob[1][ys, xs] >= prob[2][ys, xs]; fd = dmask.mean()
        if 0.25 <= fd <= 0.75:                                    # a door and a window side by side: two openings
            for sel in (dmask, ~dmask):
                sub = np.zeros(lb.shape, np.uint8); sub[ys[sel], xs[sel]] = 1; k2, lb2, st2, _ = cv2.connectedComponentsWithStats(sub, 8)
                for j in range(1, k2):
                    if st2[j, 4] < min_area: continue
                    yy, xx = np.where(lb2 == j); kind = "door" if sel is dmask else "window"; c, p1, p2 = _rect(xx, yy)
                    out.append({"kind": kind, "p1": p1.tolist(), "p2": p2.tolist(), "box": c.tolist(), "score": float((prob[1] + prob[2])[yy, xx].mean())})
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        parts = [(xs, ys)]
        if w * h > 0 and st[i, 4] / (w * h) < fill_min:          # L / T shape: split into straight arms
            x0, y0 = xs.min(), ys.min(); sub = (lb[y0:ys.max() + 1, x0:xs.max() + 1] == i).astype(np.uint8)
            pad = 40; sub = cv2.copyMakeBorder(sub, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0); hh, ww = sub.shape
            R = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0); Ri = cv2.invertAffineTransform(R); rs_ = cv2.warpAffine(sub, R, (ww, hh), flags=cv2.INTER_NEAREST)
            arms = []
            for ker in (np.ones((1, kol), np.uint8), np.ones((kol, 1), np.uint8)):
                a_ = cv2.morphologyEx(rs_, cv2.MORPH_OPEN, ker); k2, lb2, st2, _ = cv2.connectedComponentsWithStats(a_, 8)
                for j in range(1, k2):
                    if st2[j, 4] < min_area: continue
                    yy, xx = np.where(lb2 == j); P = np.stack([xx, yy], 1).astype(np.float64) @ Ri[:, :2].T + Ri[:, 2]
                    arms.append((P[:, 0] + x0 - pad, P[:, 1] + y0 - pad))
            if len(arms) >= 2: parts = arms
        for xs_, ys_ in parts:
            xi = np.clip(np.round(xs_).astype(int), 0, prob.shape[2] - 1); yi = np.clip(np.round(ys_).astype(int), 0, prob.shape[1] - 1)
            kind = "door" if prob[1][yi, xi].sum() >= prob[2][yi, xi].sum() else "window"
            c, p1, p2 = _rect(xs_, ys_); out.append({"kind": kind, "p1": p1.tolist(), "p2": p2.tolist(), "box": c.tolist(), "score": float((prob[1] + prob[2])[yi, xi].mean())})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("--tag", default="a3"); ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--kat", default="malkoc_k2"); ap.add_argument("--etiketsiz", action="store_true"); a = ap.parse_args()
    if a.mode == "egit": egit(a.tag, a.steps)
    else: tahmin(a.tag, a.kat, a.etiketsiz)
