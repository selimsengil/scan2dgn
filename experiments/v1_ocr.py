"""Rule-based rooms, v1: image only.

Seeds = room labels read by OCR (ocr_labels.py). Scale = the px/m that makes the
segmented rooms' areas agree with their printed areas (a search over candidates on a
downsampled raster). Labels/DGN are used only to SCORE: each DGN room is matched to the
OCR label that falls inside it; a DGN room with no OCR label inside scores 0.

usage: python v1_ocr.py [--floors 015,013] [--out out_v1]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import watershed

import ocr_labels as O
import v0_watershed as V0

B = V0.B


def wall_mask(img: np.ndarray, s: float, p, erase=()) -> np.ndarray:
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pad = max(1, int(0.03 * s))
    for (x0, y0, x1, y1) in erase:   # the room labels OCR read: text, even where it touches a line
        ink[max(0, int(y0) - pad):int(y1) + pad + 1, max(0, int(x0) - pad):int(x1) + pad + 1] = 0
    n0, cc0, st0, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    small = np.maximum(st0[:, cv2.CC_STAT_WIDTH], st0[:, cv2.CC_STAT_HEIGHT]) < p.text * s
    small[0] = False
    ink[small[cc0]] = 0
    kc = max(3, int(round(p.close * s))) | 1
    ko = max(3, int(round(p.open * s))) | 1
    a = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kc)))
    b = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kc, 1)))
    walls = cv2.morphologyEx(np.maximum(a, b), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko)))
    n, cc, st, _ = cv2.connectedComponentsWithStats((walls > 0).astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= p.minpiece * s
    return keep[cc]


def segment(img: np.ndarray, seeds: list, s: float, p, erase=()) -> tuple[np.ndarray, np.ndarray]:
    """labels: 0 wall, 1 outside, 2+i the room of seeds[i]; holes filled per room."""
    H, W = img.shape
    walls = wall_mask(img, s, p, erase)
    free = ~walls
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    markers = np.zeros((H, W), np.int32)
    markers[0, :] = markers[-1, :] = markers[:, 0] = markers[:, -1] = 1
    rr = max(2, int(0.05 * s))
    win = max(3, int(p.seed_win * s))
    for i, (x, y) in enumerate(seeds):
        # the most open point near the label: never a pocket inside a letter
        x, y = int(round(x)), int(round(y))
        x0, y0 = max(0, x - win), max(0, y - win)
        d = dist[y0:y + win + 1, x0:x + win + 1]
        if d.size:
            j = np.unravel_index(np.argmax(d), d.shape); y, x = y0 + j[0], x0 + j[1]
        cv2.circle(markers, (int(x), int(y)), rr, i + 2, -1)
    markers[~free] = 0
    lab = watershed(-dist, markers, mask=free)
    return lab, walls


def room_mask(lab: np.ndarray, i: int) -> np.ndarray:
    pred = lab == i + 2
    if pred.any():
        cnts, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        f = np.zeros(lab.shape, np.uint8); cv2.drawContours(f, cnts, -1, 1, -1); pred = f.astype(bool)
    return pred


def dedupe(labels: list[dict], tol_px: float) -> list[dict]:
    out = []
    for l in labels:
        if any((o["area_m2"] == l["area_m2"] if o["area_m2"] is None or l["area_m2"] is None else abs(o["area_m2"] - l["area_m2"]) < 0.01) and np.hypot(o["xy"][0] - l["xy"][0], o["xy"][1] - l["xy"][1]) < tol_px for o in out):
            continue
        out.append(l)
    return out


def dpi_scale(floor: str, denom: float) -> float | None:
    """px/m from the source TIF's DPI and the drawing scale 1/denom (the crop is 1:1 with the TIF)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    try:
        sb = json.load(open(B / floor / "SOURCE_BASE.json", encoding="utf-8"))
        dpi = Image.open(sb["source_tif"]).info.get("dpi")
        return float(dpi[0]) / 0.0254 / denom if dpi else None
    except Exception:
        return None


def find_scale(img: np.ndarray, labels: list[dict], p, prior: float | None = None) -> tuple[float, list]:
    """The candidate px/m whose segmentation's room areas best match the printed areas
    (candidates around the DPI prior when there is one)."""
    f = p.scale_ds
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    seeds = [(l["xy"][0] * f, l["xy"][1] * f) for l in labels]
    erase = [tuple(v * f for v in l["box"]) for l in labels]
    if not any(l["area_m2"] for l in labels):
        s0 = prior or p.default_scale
        return float(s0), [("no printed areas", s0)]
    cands = [prior * k for k in p.window] if prior else p.scales
    tried = []
    for s in cands:
        lab, _ = segment(small, seeds, s * f, p, erase)
        r = np.asarray([np.sqrt(room_mask(lab, i).sum() / (f * f) / l["area_m2"])
                        for i, l in enumerate(labels) if l["area_m2"] and l["area_m2"] > 0.5])
        near = np.abs(r / s - 1) < 0.10     # rooms whose implied scale agrees with the candidate
        tried.append((s, float(near.mean()), float(np.median(r[near])) if near.any() else None))
    best = max(tried, key=lambda t: t[1])
    if prior and best[1] < p.min_agree:   # the areas do not settle it: the drawing scale stands
        return float(prior), tried
    return float(best[2] or best[0]), tried


def merge_open(lab: np.ndarray, n: int, s: float, min_len_m: float) -> list[int]:
    """Seeds whose regions share a long boundary through open space are one room.
    Returns root[i] for each seed (union-find over regions 2..n+1)."""
    root = list(range(n))
    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]; i = root[i]
        return i
    a, b = lab[:, :-1], lab[:, 1:]
    c, d = lab[:-1, :], lab[1:, :]
    pairs = np.concatenate([np.stack([a[a != b], b[a != b]], 1), np.stack([c[c != d], d[c != d]], 1)])
    pairs = pairs[(pairs[:, 0] >= 2) & (pairs[:, 1] >= 2)] - 2
    if len(pairs):
        pairs.sort(1)
        u, cnt = np.unique(pairs, axis=0, return_counts=True)
        for (i, j), k in zip(u, cnt):
            if k >= min_len_m * s:
                root[find(i)] = find(j)
    return [find(i) for i in range(n)]


def run_floor(floor: str, p, outdir: Path) -> dict:
    img = cv2.imread(str(B / floor / "clean.png"), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    ocr = json.load(open(O.OUT / f"ocr_{floor}.json", encoding="utf-8"))
    labels = dedupe(ocr["labels"], 0.02 * max(H, W))
    if p.prior == "text":   # image only: the drawing's own text height as the unit
        hs = [np.ptp(np.asarray(b["quad"])[:, 1]) for b in ocr["boxes"]
              if len(b["text"]) >= 2 and np.ptp(np.asarray(b["quad"])[:, 0]) > np.ptp(np.asarray(b["quad"])[:, 1])]
        prior = float(np.median(hs)) / p.text_h_m if hs else None
    else:
        prior = dpi_scale(floor, p.denom) if p.denom > 0 else None
    s, tried = find_scale(img, labels, p, prior)
    if p.oracle_scale:   # diagnostic only: the DGN rooms' scale
        s = V0.oracle_scale(V0.load_rooms(floor))
    lab, walls = segment(img, [l["xy"] for l in labels], s, p, [l["box"] for l in labels])
    root = list(range(len(labels)))
    if p.merge_m > 0:   # seeds in one open space: one room
        root = merge_open(lab, len(labels), s, p.merge_m)
        remap = np.arange(len(labels) + 2); remap[2:] = np.asarray(root) + 2
        lab = remap[lab]
    gts = V0.load_rooms(floor)
    s_oracle = V0.oracle_scale(gts)
    res = []
    for g in gts:
        gm = V0.poly_mask(g["poly"], (H, W))
        inside = [i for i, l in enumerate(labels)
                  if 0 <= int(l["xy"][1]) < H and 0 <= int(l["xy"][0]) < W and gm[int(l["xy"][1]), int(l["xy"][0])]]
        if not inside:
            res.append({"id": g["id"], "name": g["name"], "iou": 0.0, "why": "no OCR label inside"}); continue
        i = inside[0]
        pm = room_mask(lab, root[i])
        inter = np.logical_and(pm, gm).sum(); uni = np.logical_or(pm, gm).sum()
        res.append({"id": g["id"], "name": g["name"], "iou": round(float(inter / uni), 4), "ocr": labels[i]["name"],
                    "area_text": labels[i]["area_m2"], "area_pred": round(float(pm.sum() / s / s), 2)})
    # overlay
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) // 2 + 127
    vis[walls] = (90, 90, 90)
    rng = np.random.default_rng(0)
    for i, l in enumerate(labels):
        m = lab == i + 2
        vis[m] = (vis[m] * 0.45 + rng.integers(60, 230, 3) * 0.55).astype(np.uint8)
        cv2.circle(vis, (int(l["xy"][0]), int(l["xy"][1])), max(4, int(0.08 * s)), (0, 160, 0), -1)
    for g in gts:
        if g["poly"].geom_type == "Polygon":
            cv2.polylines(vis, [np.round(np.asarray(g["poly"].exterior.coords)).astype(np.int32)], True, (0, 0, 255), max(2, int(s * 0.03)))
    sc = min(1.0, 3000 / max(H, W))
    cv2.imwrite(str(outdir / f"{floor}_overlay.png"), cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
    ious = np.array([x["iou"] for x in res])
    return {"summary": {"floor": floor, "ocr_labels": len(labels), "scale_px_m": round(s, 1),
                        "scale_oracle": round(s_oracle, 1), "rooms": len(res),
                        "no_label": int(sum(1 for x in res if x.get("why"))),
                        "iou_mean": round(float(ious.mean()), 3), "n_ge95": int((ious >= 0.95).sum()),
                        "n_ge90": int((ious >= 0.90).sum()), "n_ge80": int((ious >= 0.80).sum())},
            "scale_search": tried, "rooms": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default=",".join(V0.FLOORS))
    ap.add_argument("--close", type=float, default=0.24)
    ap.add_argument("--open", type=float, default=0.09)
    ap.add_argument("--minpiece", type=float, default=1.2)
    ap.add_argument("--text", type=float, default=0.45)
    ap.add_argument("--scale_ds", type=float, default=0.5)
    ap.add_argument("--oracle_scale", action="store_true")
    ap.add_argument("--denom", type=float, default=50, help="drawing scale 1/denom for the DPI prior; 0 = none")
    ap.add_argument("--prior", default="text", choices=["text", "dpi"], help="text: image only (default)")
    ap.add_argument("--text_h_m", type=float, default=0.22, help="typical text line height in m (unit only)")
    ap.add_argument("--window", default="0.65,0.72,0.8,0.88,0.95,1.0,1.05,1.12,1.2,1.3,1.4,1.5")
    ap.add_argument("--min_agree", type=float, default=0.3, help="share of printed areas needed to move off the prior")
    ap.add_argument("--merge_m", type=float, default=1.3, help="m of open shared boundary that makes two seeds one room; 0 = off")
    ap.add_argument("--seed_win", type=float, default=0.6, help="m: the seed moves to the most open point this near")
    ap.add_argument("--default_scale", type=float, default=150, help="px/m when no area is printed (provisional)")
    ap.add_argument("--scales", default="60,70,80,90,100,115,130,150,170,200,230,260,300,340,380")
    ap.add_argument("--out", default=str(Path(__file__).with_name("out_v1")))
    p = ap.parse_args()
    p.scales = [float(x) for x in p.scales.split(",")]
    p.window = [float(x) for x in p.window.split(",")]
    outdir = Path(p.out); outdir.mkdir(parents=True, exist_ok=True)
    tot = {"rooms": 0, "no_label": 0, "n_ge95": 0, "n_ge90": 0, "n_ge80": 0}
    allr = {}
    for f in p.floors.split(","):
        r = run_floor(f, p, outdir); allr[f] = r
        print(json.dumps(r["summary"], ensure_ascii=False), flush=True)
        for k in tot: tot[k] += r["summary"][k]
    print("TOTAL", json.dumps(tot))
    json.dump({"args": {k: v for k, v in vars(p).items()}, "total": tot, "floors": allr},
              open(outdir / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
