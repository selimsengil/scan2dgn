"""Rule-based room boundaries, v0.

Input per floor: clean.png and the printed room-label points (from the labels for now:
this stands in for OCR). Scale (px/m) is taken from the labels too (ORACLE, diagnostic
only; production gets it from OCR'd area text).

Pipeline: ink -> close (wall faces into solid walls) -> open (drop thin axis / dimension
/ furniture lines) -> watershed of free space on its distance transform, one marker per
room label + the page border as outside. Rooms are scored against the user's DGN rooms.

usage: python v0_watershed.py [--floors 015,013] [--close 0.24] [--open 0.07] [--out DIR]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, shape
from skimage.segmentation import watershed

B = Path(os.environ.get("KAT_ROOT", r"C:\Users\Selim\Documents\cizim_projeler\egitim_verisi\sam_etiket_hassas_v3_20261022\floors"))   # KAT_ROOT: another floors folder (kat_test/dgn_katlar)
FLOORS = ["001", "002", "010", "013", "015", "016", "019"]


def load_rooms(floor: str) -> list[dict]:
    p = B / floor / "LABELS_COMMON.json"
    rooms = json.load(open(p, encoding="utf-8"))["rooms"]
    if not rooms:
        rooms = json.load(open(B / floor / "ROOMS_USER_DGN.json", encoding="utf-8"))["rooms"]
    out = []
    for r in rooms:
        try:
            poly = shape(r["interior_polygon"])
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        pt = r.get("point") or {}
        xy = pt.get("xy") if isinstance(pt, dict) else None
        if xy is None:
            c = poly.representative_point(); xy = [c.x, c.y]
        try:
            a = float(str(r.get("area_text_value")).replace(",", "."))
        except Exception:
            a = None
        out.append({"id": r["id"], "name": r.get("name"), "poly": poly, "xy": xy, "area_m2": a})
    return out


def oracle_scale(rooms: list[dict]) -> float:
    v = [np.sqrt(r["poly"].area / r["area_m2"]) for r in rooms if r["area_m2"] and r["area_m2"] > 0.5]
    return float(np.median(v))


def poly_mask(poly, shape_hw) -> np.ndarray:
    m = np.zeros(shape_hw, np.uint8)
    polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], 1)
        for h in p.interiors:
            cv2.fillPoly(m, [np.round(np.asarray(h.coords)).astype(np.int32)], 0)
    return m.astype(bool)


def run_floor(floor: str, a, outdir: Path) -> dict:
    img = cv2.imread(str(B / floor / "clean.png"), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    rooms = load_rooms(floor)
    s = oracle_scale(rooms)
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # characters and small symbols: separate raw-ink pieces smaller than a text line
    n0, cc0, st0, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    small = np.maximum(st0[:, cv2.CC_STAT_WIDTH], st0[:, cv2.CC_STAT_HEIGHT]) < a.text * s
    small[0] = False
    ink[small[cc0]] = 0
    if a.dump:
        cv2.imwrite(str(outdir / f"{floor}_ink.png"), cv2.resize(255 - ink, None, fx=min(1, 3000 / max(H, W)), fy=min(1, 3000 / max(H, W)), interpolation=cv2.INTER_AREA))
    kc = max(3, int(round(a.close * s))) | 1
    ko = max(3, int(round(a.open * s))) | 1
    # a wall is two parallel faces: close across the gap in one direction at a time
    cv_ = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kc)))
    ch_ = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kc, 1)))
    solid = np.maximum(cv_, ch_)
    walls = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ko, ko)))
    # text, symbols, furniture: pieces not part of the wall system
    n, cc, st, _ = cv2.connectedComponentsWithStats((walls > 0).astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= a.minpiece * s
    walls = np.where(keep[cc], 255, 0).astype(np.uint8)
    free = walls == 0
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    markers = np.zeros((H, W), np.int32)
    markers[0, :] = markers[-1, :] = markers[:, 0] = markers[:, -1] = 1   # outside
    rr = max(2, int(0.05 * s))
    fy, fx = np.nonzero(free[::4, ::4])
    for i, r in enumerate(rooms):
        x, y = int(round(r["xy"][0])), int(round(r["xy"][1]))
        if not free[min(max(y, 0), H - 1), min(max(x, 0), W - 1)]:   # on a wall piece: nearest free pixel
            j = np.argmin((fx * 4 - x) ** 2 + (fy * 4 - y) ** 2); x, y = int(fx[j] * 4), int(fy[j] * 4)
        cv2.circle(markers, (x, y), rr, i + 2, -1)
    markers[~free] = 0
    lab = watershed(-dist, markers, mask=free)
    res = []
    for i, r in enumerate(rooms):
        pred = lab == i + 2
        if pred.any():   # holes inside the room (text blobs, furniture) belong to it
            cnts, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros((H, W), np.uint8)
            cv2.drawContours(filled, cnts, -1, 1, -1)
            pred = filled.astype(bool)
        gt = poly_mask(r["poly"], (H, W))
        inter = np.logical_and(pred, gt).sum(); uni = np.logical_or(pred, gt).sum()
        iou = inter / uni if uni else 0.0
        area_m2 = pred.sum() / s / s
        res.append({"id": r["id"], "name": r["name"], "iou": round(float(iou), 4),
                    "area_pred_m2": round(float(area_m2), 2), "area_text_m2": r["area_m2"]})
    # overlay
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) // 2 + 127
    vis[walls > 0] = (90, 90, 90)
    rng = np.random.default_rng(0)
    for i, r in enumerate(rooms):
        col = rng.integers(60, 230, 3)
        m = lab == i + 2
        vis[m] = (vis[m] * 0.45 + col * 0.55).astype(np.uint8)
        gt = np.round(np.asarray(r["poly"].exterior.coords)).astype(np.int32) if r["poly"].geom_type == "Polygon" else None
        if gt is not None:
            cv2.polylines(vis, [gt], True, (0, 0, 255), max(2, int(s * 0.03)))
    sc = min(1.0, 3000 / max(H, W))
    cv2.imwrite(str(outdir / f"{floor}_overlay.png"), cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
    ious = np.array([x["iou"] for x in res])
    summ = {"floor": floor, "scale_px_m_oracle": round(s, 1), "rooms": len(res),
            "iou_mean": round(float(ious.mean()), 3), "n_ge95": int((ious >= 0.95).sum()),
            "n_ge90": int((ious >= 0.90).sum()), "n_ge80": int((ious >= 0.80).sum())}
    return {"summary": summ, "rooms": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors", default=",".join(FLOORS))
    ap.add_argument("--close", type=float, default=0.24, help="m: gaps closed into solid walls")
    ap.add_argument("--open", type=float, default=0.07, help="m: thinner than this is not a wall")
    ap.add_argument("--minpiece", type=float, default=1.2, help="m: wall pieces shorter than this are dropped")
    ap.add_argument("--text", type=float, default=0.45, help="m: raw ink pieces smaller than this are text")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--out", default=str(config.DATA / "out_v0"))
    a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    allr, tot = {}, {"rooms": 0, "n_ge95": 0, "n_ge90": 0, "n_ge80": 0}
    for f in a.floors.split(","):
        r = run_floor(f, a, outdir)
        allr[f] = r
        print(json.dumps(r["summary"], ensure_ascii=False), flush=True)
        for k in tot: tot[k] += r["summary"][k]
    print("TOTAL", json.dumps(tot))
    json.dump({"args": vars(a), "total": tot, "floors": allr}, open(outdir / "RESULT.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
