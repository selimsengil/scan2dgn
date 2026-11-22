"""Tools for room drawing on one floor. The model chooses; the tools do the geometry.

Kat(floor) keeps the state:
  walls0      the wall mask the rules find (ink -> solid walls, thin lines dropped)
  openings    numbered neck candidates (doors, passages): a closing segment each
  closed      which openings are closed (default: all)
  deleted     wall pieces the model removed (stamps, furniture)
  labels      OCR room labels (seeds)
segment() -> per-seed room regions with measured vs printed area.
render_*() -> images with numbered candidates for the model.
Everything is image-only; the DGN rooms are used by score() alone.
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

import v0_watershed as V0
import v1_ocr as V1
import ocr_labels as O
import walls as WL

B = V0.B


@dataclass
class Opening:
    id: int
    p: tuple            # neck point (x, y)
    a: tuple            # segment end on one wall face
    b: tuple            # ... and on the other
    width_m: float
    in_wall_line: bool = True   # a gap in a wall line (closed by default); else only a candidate


@dataclass
class Params:
    close: float = 0.24
    open: float = 0.09
    minpiece: float = 1.2
    text: float = 0.45
    seed_win: float = 0.6
    neck_min: float = 0.55     # m: narrower than this is not a passage (a gap in a wall line)
    neck_max: float = 1.7      # m: wider than this is open plan, not a door
    neck_drop: float = 0.85    # a neck is at most this share of the width around it
    neck_win: float = 0.9      # m: "around it"
    scale_ds: float = 0.5
    scales: tuple = ()
    window: tuple = (0.65, 0.72, 0.8, 0.88, 0.95, 1.0, 1.05, 1.12, 1.2, 1.3, 1.4, 1.5)
    min_agree: float = 0.3
    text_h_m: float = 0.22
    default_scale: float = 150
    merge_m: float = 1.3
    absorb_share: float = 0.6


class Kat:
    def __init__(self, floor: str, p: Params | None = None):
        self.floor, self.p = floor, p or Params()
        self.img = cv2.imread(str(B / floor / "clean.png"), cv2.IMREAD_GRAYSCALE)
        self.H, self.W = self.img.shape
        ocr = json.load(open(O.OUT / f"ocr_{floor}.json", encoding="utf-8"))
        self.boxes = ocr["boxes"]
        self.labels = V1.dedupe(ocr["labels"], 0.02 * max(self.H, self.W))
        self.p.absorb_share = 0.0
        self.s = self._scale(ocr)
        self._build()
        # the scale again, from the rooms themselves: median of printed vs measured area
        for _ in range(2):
            rooms = self.segment()
            r = [np.sqrt(x["area_m2"] / x["area_text"]) for x in rooms
                 if x["area_text"] and 0.8 < x["area_text"] < 200 and x["area_m2"] > 0.3]
            r = [v for v in r if 0.6 < v < 1.8]
            if len(r) < 4:
                break
            f = float(np.median(r))
            if abs(f - 1) < 0.02:
                break
            self.s *= f
            self._build()
        self.lab = None

    def _build(self):
        self.pid, self.pieces, self._keep = WL.wall_pieces(self.img, self.s, erase=[l["box"] for l in self.labels])
        self.walls0 = WL.mask_without(self._keep, set())
        self.openings = self.find_openings()
        self.closed = {o.id for o in self.openings if o.in_wall_line}
        self.deleted: set[int] = set()
        self.lab = None

    # ---------------------------------------------------------------- scale --
    def _scale(self, ocr) -> float:
        hs = [np.ptp(np.asarray(b["quad"])[:, 1]) for b in ocr["boxes"]
              if len(b["text"]) >= 2 and np.ptp(np.asarray(b["quad"])[:, 0]) > np.ptp(np.asarray(b["quad"])[:, 1])]
        prior = float(np.median(hs)) / self.p.text_h_m if hs else None
        s, _ = V1.find_scale(self.img, self.labels, self.p, prior)
        return s

    # ------------------------------------------------------------- openings --
    def find_openings(self) -> list[Opening]:
        """Necks of the free space: points on its medial axis where the width dips to a
        door's (neck_min..neck_max) and is clearly narrower than nearby."""
        s, p = self.s, self.p
        free = ~self.walls0
        dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
        sk = skeletonize(free)
        w = 2 * dist / s                                   # width in m at each axis pixel
        r = max(2, int(p.neck_win * s))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        big = np.where(sk, w, 0).astype(np.float32)
        wmax = cv2.dilate(big, k)                          # widest axis point around
        small = np.where(sk, w, 99).astype(np.float32)
        wmin = cv2.erode(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, r // 2) | 1,) * 2))
        cand = sk & (w >= p.neck_min) & (w <= p.neck_max) & (w <= wmin + 1e-3) & (w <= p.neck_drop * wmax)
        ys, xs = np.nonzero(cand)
        order = np.argsort(w[ys, xs])
        taken = np.zeros(len(xs), bool)
        wm = self.walls0
        lim = int(p.neck_max * s * 1.2)
        out = []

        def run(x, y, dx, dy):
            """free pixels from (x, y) to the first wall along (dx, dy); None if none near"""
            for t in range(1, lim):
                xx, yy = x + dx * t, y + dy * t
                if not (0 <= xx < self.W and 0 <= yy < self.H):
                    return None
                if wm[yy, xx]:
                    return t
            return None

        for j in order:
            if taken[j]:
                continue
            x, y = int(xs[j]), int(ys[j])
            taken |= (xs - x) ** 2 + (ys - y) ** 2 < (0.5 * s) ** 2
            # the closing line is horizontal or vertical: the shorter wall-to-wall span
            best = None
            for (dx, dy) in ((1, 0), (0, 1)):
                t1, t2 = run(x, y, dx, dy), run(x, y, -dx, -dy)
                if t1 is None or t2 is None:
                    continue
                span = (t1 + t2) / s
                if p.neck_min <= span <= p.neck_max and (best is None or span < best[0]):
                    best = (span, (x + dx * t1, y + dy * t1), (x - dx * t2, y - dy * t2))
            if best is None:
                continue
            # a door is a gap IN a wall line: past each end the wall goes on along the line
            (ax, ay), (bx, by) = best[1], best[2]
            ux, uy = np.sign(ax - bx), np.sign(ay - by)
            gap = self._wall_goes_on(ax, ay, ux, uy) and self._wall_goes_on(bx, by, -ux, -uy)
            out.append(Opening(len(out) + 1, (x, y), best[1], best[2], round(best[0], 2), gap))
        return out

    def _wall_goes_on(self, x, y, ux, uy, need_m: float = 0.3, share: float = 0.85) -> bool:
        n = max(3, int(need_m * self.s)); hit = 0
        for t in range(n):
            xx, yy = int(x + ux * t), int(y + uy * t)
            if 0 <= xx < self.W and 0 <= yy < self.H and self.walls0[yy, xx]:
                hit += 1
        return hit >= share * n

    # ---------------------------------------------------------------- state --
    def walls(self) -> np.ndarray:
        wm = WL.mask_without(self._keep, self.deleted) if self.deleted else self.walls0.copy()
        t = max(2, int(0.04 * self.s))
        canvas = wm.astype(np.uint8)
        for o in self.openings:
            if o.id in self.closed:
                cv2.line(canvas, o.a, o.b, 1, t)
        if getattr(self, "extra", None) is not None:   # oracle experiments: barriers from the labels
            canvas |= self.extra.astype(np.uint8)
        return canvas.astype(bool)

    def segment(self) -> list[dict]:
        wm = self.walls()
        free = ~wm
        dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
        markers = np.zeros((self.H, self.W), np.int32)
        markers[0, :] = markers[-1, :] = markers[:, 0] = markers[:, -1] = 1
        win = max(3, int(self.p.seed_win * self.s)); rr = max(2, int(0.05 * self.s))
        for i, l in enumerate(self.labels):
            x, y = int(l["xy"][0]), int(l["xy"][1])
            x0, y0 = max(0, x - win), max(0, y - win)
            d = dist[y0:y + win + 1, x0:x + win + 1]
            if d.size:
                j = np.unravel_index(np.argmax(d), d.shape); y, x = y0 + j[0], x0 + j[1]
            cv2.circle(markers, (int(x), int(y)), rr, i + 2, -1)
        markers[~free] = 0
        lab = watershed(-dist, markers, mask=free)
        root = V1.merge_open(lab, len(self.labels), self.s, self.p.merge_m) if self.p.merge_m > 0 else list(range(len(self.labels)))
        remap = np.arange(len(self.labels) + 2); remap[2:] = np.asarray(root) + 2
        self.lab, self.root = remap[lab], root
        self.interior = self._absorb_interior(wm)
        rooms = []
        for i, l in enumerate(self.labels):
            m = self.room_mask(i)
            a = m.sum() / self.s ** 2
            rooms.append({"i": i, "name": l["name"], "area_text": l["area_m2"], "area_m2": round(float(a), 2),
                          "ok": bool(l["area_m2"] and abs(a / l["area_m2"] - 1) < 0.05)})
        return rooms

    def _absorb_interior(self, wm: np.ndarray) -> list[int]:
        """A wall piece with one and the same room on every side separates nothing: it is
        furniture (or a stub) inside that room, and the room takes its pixels."""
        # atoms: each wall piece, and each free pocket no seed reached (a bed's inside)
        r = max(2, int(0.04 * self.s))
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        atoms = []
        for pc in self.pieces:
            if pc["id"] in self.deleted:
                continue
            lut, cc = self._keep[pc["layer"]]
            x0, y0, x1, y1 = pc["bbox"]
            atoms.append(("p", pc["id"], (x0, y0, x1, y1), lambda c=cc, j=pc["j"], b=(x0, y0, x1, y1): c[b[1]:b[3], b[0]:b[2]] == j))
        pocket = ((self.lab == 0) & ~wm).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(pocket, 8)
        for j in range(1, n):
            x, y, w, h, a = st[j]
            atoms.append(("f", j, (x, y, x + w, y + h), lambda c=cc, j=j, b=(x, y, x + w, y + h): c[b[1]:b[3], b[0]:b[2]] == j))
        took, done = [], set()
        for _ in range(6):   # a bed: its frame joins the room first, then its inside
            changed = False
            for t, aid, (x0, y0, x1, y1), get in atoms:
                if (t, aid) in done:
                    continue
                X0, Y0, X1, Y1 = max(0, x0 - 2 * r), max(0, y0 - 2 * r), min(self.W, x1 + 2 * r), min(self.H, y1 + 2 * r)
                sub = np.zeros((Y1 - Y0, X1 - X0), bool)
                sub[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = get()
                blk = self.lab[Y0:Y1, X0:X1]
                ring = cv2.dilate(sub.astype(np.uint8), ker).astype(bool) & ~sub
                vals = blk[ring]
                around = np.unique(vals); around = around[around != 0]
                # one room, not the outside, and the room is most of what is around it (a
                # real wall runs into other walls and has another space behind it)
                if len(around) == 1 and around[0] >= 2 and (vals == around[0]).mean() >= self.p.absorb_share:
                    blk[sub] = around[0]
                    done.add((t, aid)); changed = True
                    if t == "p":
                        took.append(aid)
            if not changed:
                break
        return took

    def room_mask(self, i: int) -> np.ndarray:
        return V1.room_mask(self.lab, self.root[i])

    # ---------------------------------------------------------------- score --
    def score(self) -> dict:
        if self.lab is None:
            self.segment()
        res = []
        for g in V0.load_rooms(self.floor):
            gm = V0.poly_mask(g["poly"], (self.H, self.W))
            inside = [i for i, l in enumerate(self.labels) if gm[min(int(l["xy"][1]), self.H - 1), min(int(l["xy"][0]), self.W - 1)]]
            if not inside:
                res.append({"name": g["name"], "iou": 0.0, "why": "no label"}); continue
            pm = self.room_mask(inside[0])
            res.append({"name": g["name"], "iou": round(float(np.logical_and(pm, gm).sum() / np.logical_or(pm, gm).sum()), 4)})
        v = np.array([r["iou"] for r in res])
        return {"rooms": len(res), "ge95": int((v >= .95).sum()), "ge90": int((v >= .90).sum()),
                "ge80": int((v >= .80).sum()), "mean": round(float(v.mean()), 3), "per_room": res}

    # ----------------------------------------------------- the model's view --
    def room_view(self, i: int, path: Path, max_px: int = 1100) -> dict:
        """One room for the model: original pixels, the tool's region for this room in blue,
        other rooms faint, closed openings orange, candidate (open) openings cyan, numbered."""
        m = self.room_mask(i)
        l = self.labels[i]
        if m.any():
            ys, xs = np.nonzero(m)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        else:
            x0 = x1 = int(l["xy"][0]); y0 = y1 = int(l["xy"][1])
        pad = int(1.2 * self.s); cap = int(9 * self.s)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1 = max(0, max(x0 - pad, cx - cap)), min(self.W, min(x1 + pad, cx + cap))
        y0, y1 = max(0, max(y0 - pad, cy - cap)), min(self.H, min(y1 + pad, cy + cap))
        vis = cv2.cvtColor(self.img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR).astype(np.float32)
        sub = m[y0:y1, x0:x1]
        vis[sub] = vis[sub] * 0.55 + np.array([255, 140, 40]) * 0.45
        other = (self.lab[y0:y1, x0:x1] >= 2) & ~sub
        vis[other] = vis[other] * 0.8 + np.array([180, 180, 180]) * 0.2
        vis = vis.astype(np.uint8)
        f = min(1.0, max_px / max(vis.shape[:2]))
        if f < 1:
            vis = cv2.resize(vis, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        shown = []
        for o in self.openings:
            if not (x0 <= o.p[0] < x1 and y0 <= o.p[1] < y1):
                continue
            a = (int((o.a[0] - x0) * f), int((o.a[1] - y0) * f)); b = (int((o.b[0] - x0) * f), int((o.b[1] - y0) * f))
            col = (0, 110, 255) if o.id in self.closed else (230, 200, 0)
            cv2.line(vis, a, b, col, 3)
            q = (int((o.p[0] - x0) * f) + 4, int((o.p[1] - y0) * f) - 4)
            cv2.putText(vis, str(o.id), q, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(vis, str(o.id), q, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            shown.append({"id": o.id, "closed": o.id in self.closed, "width_m": o.width_m})
        cv2.imwrite(str(path), vis)
        return {"bbox": [int(x0), int(y0), int(x1), int(y1)], "openings": shown}

    # --------------------------------------------------------------- render --
    def render(self, path: Path, bbox=None, show_openings=True, gt=False, scale_to=2400):
        x0, y0, x1, y1 = bbox or (0, 0, self.W, self.H)
        vis = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        wm = self.walls()
        vis[wm & ~self.walls0] = (0, 140, 255)              # closures: orange
        if self.lab is not None:
            rng = np.random.default_rng(1)
            for i in range(len(self.labels)):
                m = self.lab == self.root[i] + 2
                vis[m] = (vis[m] * 0.55 + rng.integers(80, 230, 3) * 0.45).astype(np.uint8)
        if gt:
            for g in V0.load_rooms(self.floor):
                if g["poly"].geom_type == "Polygon":
                    cv2.polylines(vis, [np.round(np.asarray(g["poly"].exterior.coords)).astype(np.int32)], True, (0, 0, 255), 2)
        crop = vis[y0:y1, x0:x1].copy()
        f = min(1.0, scale_to / max(crop.shape[:2])) if scale_to else 1.0
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else crop
        if show_openings:
            for o in self.openings:
                if not (x0 <= o.p[0] < x1 and y0 <= o.p[1] < y1):
                    continue
                q = (int((o.p[0] - x0) * f), int((o.p[1] - y0) * f))
                col = (0, 120, 255) if o.id in self.closed else (255, 120, 0)
                cv2.putText(crop, str(o.id), q, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.imwrite(str(path), crop)


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else "015"
    out = Path(__file__).with_name("out_araclar"); out.mkdir(exist_ok=True)
    k = Kat(f)
    print("scale", round(k.s, 1), "labels", len(k.labels), "openings", len(k.openings))
    k.closed = set()
    k.segment(); sc0 = k.score(); print("no closures :", {x: sc0[x] for x in ("ge95", "ge90", "ge80", "mean")})
    k.closed = {o.id for o in k.openings}
    k.segment(); sc1 = k.score(); print("all closed  :", {x: sc1[x] for x in ("ge95", "ge90", "ge80", "mean")})
    k.render(out / f"{f}_all_closed.png", gt=True)
    json.dump({"no_closures": sc0, "all_closed": sc1}, open(out / f"{f}_score.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
