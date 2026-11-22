"""Two-model floor drawing: a reasoning model's draft decides WHICH corners a room has; NorthCUA-Mini (trained to
click pixel-exact) puts each of them on the exact inner wall corner.

For every vertex of every drafted room: a window of the floor around the vertex (enlarged to 1000 px) is shown
with the draft edges in blue and the vertex as a red cross; Mini clicks the exact corner. A click farther than
--max-move floor px from the draft vertex is ignored (the draft vertex is kept).

usage: python mini_refine.py RUN_DIR [--floor 015] [--win 70] [--max-move 30]
       -> RUN_DIR + "_mini" with the refined SONUC_<floor>.json, then kat_score.py on it
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import json
import math
import shutil
import time
from pathlib import Path

from PIL import Image, ImageDraw

import mini_trace as MT

ASK = ("The blue lines are a DRAFT of one room's boundary and the red cross marks one of its corners. The draft is "
       "slightly off. Click the EXACT point of that corner on the plan: where the two INNER wall faces (the wall lines "
       "facing into the room) meet, or the jamb end where the boundary closes an opening. Click once.")


ASK_NOMARK = ("This crop is centred on one corner of a room's boundary. The room's interior lies toward the {dir} of "
              "that corner. Click the EXACT corner point nearest to the centre of the image: where the room's INNER "
              "wall faces (the wall lines facing into the room) meet, or the jamb end where the boundary closes an "
              "opening. Click once.")


def interior_dir(pts, i):
    """Compass word for the direction from vertex i into the polygon (bisector of its two edges, flipped inward)."""
    from shapely.geometry import Point, Polygon
    P = Polygon(pts).buffer(0)
    x, y = pts[i]
    a, b = pts[i - 1], pts[(i + 1) % len(pts)]
    v1 = [a[0] - x, a[1] - y]; v2 = [b[0] - x, b[1] - y]
    n1, n2 = math.hypot(*v1) or 1, math.hypot(*v2) or 1
    bx, by = v1[0] / n1 + v2[0] / n2, v1[1] / n1 + v2[1] / n2
    if math.hypot(bx, by) < 1e-6:
        bx, by = -v1[1] / n1, v1[0] / n1
    if not P.contains(Point(x + 6 * bx / math.hypot(bx, by), y + 6 * by / math.hypot(bx, by))):
        bx, by = -bx, -by
    ang = math.degrees(math.atan2(by, bx)) % 360   # 0 = right, 90 = down (image y)
    names = ["right", "bottom-right", "bottom", "bottom-left", "left", "top-left", "top", "top-right"]
    return names[int(((ang + 22.5) % 360) // 45)]


def refine_vertex_nomark(url, floor, pts, i, win):
    x, y = pts[i]
    x0, y0 = int(x - win), int(y - win)
    x1, y1 = int(x + win), int(y + win)
    reg = floor.crop((x0, y0, x1, y1))   # PIL pads outside the image with black; keeps the vertex centred
    s = 1000 / max(reg.size)
    im = reg.resize((int(reg.width * s), int(reg.height * s)), Image.Resampling.LANCZOS).convert("RGB")
    msgs = [{"role": "system", "content": MT.SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": MT.b64(im)}},
                                         {"type": "text", "text": ASK_NOMARK.format(dir=interior_dir(pts, i))}]}]
    c = MT.parse_click(MT.call(url, msgs)["choices"][0]["message"])
    if not isinstance(c, tuple):
        return None
    return [x0 + c[0] / 1000 * im.width / s, y0 + c[1] / 1000 * im.height / s]


def refine_vertex(url, floor, pts, i, win):
    x, y = pts[i]
    x0, y0 = max(0, int(x - win)), max(0, int(y - win))
    x1, y1 = min(floor.width, int(x + win)), min(floor.height, int(y + win))
    reg = floor.crop((x0, y0, x1, y1))
    s = 1000 / max(reg.size)
    im = reg.resize((int(reg.width * s), int(reg.height * s)), Image.Resampling.LANCZOS).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    n = len(pts)
    for j in range(n):
        a, b = pts[j], pts[(j + 1) % n]
        d.line([((a[0] - x0) * s, (a[1] - y0) * s), ((b[0] - x0) * s, (b[1] - y0) * s)], fill=(0, 80, 255, 150), width=3)
    cx, cy = (x - x0) * s, (y - y0) * s
    d.line([(cx - 14, cy), (cx + 14, cy)], fill=(230, 0, 0), width=3)
    d.line([(cx, cy - 14), (cx, cy + 14)], fill=(230, 0, 0), width=3)
    msgs = [{"role": "system", "content": MT.SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": MT.b64(im)}}, {"type": "text", "text": ASK}]}]
    c = MT.parse_click(MT.call(url, msgs)["choices"][0]["message"])
    if not isinstance(c, tuple):
        return None
    return [x0 + c[0] / 1000 * im.width / s, y0 + c[1] / 1000 * im.height / s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--floor", default="015")
    ap.add_argument("--url", default="http://127.0.0.1:8092/v1/chat/completions")
    ap.add_argument("--win", type=int, default=70, help="half size of the window around a vertex, floor px")
    ap.add_argument("--max-move", dest="max_move", type=float, default=30)
    ap.add_argument("--nomark", action="store_true", help="no drawing on the crop; vertex at the centre, direction in words")
    a = ap.parse_args()
    src = Path(a.run)
    out = src.parent / (src.name + ("_mini_nomark" if a.nomark else "_mini"))
    out.mkdir(exist_ok=True)
    shutil.copy(src / f"kat_{a.floor}.png", out / f"kat_{a.floor}.png")
    floor = Image.open(src / f"kat_{a.floor}.png").convert("RGB")
    res = json.load(open(src / f"SONUC_{a.floor}.json", encoding="utf-8"))
    t0, moved, kept = time.time(), 0, 0
    for r in res["rooms"]:
        if r.get("type") == "balcony" or not r.get("boundary"):
            continue
        pts = [list(map(float, p)) for p in r["boundary"]]
        new = []
        for i in range(len(pts)):
            c = (refine_vertex_nomark if a.nomark else refine_vertex)(a.url, floor, pts, i, a.win)
            if c and math.hypot(c[0] - pts[i][0], c[1] - pts[i][1]) <= a.max_move:
                new.append([round(c[0], 1), round(c[1], 1)]); moved += 1
            else:
                new.append(pts[i]); kept += 1
        r["boundary_draft"] = r["boundary"]
        r["boundary"] = new
        print(f"{r.get('name')}: {len(pts)} corners, {time.time() - t0:.0f} s", flush=True)
    (out / f"SONUC_{a.floor}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"moved {moved}, kept {kept}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
