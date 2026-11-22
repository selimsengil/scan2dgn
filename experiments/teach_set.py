"""The teaching set for the drawing model: three legend sheets built from real plans and the labels.

  SHEET_A  what the boundary FOLLOWS: columns, the wall's inner face, door and window openings
  SHEET_B  what is NOT boundary: dimension lines, axis lines, stamps/signatures, sanitary fixtures,
           counters, shafts, furniture, text
  SHEET_C  solved rooms: the CAD boundary in green with the teacher's notes
Sources: the Tekke floor (DGN labels: rooms, doors, windows; none of the test rooms) and the crop of
013_04 (floor 013: not a floor of the test rooms 015_02, 019_18, 001_17).
Writes teach/SHEET_*.png and teach/TEACH.json (the captions, the text that goes with each sheet).

usage: python teach_set.py
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import bench132 as B

sys.path.insert(0, r"C:\Users\Selim\Documents\cizim_projeler\asistan_cizim\20261107_tekke_dort_oda")
import run_pilot as RP  # noqa: E402

OUT = Path(__file__).with_name("teach")
TILE_W, TILE_H, CAP_H, COLS = 420, 300, 92, 3
GREEN, ORANGE, BLUE, RED = (0, 165, 60), (255, 120, 0), (0, 110, 255), (220, 0, 0)

try:
    F_CAP = ImageFont.truetype("arial.ttf", 15)
    F_HEAD = ImageFont.truetype("arialbd.ttf", 17)
    F_TITLE = ImageFont.truetype("arialbd.ttf", 26)
except OSError:
    F_CAP = F_HEAD = F_TITLE = ImageFont.load_default()


def wrap(text, font, width, d):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def tile(src: Image.Image, box, head: str, cap: str, overlays=(), big=False):
    """box in src pixels; overlays: (points_in_src, colour, width, closed). big: a solved room's larger tile."""
    global TILE_W, TILE_H
    keep = (TILE_W, TILE_H)
    if big:
        TILE_W, TILE_H = 640, 470
    try:
        return _tile(src, box, head, cap, overlays)
    finally:
        TILE_W, TILE_H = keep


def _tile(src: Image.Image, box, head: str, cap: str, overlays=()):
    x0, y0, x1, y1 = [int(v) for v in box]
    reg = src.crop((x0, y0, x1, y1)).convert("RGB")
    s = min(TILE_W / reg.width, TILE_H / reg.height, 4.0)
    reg = reg.resize((max(1, int(reg.width * s)), max(1, int(reg.height * s))), Image.Resampling.LANCZOS)
    t = Image.new("RGB", (TILE_W, TILE_H + CAP_H), (255, 255, 255))
    ox, oy = (TILE_W - reg.width) // 2, (TILE_H - reg.height) // 2
    pic = Image.new("RGB", (TILE_W, TILE_H), (255, 255, 255))   # overlays drawn here: clipped to the picture
    pic.paste(reg, (ox, oy))
    dp = ImageDraw.Draw(pic)
    for pts, col, w, closed in overlays:
        q = [(ox + (x - x0) * s, oy + (y - y0) * s) for x, y in pts]
        dp.line(q + (q[:1] if closed else []), fill=col, width=w)
    t.paste(pic, (0, 0))
    d = ImageDraw.Draw(t)
    d.rectangle((0, 0, TILE_W - 1, TILE_H - 1), outline=(150, 150, 150))
    d.text((6, TILE_H + 4), head, fill=(0, 0, 0), font=F_HEAD)
    y = TILE_H + 26
    for line in wrap(cap, F_CAP, TILE_W - 10, d)[:4]:
        d.text((6, y), line, fill=(40, 40, 40), font=F_CAP); y += 16
    return t


def sheet(title: str, tiles, path: Path, cols: int = COLS):
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    W, H = cols * tw + (cols + 1) * 10, 46 + rows * (th + 10)
    S = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(S)
    d.text((12, 8), title, fill=(0, 0, 0), font=F_TITLE)
    for i, t in enumerate(tiles):
        S.paste(t, (10 + (i % cols) * (tw + 10), 46 + (i // cols) * (th + 10)))
    S.save(path)
    return S.size


def closed(p):
    return p[:-1] if len(p) > 1 and p[0] == p[-1] else p


def main():
    OUT.mkdir(exist_ok=True)
    tek = RP.source().convert("RGB")
    lab = json.loads((RP.SOURCE_DIR / "DGN_TARGETS.json").read_text(encoding="utf8"))["elements"]
    by = {e["id"]: e for e in lab}
    salon = closed(by[54916]["points"])                       # SALON 20.00 (teaching example 03)
    door, window = closed(by[55366]["points"]), closed(by[55475]["points"])
    c13 = next(c for k, f, n, c, r, l in B.rooms(1400) if k == "013_04")
    room = lambda: (salon, GREEN, 3, True)

    a = [
        tile(c13, (500, 110, 620, 230), "KOLON (column)",
             "A small rectangle filled with diagonal hatching, attached to the wall. Where it sticks into a room, the "
             "room boundary goes AROUND it (a notch)."),
        tile(c13, (500, 620, 620, 740), "KOLON (column)",
             "Same symbol in the middle of a long wall: the boundary steps in and out around the hatched block."),
        tile(c13, (110, 1160, 230, 1280), "KOLON (column)",
             "A hatched block at a wall end next to a door. Hatching = solid concrete, never room area."),
        tile(tek, (800, 1760, 1120, 1990), "DUVAR IC YUZU (inner face)",
             "A wall is two parallel thick lines. GREEN = the CAD boundary: it runs on the line facing INTO the room, "
             "not on the outer line or the middle.", [room()]),
        tile(tek, (860, 1740, 1140, 1900), "KAPI ACIKLIGI (door opening)",
             "ORANGE box = the door opening between its two jambs. The leaf and the swing arc are not boundary: the "
             "green line closes the opening STRAIGHT, continuing the inner face.", [room(), (door, ORANGE, 2, True)]),
        tile(c13, (280, 700, 440, 920), "KAPI (door) + label",
             "Door leaf (thin line) and quarter-circle swing. 'K2-90/210' is the door tag (width/height cm), not a line "
             "to follow."),
        tile(tek, (520, 2440, 920, 2660), "PENCERE ACIKLIGI (window)",
             "BLUE box = window: thin parallel lines set inside the outer wall. The room boundary passes the window "
             "straight along the wall's inner face.", [room(), (window, BLUE, 2, True)]),
        tile(c13, (270, 20, 400, 130), "BALKON KAPISI (balcony door)",
             "Door to the balcony, tag 'K2-90/210'. The balcony is a separate room: the boundary closes this opening "
             "straight too."),
        tile(c13, (520, 1150, 640, 1300), "DUVAR DONUSU (wall return)",
             "Where walls meet or end, follow the inner face exactly around the corner; small offsets are real."),
    ]
    c01 = next(c for k, f, n, c, r, l in B.rooms(1400) if k == "001_01")   # its left part only (not the bathroom side)
    b = [
        tile(c13, (290, 762, 570, 818), "OLCU CIZGISI (dimension line)",
             "Thin line with short ticks at its ends and numbers on it (cm). It crosses rooms and walls: NEVER boundary."),
        tile(c13, (220, 230, 520, 500), "MUHUR + IMZA (stamp, signature)",
             "Round municipal stamp and hand signature drawn over the plan. Ignore completely; lines under it continue."),
        tile(c13, (240, 570, 410, 870), "VITRIFIYE (sanitary fixtures)",
             "Shower tray, washbasin, WC pan. Fixtures stand inside the room; the boundary is the wall behind them."),
        tile(c13, (90, 190, 230, 440), "TEZGAH / LAVABO (counter)",
             "Long rectangles with basins = counter or vanity. Furniture, not boundary."),
        tile(c13, (55, 530, 245, 880), "SAFT / BOSLUK (shaft)",
             "A box crossed with an X = shaft or void. It is not part of the neighbouring room."),
        tile(c01, (60, 640, 900, 760), "AKS CIZGISI (axis line)",
             "Dash-dot or very long thin lines running through rooms and walls (structural axes). Not boundary."),
        tile(c01, (520, 770, 1240, 1080), "MOBILYA: YATAK, DOLAP (furniture)",
             "Bed with pillow, wardrobe with hangers: furniture inside the room. The boundary is the wall line behind them."),
        tile(c01, (220, 180, 700, 470), "MOBILYA: MASA, KOLTUK (furniture)",
             "Desk and armchair. Closed shapes inside a room are furniture, not walls: walls are long double lines."),
        tile(c13, (170, 1030, 400, 1150), "YAZI / ETIKET (text)",
             "Room name and area ('yatod.nisi', 'ul:7.60 m2'). It names the room; it is not a line to follow."),
    ]
    teach = json.loads((RP.SOURCE_DIR / "OGRETIM_ORNEKLERI.json").read_text(encoding="utf8"))["examples"]
    items = {r["id"]: r for r in lab}
    c = []
    for num in ("06", "03", "05"):
        ex = next(e for e in teach if e["number"] == num)
        x0, y0, x1, y1 = ex["crop_box"]
        pts = closed(items[ex["room_id"]]["points"])
        c.append(tile(tek, (x0, y0, x1, y1), f"SOLVED: {ex['title']}", "GREEN = correct boundary. " + ex["focus"],
                      [(pts, GREEN, 3, True)], big=True))
    sizes = {"A": sheet("A. What the room boundary FOLLOWS", a, OUT / "SHEET_A.png"),
             "B": sheet("B. What is NOT boundary (ignore)", b, OUT / "SHEET_B.png"),
             "C": sheet("C. Solved rooms (green = CAD boundary)", c, OUT / "SHEET_C.png", cols=2)}
    json.dump({"sizes": sizes}, open(OUT / "TEACH.json", "w", encoding="utf-8"), indent=1)
    print(sizes)


if __name__ == "__main__":
    main()
