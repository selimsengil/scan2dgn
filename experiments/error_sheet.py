"""Contact sheet of the rooms with a given error kind (from ERRORS.json): DGN green, model magenta, red dot = label.

usage: python error_sheet.py TAG KIND [N]   ->  runs/b132_<TAG>/err_<KIND>.png
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

R = Path(__file__).with_name("runs")
T = 400


def main():
    tag, kind = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    d = R / f"b132_{tag}"
    res = json.load(open(d / "RESULT.json", encoding="utf-8"))
    err = json.load(open(d / "ERRORS.json", encoding="utf-8"))
    keys = sorted((k for k, v in err.items() if v["error"] == kind), key=lambda k: err[k].get("IoU", 0))[:n]
    cols = 5
    sheet = Image.new("RGB", (cols * T, ((len(keys) + cols - 1) // cols) * (T + 22)), "white")
    for i, k in enumerate(keys):
        im = Image.open(d / f"{k}.png").convert("RGB")
        s = T / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
        dr = ImageDraw.Draw(im)
        for pts, col, w in ((res[k]["reference"], (0, 170, 60), 4), (res[k].get("snapped") or res[k].get("points"), (210, 0, 210), 3)):
            if pts and len(pts) >= 3:
                q = [(x * s, y * s) for x, y in pts]
                dr.line(q + q[:1], fill=col, width=w)
        x, y = (i % cols) * T, (i // cols) * (T + 22)
        sheet.paste(im, (x, y + 22))
        ImageDraw.Draw(sheet).text((x + 4, y + 4), f"{k} {res[k]['name'][:24]} IoU {err[k].get('IoU', 0):.2f}", fill="black")
    out = d / f"err_{kind}.png"
    sheet.save(out)
    print(out, len(keys))


if __name__ == "__main__":
    main()
