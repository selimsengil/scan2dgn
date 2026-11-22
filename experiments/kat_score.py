"""Score a whole-floor answer (SONUC_<floor>.json, rooms in the floor image's pixel frame) against the DGN rooms.

- Balconies (type "balcony") are left out: the DGN room set has none.
- One-to-one matching of predicted polygons to DGN rooms by IoU (Hungarian); a pair counts as found at IoU >= 0.5.
- Merge errors: a prediction that covers >= 50 % of two or more DGN rooms. Split errors: a DGN room of which
  two or more predictions each cover >= 25 %.
- Per matched room: IoU and corners@10 (benchmark metric), plus the printed name/area the model wrote.
- Audit: commands in events.jsonl that touch DGN/label/benchmark files outside the test folder.
- Writes <run_dir>/SKOR.json and <run_dir>/SKOR.html (floor overlay + per-room tiles).

usage: python kat_score.py RUN_DIR [FLOOR]
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

import qwen_test as QT
import repair_rescore as RR
import v0_watershed as V0

H = type("H", (), {"inside": None})()
LEAK = re.compile(r"egitim_verisi|sam_etiket|\.dgn\b|room_boundary_benchmark|floors[\\/]+0\d\d|rooms\.json|RESULT\.json|"
                  r"spec_inputs|b132_|reference|DOWNLOAD_MANIFEST", re.I)


def poly(pts):
    p = Polygon(pts)
    return p if p.is_valid else p.buffer(0)


def tile(img, box, gt, pr, size=300):
    x0, y0, x1, y1 = box
    m = 0.15 * max(x1 - x0, y1 - y0) + 10
    bx = (max(0, int(x0 - m)), max(0, int(y0 - m)), min(img.width, int(x1 + m)), min(img.height, int(y1 + m)))
    c = img.crop(bx)
    s = size / max(c.size)
    c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))))
    d = ImageDraw.Draw(c)
    for pts, col, w in ((gt, (0, 170, 60), 3), (pr, (210, 0, 210), 2)):
        if pts and len(pts) >= 3:
            q = [((x - bx[0]) * s, (y - bx[1]) * s) for x, y in pts]
            d.line(q + q[:1], fill=col, width=w)
    b = io.BytesIO(); c.save(b, "JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()


def audit(run):
    hits = []
    ev = run / "events.jsonl"
    if not ev.exists():
        return ["events.jsonl yok"]
    for line in ev.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        it = e.get("item") or {}
        cmd = it.get("command") or ""
        if e.get("type") == "item.started" and cmd and LEAK.search(cmd):
            hits.append(cmd[:300])
    return hits


def main():
    run = Path(sys.argv[1])
    floor = sys.argv[2] if len(sys.argv) > 2 else "015"
    ans = json.load(open(run / f"SONUC_{floor}.json", encoding="utf-8-sig"))
    gts = [g for g in V0.load_rooms(floor) if g["poly"].geom_type == "Polygon"]
    G = [g["poly"] for g in gts]
    preds = [r for r in ans.get("rooms", []) if r.get("type") != "balcony" and r.get("boundary") and len(r["boundary"]) >= 3]
    P = []
    for r in preds:
        pts = r["boundary"]
        if not Polygon(pts).is_valid:
            pts = RR.repair(pts)
        r["_pts"] = pts
        P.append(poly(pts))
    iou = np.zeros((len(G), len(P)))
    for i, g in enumerate(G):
        for j, p in enumerate(P):
            if g.intersects(p):
                inter = g.intersection(p).area
                iou[i, j] = inter / g.union(p).area
    gi, pj = linear_sum_assignment(-iou) if len(P) else ([], [])
    match = {int(i): int(j) for i, j in zip(gi, pj) if iou[i, j] >= 0.5}
    rows = []
    for i, g in enumerate(gts):
        ref = list(g["poly"].exterior.coords)[:-1]
        row = {"dgn": i, "dgn_name": g["name"], "dgn_area": g["area_m2"], "IoU": 0.0, "corners@10": 0.0, "pred": None}
        if i in match:
            j = match[i]
            m = QT.brief(QT.score(preds[j]["_pts"], ref, H))
            row.update({"IoU": round(float(iou[i, j]), 3), "corners@10": m.get("corners@10", 0.0), "pred": j,
                        "pred_name": preds[j].get("name"), "pred_area": preds[j].get("printed_area_m2"),
                        "pred_status": preds[j].get("boundary_status")})
        rows.append(row)
    merges, splits = [], []
    for j, p in enumerate(P):
        cov = [i for i, g in enumerate(G) if g.intersects(p) and g.intersection(p).area / g.area >= 0.5]
        if len(cov) >= 2:
            merges.append({"pred": j, "name": preds[j].get("name"), "covers": [gts[i]["name"] for i in cov]})
    for i, g in enumerate(G):
        by = [j for j, p in enumerate(P) if g.intersects(p) and g.intersection(p).area / g.area >= 0.25]
        if len(by) >= 2:
            splits.append({"dgn": i, "name": gts[i]["name"], "preds": [preds[j].get("name") for j in by]})
    found = [r for r in rows if r["pred"] is not None]
    summary = {
        "dgn_rooms": len(gts), "predicted_polygons": len(P), "balconies_given": sum(r.get("type") == "balcony" for r in ans.get("rooms", [])),
        "found_IoU>=0.5": len(found), "IoU>=0.90": sum(r["IoU"] >= .9 for r in rows), "IoU>=0.95": sum(r["IoU"] >= .95 for r in rows),
        "mean_IoU_all_dgn": round(float(np.mean([r["IoU"] for r in rows])), 3),
        "mean_IoU_found": round(float(np.mean([r["IoU"] for r in found])), 3) if found else 0,
        "mean_corners_found": round(float(np.mean([r["corners@10"] for r in found])), 3) if found else 0,
        "extra_predictions": len(P) - len(found), "merges": merges, "splits": splits,
    }
    hits = audit(run)
    summary["audit_suspicious_commands"] = len(hits)
    json.dump({"summary": summary, "rooms": rows, "audit": hits}, open(run / "SKOR.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    img = Image.open(run / f"kat_{floor}.png").convert("RGB")
    ov = img.copy(); d = ImageDraw.Draw(ov)
    for g in G:
        d.line(list(g.exterior.coords), fill=(0, 170, 60), width=4)
    for r in preds:
        q = [tuple(p) for p in r["_pts"]]
        d.line(q + q[:1], fill=(210, 0, 210), width=3)
    ov.thumbnail((1500, 1500))
    b = io.BytesIO(); ov.save(b, "JPEG", quality=82); big = base64.b64encode(b.getvalue()).decode()
    css = ("body{margin:0;background:#f6f6f4;color:#1d1d1b;font:14px/1.45 system-ui,Segoe UI,sans-serif}"
           "@media (prefers-color-scheme:dark){body{background:#161615;color:#ececE8}}"
           "main{max-width:1500px;margin:0 auto;padding:18px 16px} img{max-width:100%;height:auto;border-radius:4px;background:#fff}"
           ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px} figure{margin:0}"
           "figcaption{font-size:12.5px} pre{white-space:pre-wrap;font-size:12.5px}")
    out = [f"<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Kat {floor} Skoru</title><style>{css}</style></head><body><main><h1>Kat {floor} — {run.name}</h1>"
           "<p>Yeşil: DGN · Mor: modelin çizimi</p>", f"<pre>{json.dumps(summary, ensure_ascii=False, indent=1)}</pre>",
           f"<img alt='kat' src='data:image/jpeg;base64,{big}'><h2>Odalar</h2><div class='g'>"]
    for r, g in zip(rows, gts):
        pr = preds[r["pred"]]["_pts"] if r["pred"] is not None else None
        out.append(f"<figure><img alt='{r['dgn_name']}' src='data:image/jpeg;base64,{tile(img, g['poly'].bounds, list(g['poly'].exterior.coords)[:-1], pr)}'>"
                   f"<figcaption>{r['dgn']} {r['dgn_name']} {r['dgn_area'] or ''} → IoU {r['IoU']:.2f} · köşe {r['corners@10']:.2f}"
                   f"{' · ' + str(r.get('pred_name')) if r['pred'] is not None else ' · BULUNAMADI'}</figcaption></figure>")
    out.append("</div></main></body></html>")
    (run / "SKOR.html").write_text("".join(out), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
