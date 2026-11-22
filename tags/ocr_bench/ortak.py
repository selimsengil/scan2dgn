"""OCR test on the hand-labelled Malkoc K2 (ETIKET.json): ground truth = the tag text of each of the 40 doors/windows.
A: spotting on the whole sheet -> list of {box (4 pts, sheet px), text}; a tag is found when a read's centre lies within
   30 cm of the opening and its digits match.   B: reading level crops of each opening (0 and 180 deg) -> text per crop.
Match levels: 'tam' = whole tag ("P1 150/140", "90x220"), 'sayi' = the numbers ("150/140", "90x220")."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, re
from pathlib import Path
import cv2, numpy as np
from shapely.geometry import Point, Polygon
H = config.KT / "ocr_test"; ET = config.KT / "el_etiket" / "malkoc_k2" / "ETIKET.json"
FID = "Gonen_Malkoc_361_79__K2_56155"; IMG = H.parent / "dgn_katlar" / FID / "clean.png"
def gt(): return json.load(open(ET, encoding="utf-8"))
def sheet(): return cv2.imdecode(np.fromfile(str(IMG), np.uint8), 0)
def norm(t):
    t = t.upper().replace("×", "X").replace("*", "X").replace("\\", "/").replace("|", "1").replace("O", "0")
    return re.sub(r"[^0-9XP/]", "", t)
def nums(t): return re.sub(r"[^0-9X/]", "", norm(t)).lstrip("X/").replace("P", "")
def match(read, tag):
    r, g = norm(read), norm(tag); rn, gn = nums(read), nums(tag)
    if g in r: return "tam"
    if gn and gn in rn: return "sayi"
    gd, rd = re.sub(r"\D", "", gn), re.sub(r"\D", "", norm(read))
    if len(gd) >= 4 and gd in rd: return "rakam"          # digits right, '/' or 'x' dropped (format is fixed, parse later)
    return None
ORDER = {"tam": 0, "sayi": 1, "rakam": 2}
def score_spot(reads, name):
    E = gt(); s = E["px_per_m"]; rows = []; used = set()
    for o in E["openings"]:
        B = Polygon(o["poly"]).buffer(0.30 * s); best = None
        for i, r in enumerate(reads):
            c = np.mean(np.asarray(r["box"], float), 0)
            if not B.contains(Point(*c)): continue
            m = match(r["text"], o["etiket"])
            if m and (best is None or ORDER[m] < ORDER[best[1]]): best = (i, m)
        if best: used.add(best[0])
        rows.append({"id": o["id"], "kind": o["kind"], "etiket": o["etiket"], "sonuc": best[1] if best else None,
                     "okunan": reads[best[0]]["text"] if best else None})
    return summarize(rows, name, extra={"okuma_sayisi": len(reads)})
def summarize(rows, name, extra=None):
    out = {"model": name, **(extra or {})}
    for k in ("door", "window"):
        R = [r for r in rows if r["kind"] == k]
        out[k] = {"n": len(R), "tam": sum(r["sonuc"] == "tam" for r in R), "sayi_dahil": sum(r["sonuc"] in ("tam", "sayi") for r in R), "rakam_dahil": sum(r["sonuc"] is not None for r in R)}
    out["satirlar"] = rows; return out
def crops(margin_m=0.15, scale=1.0):
    """Level crops of each opening: list of (id, kind, tag, [img0, img180])."""
    E = gt(); s = E["px_per_m"]; g = sheet(); out = []
    for o in E["openings"]:
        c = np.asarray(o["poly"], float); e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e))
        d = c[(k + 1) % 4] - c[k]; ang = np.degrees(np.arctan2(d[1], d[0])); m = c.mean(0)
        L = (max(e) / 2 + margin_m * s) * scale; Hh = (min(e) / 2 + margin_m * s) * scale
        M = cv2.getRotationMatrix2D(tuple(m), ang, scale); M[:, 2] += [L - m[0], Hh - m[1]]
        v = cv2.warpAffine(g, M, (int(2 * L), int(2 * Hh)), borderValue=255)
        out.append((o["id"], o["kind"], o["etiket"], [v, cv2.rotate(v, cv2.ROTATE_180)]))
    return out
def score_crops(res, name, extra=None):
    """res: {id: [text0, text180]}"""
    E = gt(); rows = []
    for o in E["openings"]:
        ts = res.get(o["id"], []); ms = [match(t, o["etiket"]) for t in ts]
        m = "tam" if "tam" in ms else ("sayi" if "sayi" in ms else ("rakam" if "rakam" in ms else None))
        rows.append({"id": o["id"], "kind": o["kind"], "etiket": o["etiket"], "sonuc": m, "okunan": ts})
    return summarize(rows, name, extra)
