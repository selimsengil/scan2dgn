"""Turns the DGN snapshots (dgn_cikar.py) into labelled floors: for every floor plan frame of every project,
the scan cut out of the sheet TIF and the user's own drawing in that cut's pixels:
rooms (OdaAlani, with name and m2), door openings (KapiAcikligi), windows (PencereAcikligi), balconies (BalkonAlani),
common areas (OrtakAlan).  Geometry is only moved by the raster placement stored in the DGN; nothing is redrawn.
Output: kat_test/dgn_katlar/<floor id>/clean.png + LABELS_COMMON.json (same fields the other scripts read) + KONTROL.jpg
The blind floor (Kisla 83, floor 1 = old 015) is not written unless --blind.
usage: python dgn_kat.py [--only Marmara] [--blind]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, math, re, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, mapping

Image.MAX_IMAGE_PIXELS = None
SRC = config.KT / "dgn"; OUT = config.KT / "dgn_katlar"
LEVELS = {"OdaAlani": "room", "KapiAcikligi": "door", "PencereAcikligi": "window", "BalkonAlani": "balcony", "OrtakAlan": "common"}
COL = {"room": (0, 150, 60), "door": (225, 30, 60), "window": (0, 170, 200), "balcony": (150, 60, 200), "common": (230, 140, 0)}


class RasterMap:
    def __init__(s, d): s.d = d

    def pixels(s, x, y):
        d = s.d; a = d["rotation"]; b = a + d["affinity"]; det = math.cos(a) * math.cos(b) + math.sin(a) * math.sin(b)
        dx, dy = x - d["origin"][0], y - d["origin"][1]; u = (math.cos(b) * dx + math.sin(b) * dy) / det; v = (-math.sin(a) * dx + math.cos(a) * dy) / det
        return [u / d["extent"][0] * d["pixels"][0], (1 - v / d["extent"][1]) * d["pixels"][1]]


def short(name): return re.sub(r"[^A-Za-z0-9]+", "_", name.replace("Balikesir_", "").split("(")[0].replace("%29", "").replace("28avsa", "")).strip("_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--blind", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); index = []; seen = []
    for d in sorted(p for p in SRC.iterdir() if (p / "native_snapshot.json").exists()):
        if a.only and a.only.lower() not in d.name.lower(): continue
        S = json.load(open(d / "native_snapshot.json", encoding="utf-8")); E = S["elements"]; assert S["unit"] == "m", S["unit"]
        tags = {}
        for e in E:
            if "tag" in e: tags.setdefault(e["base"], {})[e["tag"]] = e["value"]
        frames = [e for e in E if e["level"] == "KatPlaniSiniri" and e.get("points")]; floors = [e for e in E if e["level"] == "KatAlani" and e.get("points")]
        tifs = {}
        for fr in frames:
            FP = Polygon([p[:2] for p in fr["points"]]); z = fr["points"][0][2]
            fl = [f for f in floors if abs(f["points"][0][2] - z) < 1e-3 and FP.covers(Polygon([p[:2] for p in f["points"]]).representative_point())]
            katno = str(tags.get(fl[0]["id"], {}).get("Kat No", "X")) if fl else "X"; fid = f"{short(d.name)}__{re.sub(r'[^A-Za-z0-9]+', '', katno) or 'X'}_{fr['id']}"
            blind = "Kisla_143_83" in d.name and katno == "1"
            if blind and not a.blind: index.append({"floor": fid, "blind": True}); print(fid, "BLIND floor, not written"); continue
            cands = []
            for r in S["rasters"]:
                m = RasterMap(r); px = [m.pixels(*p[:2]) for p in fr["points"]]; w, h = r["pixels"]
                if all(-2 <= p[0] <= w + 2 and -2 <= p[1] <= h + 2 for p in px): cands.append((r, m, px))
            if len(cands) != 1: index.append({"floor": fid, "error": f"{len(cands)} raster placements contain the frame"}); print(fid, "SKIPPED:", len(cands), "raster placements"); continue
            r, m, px = cands[0]; tif = r["tif"]
            if tif not in tifs: tifs[tif] = Image.open(tif)
            im = tifs[tif]; assert list(im.size) == [int(v) for v in r["pixels"]], "TIF size differs from the registration"
            box = [max(0, math.floor(min(p[0] for p in px)) - 20), max(0, math.floor(min(p[1] for p in px)) - 20), min(im.width, math.ceil(max(p[0] for p in px)) + 20), min(im.height, math.ceil(max(p[1] for p in px)) + 20)]
            # an upper floor drawn again over the same plan of the sheet (2nd floor = copy of the 1st) adds no new picture
            same = None
            for t2, b2, f2 in seen:
                if t2 != tif: continue
                ix = max(0, min(box[2], b2[2]) - max(box[0], b2[0])); iy = max(0, min(box[3], b2[3]) - max(box[1], b2[1])); u = (box[2] - box[0]) * (box[3] - box[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - ix * iy
                if ix * iy / max(u, 1) > 0.8: same = f2
            if same: index.append({"floor": fid, "kat_no": katno, "same_picture_as": same}); print(fid, "same plan picture as", same, "- not written again"); continue
            seen.append((tif, box, fid))
            crop = im.crop(box).convert("L"); out = OUT / fid; out.mkdir(exist_ok=True); crop.save(out / "clean.png")
            rooms, structs, cnt = [], [], {}
            for e in E:
                kind = LEVELS.get(e["level"])
                if not kind or not e.get("points") or len(e["points"]) < 3 or abs(e["points"][0][2] - z) > 1e-3: continue
                P = Polygon([p[:2] for p in e["points"]])
                if not P.is_valid: P = P.buffer(0)
                if P.is_empty or not FP.covers(P.representative_point()): continue
                pts = [[q[0] - box[0], q[1] - box[1]] for q in (m.pixels(*p[:2]) for p in e["points"])]
                Q = Polygon(pts); Q = Q if Q.is_valid else Q.buffer(0)
                if Q.is_empty or Q.geom_type != "Polygon": continue
                t = tags.get(e["id"], {}); cnt[kind] = cnt.get(kind, 0) + 1; rp = Q.representative_point()
                if kind == "room":
                    m2 = t.get("Oda M2"); rooms.append({"id": f"{fid}:dgn:{e['id']}", "kind": "room", "name": str(t.get("Oda Adı", "")), "area_text_value": m2 if isinstance(m2, (int, float)) and m2 > 0 else None,
                                                         "interior_polygon": mapping(Q), "point": {"xy": [rp.x, rp.y], "origin": "inside point of the polygon, not the printed label"}, "boundary_origin": "user_drawn_DGN_vertices_only"})
                else: structs.append({"id": f"{fid}:dgn:{e['id']}", "kind": kind, "geometry": mapping(Q), "origin": "user_drawn_DGN", "tags": {k: v for k, v in t.items()}})
            ar = [(Polygon(rm["interior_polygon"]["coordinates"][0]).area / rm["area_text_value"]) ** 0.5 for rm in rooms if rm["area_text_value"]]
            scale_reg = r["pixels"][0] / r["extent"][0]
            L = {"schema": "dgn-floor-v1", "floor_id": fid, "project": d.name, "kat_no": katno, "image": {"width": crop.width, "height": crop.height}, "px_per_m": scale_reg, "px_per_m_from_room_areas": float(np.median(ar)) if ar else None,
                 "source": {"dgn": S["source_dgn"], "dgn_sha256": S["source_sha256"], "tif": tif, "crop_tif_xyxy": box, "raster_rotation": r["rotation"], "frame_element": fr["id"]}, "rooms": rooms, "structures": structs}
            json.dump(L, open(out / "LABELS_COMMON.json", "w", encoding="utf-8"), ensure_ascii=False)
            s = min(1.0, 2200 / max(crop.size)); pv = crop.convert("RGB").resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS); dr = ImageDraw.Draw(pv)
            for obj in structs + rooms:
                g = obj.get("interior_polygon") or obj["geometry"]; xy = [(x * s, y * s) for x, y in g["coordinates"][0]]; dr.line(xy, fill=COL[obj["kind"]], width=2)
            pv.save(out / "KONTROL.jpg", quality=80)
            index.append({"floor": fid, "project": d.name, "kat_no": katno, "size": list(crop.size), "px_per_m": round(scale_reg, 1), "px_per_m_areas": round(float(np.median(ar)), 1) if ar else None, **cnt})
            print(index[-1], flush=True)
    json.dump(index, open(OUT / "INDEX.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
