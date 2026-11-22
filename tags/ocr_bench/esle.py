"""Tag finding without relying on OCR alone: copies of the same tag / symbol are found by image matching (word spotting).
Chain per sheet:
  1. seeds: number-pair reads of cheap OCRs (SONUC2: rapidocr_4yon, doctr, florence2) + opening-net (a1) boxes (whole symbol)
  2. each seed -> template (level crop) -> normalized cross-correlation over the sheet, 4 orientations, scale 0.5
  3. instances grouped by seed; seeds whose instances coincide are merged
  4. numbers of a group: vote over the seeds' readings + GLM-OCR readings of its best copies (for symbol groups only pairs whose
     first number fits the symbol width count); weak copies (< --hi) are kept only if GLM reads the same numbers there
     (text groups) or they are not on top of another group (symbol groups)
Output: reads [{box, text}] scored by test2.   usage: python esle.py KEY [--hi 0.6] [--lo 0.45]   (.venv-spec, 5070 Ti)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, os, re, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from collections import Counter
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from shapely.geometry import Polygon
H = config.KT / "ocr_test"
import test2

ap = argparse.ArgumentParser(); ap.add_argument("key"); ap.add_argument("--hi", type=float, default=0.6); ap.add_argument("--lo", type=float, default=0.55); ap.add_argument("--maxkopya", type=int, default=80)
ap.add_argument("--sc", type=float, default=0.5); ap.add_argument("--simge_lo", type=float, default=0.6); ap.add_argument("--tag", default="esle"); a = ap.parse_args()
fid = test2.FLOORS[a.key]; t0 = time.time()
L = json.load(open(test2.K / "dgn_katlar" / fid / "LABELS_COMMON.json", encoding="utf-8")); s = L.get("px_per_m_from_room_areas") or L["px_per_m"]
g = cv2.imdecode(np.fromfile(str(test2.K / "dgn_katlar" / fid / "clean.png"), np.uint8), 0)


def frame(box):
    c = np.asarray(box, float)[:4]; e = [np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)]; k = int(np.argmax(e)); d = c[(k + 1) % 4] - c[k]
    return c.mean(0), np.degrees(np.arctan2(d[1], d[0])), max(e), min(e)


def level(box, pad=3, sc=1.0):
    m, ang, Lw, Lh = frame(box); L2, H2 = (Lw / 2 + pad) * sc, (Lh / 2 + pad) * sc
    M = cv2.getRotationMatrix2D(tuple(m), ang, sc); M[:, 2] += [L2 - m[0], H2 - m[1]]
    return cv2.warpAffine(g, M, (max(6, int(2 * L2)), max(6, int(2 * H2))), borderValue=255, flags=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_LINEAR)


# ---- 1. seeds
seeds = []
for name in ("rapidocr_4yon", "doctr_db50_parseq", "florence2_large"):
    p = test2.OUT / f"{name}__{a.key}.json"
    if not p.exists(): continue
    for r in json.load(open(p, encoding="utf-8"))["okumalar"]:
        ps = [p for p in test2.pairs(r["text"]) if 50 <= p[0] <= 300 and 130 <= p[1] <= 300]
        if ps: seeds.append({"box": r["box"], "pairs": ps, "src": name, "kind": "yazi"})
byp = {}
for sd in seeds: byp.setdefault(sd["pairs"][0], []).append(sd)
seeds = [x for v in byp.values() for x in v[:2]]
A1 = test2.K / "aciklik_ag" / "a1" / f"{fid}_tahmin.json"
if A1.exists():
    r1 = 100.0 / s if s > 130 else 1.0
    for b in json.load(open(A1)): seeds.append({"box": (np.asarray(b["box"]) / r1).tolist(), "pairs": [], "src": "a1_" + b["kind"], "kind": "simge"})
print("tohum", len(seeds), Counter(x["src"] for x in seeds), flush=True)

# ---- 2. matching
G = cv2.resize(g, None, fx=a.sc, fy=a.sc, interpolation=cv2.INTER_AREA).astype(np.float32); groups = []


def covered(box):
    c = np.asarray(box, float).mean(0)
    return any(any(np.hypot(*(c - np.asarray(i["box"]).mean(0))) < 12 for i in gr["inst"]) for gr in groups)


for sd in sorted(seeds, key=lambda x: x["kind"] != "yazi"):
    if covered(sd["box"]): continue
    q = level(sd["box"]); m0, ang0, Lw, Lh = frame(sd["box"])
    ink = q < 128
    if ink.any():                                            # tighten the seed to its ink (+3 px)
        yy, xx = np.where(ink); q = q[max(0, yy.min() - 3):yy.max() + 4, max(0, xx.min() - 3):xx.max() + 4]
    if q.shape[0] < 8 or q.shape[1] < 12 or (q < 128).mean() < 0.02: continue
    if Lw > (2.6 if sd["kind"] == "simge" else 1.2) * s: continue          # a text seed longer than 1.2 m is not one tag
    inst = []
    for rot in range(4):
        Q = cv2.resize(np.rot90(q, rot).copy(), None, fx=a.sc, fy=a.sc, interpolation=cv2.INTER_AREA).astype(np.float32)
        if min(Q.shape) < 4 or Q.std() < 1: continue
        R = cv2.matchTemplate(G, Q, cv2.TM_CCOEFF_NORMED); h, w = Q.shape
        k = max(3, int(0.6 * min(h, w)) | 1); mx = cv2.dilate(R, np.ones((k, k), np.uint8))   # local maxima only
        ys, xs = np.where((R > a.lo) & (R >= mx))
        if len(ys) > 300: o = np.argsort(-R[ys, xs])[:300]; ys, xs = ys[o], xs[o]
        for y, x in sorted(zip(ys, xs), key=lambda p: -R[p]):
            cx, cy = (x + w / 2) / a.sc, (y + h / 2) / a.sc
            if any(np.hypot(cx - i["c"][0], cy - i["c"][1]) < 0.6 * min(Lw, Lh) + 6 for i in inst): continue
            ww, hh = (w / a.sc, h / a.sc); box = [[cx - ww / 2, cy - hh / 2], [cx + ww / 2, cy - hh / 2], [cx + ww / 2, cy + hh / 2], [cx - ww / 2, cy + hh / 2]]
            inst.append({"c": (cx, cy), "box": box, "v": float(R[y, x]), "rot": rot})
    if len(inst) == 0: continue
    if len(inst) > a.maxkopya: continue                       # generic template: matches everywhere
    groups.append({"seeds": [sd], "inst": inst, "kind": sd["kind"], "Lw_cm": Lw / s * 100})
    if len(groups) % 10 == 0: print("  grup", len(groups), round(time.time() - t0), "s", flush=True)
print("grup", len(groups), "kopya", sum(len(x["inst"]) for x in groups), round(time.time() - t0), "s", flush=True)

# ---- 3. merge groups that share copies (same kind)
merged = []
for gr in groups:
    for mg in merged:
        if mg["kind"] != gr["kind"]: continue
        C = [i["c"] for i in mg["inst"]]; sh = sum(any(np.hypot(i["c"][0] - c[0], i["c"][1] - c[1]) < 12 for c in C) for i in gr["inst"])
        if sh >= 0.5 * min(len(gr["inst"]), len(mg["inst"])):
            mg["seeds"] += gr["seeds"]
            for i in gr["inst"]:
                if not any(np.hypot(i["c"][0] - c[0], i["c"][1] - c[1]) < 12 for c in C): mg["inst"].append(i)
            break
    else: merged.append(gr)
groups = merged; print("birlesik grup", len(groups), flush=True)

# ---- 4. GLM readings: best copies of each group + weak copies of text groups
from transformers import AutoProcessor, AutoModelForImageTextToText
MP = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\GLM-OCR"; assert "5070" in torch.cuda.get_device_name(0)
model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP)


MPV = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\PaddleOCR-VL-1.6"
vmodel = AutoModelForImageTextToText.from_pretrained(MPV, dtype=torch.bfloat16).to("cuda").eval(); vproc = AutoProcessor.from_pretrained(MPV)


def vl(box):
    out = []
    v = level(box, pad=4, sc=2.0)
    for im in (v, cv2.rotate(v, cv2.ROTATE_180)):
        msg = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(im).convert("RGB")}, {"type": "text", "text": "OCR:"}]}]
        inp = vproc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad(): o = vmodel.generate(**inp, max_new_tokens=32, do_sample=False, repetition_penalty=1.05)
        out.append(vproc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).replace("\n", " "))
    return out


def pairs_w(t, wcm):
    """number pairs whose first number fits the symbol width; slash-less digit runs ('150140') split the same way"""
    out = [p for p in test2.pairs(t) if abs(p[0] - wcm) <= 15]
    for run in re.findall(r"\d{5,7}", t.upper().replace("O", "0")):
        for k in (2, 3):
            a_, b_ = int(run[:k]), run[k:]
            for b2 in {b_[:3], b_[-3:]}:
                if len(b2) == 3 and abs(a_ - wcm) <= 15 and 100 <= int(b2) <= 300: out.append((a_, int(b2)))
    return out


def glm(box):
    out = []
    v = level(box, pad=4, sc=2.0)
    for im in (v, cv2.rotate(v, cv2.ROTATE_180)):
        msg = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(im).convert("RGB")}, {"type": "text", "text": "Text Recognition:"}]}]
        inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda"); inp.pop("token_type_ids", None)
        with torch.no_grad(): o = model.generate(**inp, max_new_tokens=32, do_sample=False, repetition_penalty=1.05)
        out.append(proc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).replace("\n", " "))
    return out


nread = 0; reads = []
for gr in groups:
    votes = Counter()
    for sd in gr["seeds"]:
        for p in sd["pairs"]: votes[p] += 2
    best = sorted(gr["inst"], key=lambda i: -i["v"])[:4 if gr["kind"] == "yazi" else 3]
    if gr["kind"] == "simge": best = sorted(gr["inst"], key=lambda i: -i["v"])[:6]
    for i in best:
        i["glm"] = glm(i["box"]) + (vl(i["box"]) if gr["kind"] == "simge" else []); nread += 1
        for t in i["glm"]:
            for p in (pairs_w(t, gr["Lw_cm"]) if gr["kind"] == "simge" else test2.pairs(t)): votes[p] += 1
    if not votes:
        if gr["kind"] == "simge":           # unreadable symbol group: width from the drawing, height flagged
            w_ = int(round(gr["Lw_cm"] / 10.0) * 10); gr["sayi"] = (w_, None)
            for i in gr["inst"]:
                if i["v"] >= a.simge_lo: reads.append({"box": i["box"], "text": f"{w_}/?", "v": round(i["v"], 3), "grup": groups.index(gr)})
        continue
    (n1, n2), nv = votes.most_common(1)[0]; gr["sayi"] = (n1, n2)
    dig = f"{n1}{n2}"; weak = sorted([i for i in gr["inst"] if i["v"] < a.hi], key=lambda i: -i["v"])[:12]; wset = {id(i) for i in weak}
    for i in gr["inst"]:
        if i["v"] < a.hi and id(i) not in wset: continue
        if gr["kind"] == "simge" and i["v"] < a.simge_lo: continue
        if i["v"] < a.hi and gr["kind"] == "yazi":
            if "glm" not in i: i["glm"] = glm(i["box"]); nread += 1
            if not any(dig in re.sub(r"\D", "", t.upper().replace("O", "0").replace("Z", "2")) for t in i["glm"]): continue
        reads.append({"box": i["box"], "text": f"{n1}/{n2}", "v": round(i["v"], 3), "grup": groups.index(gr)})
    if groups.index(gr) % 10 == 0: print("  okunan grup", groups.index(gr), "/", len(groups), nread, round(time.time() - t0), "s", flush=True)
print("glm okuma", nread, "kabul edilen kopya", len(reads), round(time.time() - t0), "s", flush=True)
json.dump([{"kind": gr["kind"], "sayi": gr.get("sayi"), "tohum": [sd["src"] for sd in gr["seeds"]], "kopya": len(gr["inst"])} for gr in groups],
          open(H / "SONUC2" / f"_{a.tag}_gruplar__{a.key}.txt", "w"), indent=0)
test2.save(a.tag, a.key, reads, time.time() - t0, {"grup": len(groups), "glm_okuma": nread})
