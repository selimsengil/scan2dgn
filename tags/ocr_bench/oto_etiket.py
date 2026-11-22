"""Auto-label door/window tags on the training floors with PaddleOCR-VL-1.6 spotting (5070 Ti only).
Per reliable DGN opening (aciklik_ag/veri): crop its bbox + 0.6 m (sheet orientation), spotting -> text lines with 4-point
boxes.  A tag is accepted when a 'number/number' or 'number x number' read has its first number within 6 cm of the DGN
width.  The reader never decides door/window: it only gives the place and the numbers.
Output: oto/<floor>.json {openings: [{id, kind, dgn_cm, kabul, metin, sayilar, box}], okumalar: [...all reads...]}
usage: python oto_etiket.py [--only substr] [--limit N] [--batch 8]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, os, re, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from shapely.geometry import shape
from transformers import AutoProcessor, AutoModelForImageTextToText
K = config.KT; V = K / "aciklik_ag" / "veri"; R = K / "dgn_katlar"; OUT = config.KT / "ocr_test" / "oto"
MP = r"C:\Users\Selim\Documents\Proje\ocr_modelleri\PaddleOCR-VL-1.6"
UP = os.environ.get("OTO_UP", "0") == "1"
PAT = re.compile(r"(\d{2,3})[/xX×*1Il|]?(\d{3})")   # 90/210, 90x220, 901210 (slash read as 1), 90210
SKIP = ("kisla", "malkoc_361_79__k2")
model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.bfloat16).to("cuda").eval(); proc = AutoProcessor.from_pretrained(MP)
proc.tokenizer.padding_side = "left"
assert "5070" in torch.cuda.get_device_name(0)
def spot(imgs):
    ims = []
    for a in imgs:
        im = Image.fromarray(a).convert("RGB")
        if UP and max(im.size) < 1500: im = im.resize((im.size[0] * 2, im.size[1] * 2), Image.LANCZOS)
        ims.append(im)
    msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": "Spotting:"}]}] for im in ims]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True,
                                   images_kwargs={"size": {"shortest_edge": 112896, "longest_edge": 2048 * 28 * 28}}).to(model.device)
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=320, do_sample=False, repetition_penalty=1.05)
    return [proc.decode(o[inp["input_ids"].shape[-1]:], skip_special_tokens=False).replace("</s>", "").replace("<|end_of_sentence|>", "") for o in out]
def parse(txt, x0, y0, w, h):
    out = []
    for line in txt.split("\n"):
        locs = [int(v) for v in re.findall(r"<\|LOC_(\d+)\|>", line)]; t = re.sub(r"<\|[^|]*\|>", "", line).strip()
        if len(locs) >= 8 and t: out.append({"text": t, "box": [[round(x0 + locs[2 * i] / 1000 * w, 1), round(y0 + locs[2 * i + 1] / 1000 * h, 1)] for i in range(4)]})
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=1); a = ap.parse_args()
    OUT.mkdir(exist_ok=True); floors = [p.stem for p in sorted(V.glob("*.json")) if not any(k in p.stem.lower() for k in SKIP) and a.only.lower() in p.stem.lower()]
    T0 = time.time(); tot = {"door": [0, 0], "window": [0, 0]}
    for fid in floors:
        if (OUT / f"{fid}.json").exists() and not a.limit: continue
        d = json.load(open(V / f"{fid}.json", encoding="utf-8")); s = d["px_per_m"]; ops = d["openings"][: a.limit or None]
        g = cv2.imdecode(np.fromfile(str(R / fid / "clean.png"), np.uint8), 0); H, W = g.shape; jobs = []
        for o in ops:
            P = shape(o["poly"]); x0, y0, x1, y1 = P.bounds; m = 0.6 * s; x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m)); x1, y1 = int(min(W, x1 + m)), int(min(H, y1 + m))
            c = np.array(P.minimum_rotated_rectangle.exterior.coords)[:4]; wcm = max(np.hypot(*(c[(k + 1) % 4] - c[k])) for k in range(4)) / s * 100
            jobs.append((o, (x0, y0, x1 - x0, y1 - y0), g[y0:y1, x0:x1], wcm))
        rows = []; reads = []; t0 = time.time()
        for i in range(0, len(jobs), a.batch):
            B = jobs[i:i + a.batch]; txts = spot([j[2] for j in B])
            for (o, (x0, y0, w, h), _, wcm), txt in zip(B, txts):
                rd = parse(txt, x0, y0, w, h); reads += rd; best = None
                for r in rd:
                    for mt in PAT.finditer(r["text"].replace(" ", "")):
                        n1, n2 = int(mt.group(1)), int(mt.group(2))
                        if not 100 <= n2 <= 260: continue
                        if abs(n1 - wcm) <= 12 and (best is None or abs(n1 - wcm) < best[0]): best = (abs(n1 - wcm), r, [n1, n2])
                rows.append({"id": o["id"], "kind": o["kind"], "dgn_cm": round(wcm, 1), "kabul": best is not None,
                             "metin": best[1]["text"] if best else None, "sayilar": best[2] if best else None, "box": best[1]["box"] if best else None})
        json.dump({"floor": fid, "px_per_m": s, "openings": rows, "okumalar": reads, "sure_s": round(time.time() - t0, 1)}, open(OUT / f"{fid}.json", "w", encoding="utf-8"), ensure_ascii=False)
        for k in tot:
            R_ = [r for r in rows if r["kind"] == k]; tot[k][0] += len(R_); tot[k][1] += sum(r["kabul"] for r in R_)
        print(f"{fid[:44]:46} kapi {sum(r['kabul'] for r in rows if r['kind']=='door')}/{sum(r['kind']=='door' for r in rows)} pencere {sum(r['kabul'] for r in rows if r['kind']=='window')}/{sum(r['kind']=='window' for r in rows)} {time.time()-t0:.0f}s top {time.time()-T0:.0f}s", flush=True)
    print("TOPLAM", json.dumps(tot))
