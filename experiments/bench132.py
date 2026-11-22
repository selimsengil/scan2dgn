"""132-room benchmark: every DGN room of the 7 labelled floors, drawn by a model in 0-1000 grounding.

Per room: crop = DGN room bbox + margin (the room's location is given; this measures DRAWING, not
finding), target = DGN name + printed area. No door hint. Long crops are downscaled to <= max_side.
Answer: corners in 0-1000 of the crop. Scored raw and snapped (snap.py) with the benchmark metrics.

backends:
  openai  : an OpenAI-compatible server (llama-server: Qwen or DeepSeek GGUF), few-shot possible
  colibri : the local Colibri DeepSeek-V4.1 engine (one image per request: no few-shot)

usage: python bench132.py --backend openai --url http://127.0.0.1:8091 --tag qwen_noshot
       python bench132.py --backend openai --shots 06,05,03 --tag qwen_fewshot
       python bench132.py --backend colibri --exe ...deepseek_v41_gv.exe --tag ds500_noshot
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import base64
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import qwen_test as QT
import snap
import v0_watershed as V0

FLOORS = ["001", "002", "010", "013", "015", "016", "019"]
MIN_SIDE = 700
MARK = True   # a red dot on the room's printed label: which room is meant
RULES = QT.RULES + (" Ölçü çizgileri ince, uçlarında çentik ya da ok olan ve üstünde sayı yazan çizgilerdir; sınır değildir.")
SYSTEM = ("Sen mimari kat planlarından oda sınırı çizen dikkatli bir teknik ressamsın. " + RULES +
          " Çözülmüş örnek verilirse onu incele ve yeni odada aynı kuralları aynı şekilde uygula. Yalnız JSON yaz.")
Q = ("Görüntü bir kat planından '{target}' odasının kesiti. Çizilecek oda, KIRMIZI noktanın bulunduğu odadır (nokta odanın "
     "yazısının yerinde). Komşu odalar da görünebilir; yalnız kırmızı noktayı içeren odayı çiz. Odanın iç sınırının "
     "köşelerini saat yönünde sırayla, görüntünün 0-1000 normalize koordinatlarında ver. "
     "JSON: {{\"points\": [[x, y], ...], \"neden\": \"kısa açıklama\"}}")


def data_url(img: Image.Image) -> str:
    b = io.BytesIO(); img.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def rooms(max_side: int):
    for f in FLOORS:
        img = Image.open(V0.B / f / "clean.png").convert("RGB")
        gts = V0.load_rooms(f)
        s = V0.oracle_scale(gts)
        for k, g in enumerate(gts):
            if g["poly"].geom_type != "Polygon":
                continue
            x0, y0, x1, y1 = g["poly"].bounds
            m = max(0.6 * s, 0.12 * max(x1 - x0, y1 - y0))
            bx = [max(0, int(x0 - m)), max(0, int(y0 - m)), min(img.width, int(x1 + m)), min(img.height, int(y1 + m))]
            crop = img.crop(bx)
            # small crops (low-DPI scans, WCs, shafts) up to a short side of MIN_SIDE; big ones down to max_side
            f_down = min(max_side / max(crop.size), max(1.0, MIN_SIDE / min(crop.size)))
            if f_down != 1:
                crop = crop.resize((int(crop.width * f_down), int(crop.height * f_down)),
                                   Image.Resampling.LANCZOS if f_down < 1 else Image.Resampling.BICUBIC)
            ref = [[(x - bx[0]) * f_down, (y - bx[1]) * f_down] for x, y in list(g["poly"].exterior.coords)[:-1]]
            name = (g["name"] or "ODA") + (f" {g['area_m2']:.2f} m2" if g["area_m2"] else "")
            label = [(g["xy"][0] - bx[0]) * f_down, (g["xy"][1] - bx[1]) * f_down]   # the printed room label
            yield f"{f}_{k:02d}", f, name, crop, ref, label


def shot_messages(numbers):
    import qwen_fewshot_test as FS
    return FS.example_messages(set(numbers.split(","))) if numbers else []


def teach_messages():
    """The teaching set (three legend sheets + TEACH.md) as one user turn and a short acknowledgement.
    Same for every room, so llama-server keeps it in its prompt cache."""
    import ds_agent2 as A2
    intro, a, b, c, _how = A2.sections()
    how = ("HOW TO ANSWER: give the corners of the room's inner boundary in clockwise order, in the task "
           "image's normalised 0-1000 coordinates (x right, y down). Corners only where the boundary changes "
           "direction; follow column shoulders and wall returns; close doors straight on the inner face.")
    content = [{"type": "text", "text": intro}]
    for title, sheet, txt in (("SHEET A", "SHEET_A.png", a), ("SHEET B", "SHEET_B.png", b), ("SHEET C", "SHEET_C.png", c)):
        content += [{"type": "text", "text": title + ":"},
                    {"type": "image_url", "image_url": {"url": data_url(Image.open(A2.TEACH / sheet).convert("RGB"))}},
                    {"type": "text", "text": txt}]
    content.append({"type": "text", "text": how})
    return [{"role": "user", "content": content},
            {"role": "assistant", "content": "Understood. I will apply these conventions to every room."}]


def _post(url, body):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def ask_openai(url, msgs, max_tokens=900, stop=None, think=0):
    """think > 0: thinking on with that many tokens for the thought (+900 for the answer). If the thought
    runs out before an answer, a second call without thinking gets the thought as notes and must answer."""
    body = {"messages": msgs, "max_tokens": max_tokens + think, "temperature": 0.6 if think else 0,
            "chat_template_kwargs": {"enable_thinking": bool(think)}}
    if stop and not think:
        body["stop"] = stop
    r = _post(url, body)
    m = r["choices"][0]["message"]
    content, thought = m.get("content") or "", m.get("reasoning_content") or ""
    tm = dict(r.get("timings", {}))
    tm["thought_chars"] = len(thought)
    tm["forced"] = False
    if think and '"points"' not in content:
        notes = (thought or content)[-12000:]
        msgs2 = msgs[:-1] + [{"role": "user", "content": msgs[-1]["content"] + [
            {"type": "text", "text": "Your own analysis of this room (time is up, do not continue it):\n" + notes +
             "\n\nNow answer ONLY with the JSON. Build the polygon FROM THIS ANALYSIS: use the walls, columns, "
             "corners and coordinates you identified in it; where it was unfinished, complete it consistently."}]}]
        r2 = _post(url, {"messages": msgs2, "max_tokens": max_tokens, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}})
        content = r2["choices"][0]["message"].get("content") or ""
        t2 = r2.get("timings", {})
        tm["predicted_n"] = (tm.get("predicted_n") or 0) + (t2.get("predicted_n") or 0)
        tm["forced"] = True
    return content, tm


STAT_KEYS = ("vision_seconds", "vision_rows", "prefill_only_seconds", "prefill_disk_seconds", "prefill_io_mb", "first_output_seconds",
             "decode_seconds", "wall_seconds", "prompt_tokens", "generated_tokens", "decode_tps", "expert_io_gbps", "cpu_cache_hit_pct",
             "prof_disk_s", "prof_expert_s", "prof_attn_s", "prof_engram_s")


class Colibri:
    def __init__(self, exe: str, out: Path, snap_dir: str = "", extra_env: dict | None = None):
        sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io")
        import deepseek_loop as L1   # start_session (contract, gpus by name, READ_DEPTH 48)
        L1.EXE = Path(exe)
        self.S = L1.S
        env = {"COLI_VK_SHADERS": str(Path(exe).parent / "shaders")}
        if snap_dir:
            env["SNAP"] = snap_dir
        env.update(extra_env or {})
        self.sess = L1.start_session(out, 900, env)
        self.n = 0
        self.out = out

    def ask(self, img: Image.Image, text: str, budget: int = 700, stop_text: str | None = None, strata_like: bool = False):
        if strata_like:
            # Strata's input: the picture resized (not padded) to sides that are multiples of 42 within 1024 tokens,
            # and the official template's blank line between the image and the text
            sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\third_party\strata-control-0.1.58")
            from serve.ds_pictures import fit_size
            w, h = fit_size(img.width, img.height)
            if (w, h) != img.size:
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            text = "\n\n" + text
        p = self.out / "_img.png"; img.save(p)
        prepared = self.S.prepare_multimodal_payload(p, SYSTEM, text, enable_thinking=False)
        self.n += 1
        r = self.sess.submit_multimodal(self.out / f"req_{self.n:03d}", prepared, generation_budget=budget, timeout_s=3600,
                                        stop_text=stop_text)
        return r["raw"], {"predicted_n": r.get("generated_tokens"), "decode_tps": r.get("decode_tps"),
                          "stats": {k: r.get(k) for k in STAT_KEYS}}

    def close(self):
        self.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "colibri"], default="openai")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--exe", default=r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io\deepseek_v41_gv.exe")
    ap.add_argument("--shots", default="")
    ap.add_argument("--teach", action="store_true", help="openai: put the teaching set (teach/) before the shots")
    ap.add_argument("--think", type=int, default=0, help="openai: thinking on with this many thought tokens")
    ap.add_argument("--snap", default="", help="colibri: model folder (default: the 500 GB original)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max_side", type=int, default=1400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--env", action="append", default=[], help="colibri: extra engine env K=V")
    ap.add_argument("--only", default="", help="comma list of room keys (e.g. 001_03) to run")
    ap.add_argument("--budget", type=int, default=700, help="colibri: generation budget")
    ap.add_argument("--strata-like", dest="strata_like", action="store_true",
                    help="colibri: resize the crop to Strata's size (multiples of 42) and add the template's blank line")
    ap.add_argument("--stop-after-points", dest="stop_points", action="store_true",
                    help="end the answer when it reaches \"neden\" (the points come first: they are not changed)")
    a = ap.parse_args()
    out = Path(__file__).with_name("runs") / ("b132_" + a.tag); out.mkdir(parents=True, exist_ok=True)
    resf = out / "RESULT.json"
    res = json.load(open(resf, encoding="utf-8")) if resf.exists() else {}
    shots = shot_messages(a.shots) if a.backend == "openai" else []
    if a.teach and a.backend == "openai":
        shots = teach_messages() + shots
    col = Colibri(a.exe, out, a.snap, dict(kv.split("=", 1) for kv in a.env)) if a.backend == "colibri" else None
    only = set(a.only.split(",")) if a.only else None
    n = 0
    try:
        for key, f, name, crop0, ref, label in rooms(a.max_side):
            if key in res or (only and key not in only):
                continue
            if a.limit and n >= a.limit:
                break
            n += 1
            crop = crop0.copy()
            if MARK:
                from PIL import ImageDraw
                rr = max(6, int(0.012 * max(crop.size)))
                ImageDraw.Draw(crop).ellipse((label[0] - rr, label[1] - rr, label[0] + rr, label[1] + rr), fill=(230, 0, 0))
            W, H = crop.size
            text = Q.format(target=name)
            t0 = time.time()
            try:
                if col:
                    ans, tm = col.ask(crop, text, a.budget, '"neden"' if a.stop_points else None, a.strata_like)
                else:
                    msgs = [{"role": "system", "content": SYSTEM}] + shots + [
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(crop)}}, {"type": "text", "text": text}]}]
                    ans, tm = ask_openai(a.url, msgs, stop=['"neden"'] if a.stop_points else None, think=a.think)
            except Exception as e:
                ans, tm = "", {"error": str(e)[:200]}
            js = QT.parse_json(ans)
            try:
                px = [[float(x) * W / 1000.0, float(y) * H / 1000.0] for x, y in (js.get("points") or [])]
            except Exception:
                px = []
            if len(px) >= 3:
                import repair_rescore
                px = repair_rescore.repair(px)   # ring traced twice / self-touching -> one simple ring
            sn = snap.snap_polygon(crop0, px) if len(px) >= 3 else []
            seed = None
            score = lambda p: QT.brief(QT.score(p, ref, type("H", (), {"inside": None})())) if len(p) >= 3 else {"valid": False, "reason": "no polygon"}
            res[key] = {"floor": f, "name": name, "size": [W, H], "points": px, "snapped": sn, "reference": ref,
                        "raw": score(px), "snap": score(sn), "neden": js.get("neden"), "seconds": round(time.time() - t0, 1),
                        "tokens": tm.get("predicted_n"), "tps": tm.get("predicted_per_second") or tm.get("decode_tps"), "error": tm.get("error"),
                        "engine": tm.get("stats"), "prompt_n": tm.get("prompt_n"), "thought_chars": tm.get("thought_chars"),
                        "forced": tm.get("forced")}
            crop.save(out / f"{key}.png")
            r = res[key]
            print(key, name[:24], "raw", r["raw"].get("IoU"), r["raw"].get("F1@5"), "| snap", r["snap"].get("IoU"), r["snap"].get("F1@5"), "|", r["seconds"], "s", flush=True)
            json.dump(res, open(resf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    finally:
        if col:
            col.close()
    summarize(res, out)


def summarize(res, out):
    def agg(kind):
        v = [r[kind] for r in res.values()]
        ok = [x for x in v if x.get("valid", True) and "IoU" in x]
        iou = np.array([x["IoU"] for x in ok]) if ok else np.array([])
        f5 = np.array([x["F1@5"] for x in ok]) if ok else np.array([])
        return {"rooms": len(v), "valid": len(ok), "IoU>=0.95": int((iou >= .95).sum()), "IoU>=0.90": int((iou >= .90).sum()),
                "F1@5>=0.9": int((f5 >= .9).sum()), "mean_IoU": round(float(iou.mean()), 3) if len(iou) else 0,
                "mean_F1@5": round(float(f5.mean()), 3) if len(f5) else 0}
    s = {"raw": agg("raw"), "snap": agg("snap"),
         "mean_seconds": round(float(np.mean([r["seconds"] for r in res.values()])), 1) if res else 0}
    json.dump(s, open(out / "SUMMARY.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("SUMMARY", json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
