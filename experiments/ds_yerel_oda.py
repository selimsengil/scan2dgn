"""Single-room test on the LOCAL DeepSeek V4.1 engine (never 015 for development): one request per room, the model
writes the room's corners itself. Measures quality (IoU against GT) and the real time per call.
usage: python ds_yerel_oda.py --rooms 11,7 --tag d1 [--think] [--budget 220] [--side 640]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\scripts")
sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io")
import v0_watershed as V0  # noqa: E402
import tek_oda_test as T  # noqa: E402
import qwen_izgara as QI  # noqa: E402
import colibri_v41_session_20261105 as S  # noqa: E402
import bench_replay as BR  # noqa: E402

IO = Path(r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io")
SYS = "You are inspecting a scanned architectural floor plan. Follow the requested output format."

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="019"); ap.add_argument("--rooms", default="11,7"); ap.add_argument("--tag", default="d1")
    ap.add_argument("--think", action="store_true"); ap.add_argument("--effort", default="low"); ap.add_argument("--budget", type=int, default=220)
    ap.add_argument("--side", type=int, default=640); ap.add_argument("--exe", default=str(IO / "deepseek_v41_pf.exe"))
    a = ap.parse_args(); assert a.floor != "015"
    out = T.OUT / f"ds_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    usage = out / "USAGE.bin"; shutil.copyfile(BR.SEED, usage)
    contract = S.load_json(BR.REF / "CONTRACT.json")
    contract["args"][0] = a.exe; contract["exe_sha256"] = S.sha256_file(Path(a.exe))
    contract["env"].update({"COLI_USAGE": str(usage), "COLI_VK_SHADERS": str(IO / "shaders"), "V41_STATS": "1"})
    contract["tag"] = "tek_oda_" + a.tag
    BR.apply_gpus(contract, "5070")
    im = Image.open(V0.B / a.floor / "clean.png").convert("RGB"); W, H = im.size
    gts = V0.load_rooms(a.floor); rows = []; tiles = []
    sess = S.ColibriV41Session(out / "session", contract)
    try:
        ready = sess.start(timeout_s=400); print("load", round(ready["load_seconds"], 1), flush=True)
        for k, i in enumerate(int(v) for v in a.rooms.split(",")):
            g = gts[i]; box = T.crop_box(g["poly"], W, H, [11, 18, 7, 21, 14].index(i) if i in (11, 18, 7, 21, 14) else k)
            c = im.crop(box); s = a.side / max(c.size); c = c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)
            ip = out / f"r{i}.png"; c.save(ip)
            m2 = f' (area text "A:{g["area_m2"]} m2")' if g["area_m2"] and g["area_m2"] > 0 else ""
            prep = S.prepare_multimodal_payload(ip, SYS, QI.P1R.format(name=g["name"], m2=m2, rules=QI.RULES), enable_thinking=a.think, reasoning_effort=a.effort)
            r = sess.submit_multimodal(out / f"req_r{i}", prep, generation_budget=a.budget, timeout_s=3600)
            raw = r["raw"]; ans = raw.split("</think>")[-1]
            pts = T.parse_points(ans)
            pred = [[box[0] + x / 1000 * (box[2] - box[0]), box[1] + y / 1000 * (box[3] - box[1])] for x, y in pts]
            sc = T.score(pred, g["poly"]) if len(pred) >= 3 else {"iou": 0.0}
            rows.append({"room": i, "prompt_tokens": r["prompt_tokens"], "vision_rows": r["vision_rows"], "prefill_s": round(r["first_output_seconds"], 1),
                         "decode_s": round(r["decode_seconds"], 1), "gen_tokens": r["generated_tokens"], "tps": round(r["decode_tps"], 2), "limited": r.get("limited"), **sc})
            print(rows[-1], flush=True); print("   answer:", ans[-300:].replace("\n", " "), flush=True)
            tiles.append(T.tile(im, box, g["poly"], pred if len(pred) >= 3 else None, f"r{i} DeepSeek yerel IoU {sc.get('iou')} {r['wall_seconds']:.0f}s"))
            json.dump({"row": rows[-1], "raw": raw, "pred": pred}, open(out / f"r{i}.json", "w", encoding="utf-8"), ensure_ascii=False)
    finally:
        sess.close()
    if tiles:
        sh = Image.new("RGB", (sum(t.width for t in tiles) + 10 * len(tiles), max(t.height for t in tiles)), "white"); x = 0
        for t in tiles: sh.paste(t, (x, 0)); x += t.width + 10
        sh.save(T.OUT / f"sayfa_ds_{a.tag}.png")
