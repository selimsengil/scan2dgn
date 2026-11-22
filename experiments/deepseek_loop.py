"""DeepSeek decides, the tools draw: one floor, room by room.

For every room whose measured area disagrees with its printed area (or has none), the
model sees the room view (araclar.Kat.room_view) and answers JSON: which numbered
openings to close or open. The tools apply it and re-measure. Scored against the DGN
rooms before and after.

usage: python deepseek_loop.py --floor 015 --out runs/015_a [--thinking --budget 1500] [--rounds 2]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\scripts")
sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io")
import colibri_v41_session_20261105 as S  # noqa: E402
import bench_replay as BR  # noqa: E402

import araclar as A  # noqa: E402

EXE = Path(r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io\deepseek_v41_pf.exe")

SYSTEM = ("Sen mimari kat planlarını okuyan dikkatli bir teknik ressamsın. Görevin, bir aracın çizdiği oda "
          "alanını kontrol etmek ve aracın açıklık kararlarını düzeltmek. Koordinat yazmazsın; yalnız numaralı "
          "seçenekler arasından seçersin. Cevabın yalnız tek bir JSON nesnesidir.")

USER = """Görüntüde bir kat planı kesiti var. MAVİ boyalı alan, aracın "{name}" odası için bulduğu alan.
Planda bu odanın yazılı alanı {area_text}; aracın ölçtüğü alan {area_m2} m².

Kurallar:
- Oda sınırı duvarın odaya bakan iç yüzünden geçer.
- Kapı ve pencere açıklıkları duvar doğrultusunda düz kapatılır; oda kapıdan komşu odaya, koridora, balkona ya da dışarı taşmaz.
- Mobilya, yazı, ölçü ve aks çizgileri sınır değildir.

Görüntüdeki numaralı çizgiler aracın açıklık adayları:
- TURUNCU numaralı çizgi: araç bu açıklığı KAPATTI (sınır sayılıyor).
- CAMGÖBEĞİ numaralı çizgi: araç bu adayı AÇIK bıraktı (geçilebilir sayılıyor).
Bu kesitteki adaylar: {openings}

Mavi alan odayı doğru gösteriyor mu? Komşu bir alana taşıyorsa hangi camgöbeği adayın kapatılması gerektiğini,
odanın bir parçası yanlışlıkla dışarıda kaldıysa hangi turuncu çizginin açılması gerektiğini seç.
Yalnız şu JSON ile cevap ver:
{{"dogru": true veya false, "kapat": [numaralar], "ac": [numaralar], "sorun": "kısa açıklama"}}"""


def parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"_error": "no json", "_raw": raw[:400]}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {"_error": "bad json", "_raw": raw[:400]}
    for k in ("kapat", "ac"):
        d[k] = [int(x) for x in d.get(k, []) if str(x).strip().lstrip("-").isdigit()]
    return d


def start_session(out: Path, thinking_budget: int, extra_env: dict | None = None) -> S.ColibriV41Session:
    contract = S.load_json(BR.REF / "CONTRACT.json")
    contract["env"].update(extra_env or {})
    usage = out / "USAGE.bin"
    shutil.copyfile(BR.SEED, usage)
    contract["args"][0] = str(EXE)
    contract["exe_sha256"] = S.sha256_file(EXE)
    contract["env"]["COLI_USAGE"] = str(usage)
    contract["env"]["V41_READ_DEPTH"] = "48"
    contract["tag"] = "oda_kural_deepseek_loop"
    BR.apply_gpus(contract, "5070,,")
    if contract["env"].get("COLI_VK_TIER") == "0":   # no tier: its startup lines (host pool, ...) are absent
        contract["verified_markers"] = [m for m in contract.get("verified_markers", []) if "NVIDIA" in m]
    sess = S.ColibriV41Session(out / "session", contract)
    sess.start(timeout_s=300)
    return sess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default="015")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.05, help="area agreement that needs no model")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    k = A.Kat(a.floor)
    k.segment()
    before = k.score()
    print("tools only:", {x: before[x] for x in ("ge95", "ge90", "ge80", "mean")}, flush=True)
    log = {"floor": a.floor, "args": vars(a), "before": before, "steps": []}

    sess = start_session(out, a.budget)
    n = 0
    try:
        for rnd in range(1, a.rounds + 1):
            rooms = k.segment()
            seen = set()
            for r in rooms:
                i = r["i"]
                if k.root[i] in seen:
                    continue            # the same region as an earlier seed
                seen.add(k.root[i])
                if r["ok"] and rnd == 1 and r["area_text"]:
                    continue            # area agrees: accepted without the model
                n += 1
                img = out / f"r{rnd}_{i:02d}.png"
                view = k.room_view(i, img)
                ops = ", ".join(f"{o['id']} ({'kapalı' if o['closed'] else 'açık'}, {o['width_m']} m)" for o in view["openings"]) or "yok"
                at = f"{r['area_text']} m²" if r["area_text"] else "yazılı değil"
                prepared = S.prepare_multimodal_payload(img, SYSTEM, USER.format(name=r["name"], area_text=at, area_m2=r["area_m2"], openings=ops),
                                                        enable_thinking=a.thinking, reasoning_effort=a.effort)
                t0 = time.time()
                res = sess.submit_multimodal(out / f"req_{n:03d}", prepared, generation_budget=a.budget, timeout_s=3600)
                d = parse(res["raw"])
                ids = {o["id"] for o in view["openings"]}
                close = [x for x in d.get("kapat", []) if x in ids]
                opn = [x for x in d.get("ac", []) if x in ids]
                k.closed |= set(close); k.closed -= set(opn)
                after_rooms = k.segment()
                ar = after_rooms[i]
                step = {"round": rnd, "seed": i, "name": r["name"], "area_text": r["area_text"], "area_before": r["area_m2"],
                        "area_after": ar["area_m2"], "answer": d, "applied_close": close, "applied_open": opn,
                        "seconds": round(time.time() - t0, 1), "tokens": res.get("generated_tokens")}
                log["steps"].append(step)
                print(json.dumps(step, ensure_ascii=False), flush=True)
                json.dump(log, open(out / "LOG.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    finally:
        sess.close()
    k.segment()
    after = k.score()
    log["after"] = after
    print("tools only :", {x: before[x] for x in ("ge95", "ge90", "ge80", "mean")})
    print("with model :", {x: after[x] for x in ("ge95", "ge90", "ge80", "mean")})
    k.render(out / "final.png", gt=True)
    json.dump(log, open(out / "LOG.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
