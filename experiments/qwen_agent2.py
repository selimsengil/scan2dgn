"""Qwen draws a room in 6.1 Sol's recommended staged flow (codex_cevap_03.md, item 7):

  1 start     : room crop at original pixels (ruler), target name, verified entrance in orange.
                Qwen describes the loop in words (which faces, where it turns).
  2 measure   : at most 2 focus + 1 profile, each naming what it resolves.
  3 draft     : measuring tools are withdrawn; Qwen MUST write the ordered corners (or unresolved).
  4 correct   : the draft is drawn thin on the raster; Qwen answers keep / replace_vertex / unresolved, once.
Draft and corrected polygons are both scored (benchmark metrics vs the DGN room).

usage: python qwen_agent2.py [--think] [--out runs/qwen_agent2_a] [--cases 04_salon]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import time
from pathlib import Path

from PIL import ImageDraw

import qwen_agent as QA
import qwen_test as QT

SYSTEM = QA.SYSTEM.split("Araçların")[0] + "Her mesajda yalnız istenen JSON'u yaz."


def img_msg(img, text):
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": QA.to_data_url(img)}}, {"type": "text", "text": text}]}


def draft_view(crop, hint, pts):
    W, H = crop.size
    v = QA.ruled_view(crop, (0, 0, W, H), 1, hint)
    d = ImageDraw.Draw(v)
    if len(pts) >= 2:
        q = [(p[0] + QA.PAD, p[1] + QA.PAD) for p in pts]
        d.line(q + q[:1], fill=(255, 120, 0), width=1)
        for i, (x, y) in enumerate(q):
            d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(220, 0, 160), width=2)
            d.text((x + 5, y - 14), str(i), fill=(220, 0, 160), font=QA.FONT)
    return v


def as_points(obj):
    try:
        return [[float(p[0]), float(p[1])] for p in obj]
    except Exception:
        return []


def run_case(name, crop, target, hint, ref, a, out: Path):
    W, H = crop.size
    ask = lambda msgs, n: QA.chat(msgs, a.think, (8000 if a.think else n))
    over = QA.ruled_view(crop, (0, 0, W, H), 1, hint)
    over.save(out / f"{name}_0_overview.png")
    msgs = [{"role": "system", "content": SYSTEM},
            img_msg(over, f"Bu, '{target}' odasının bulunduğu kesit ({W}x{H} piksel, özgün çözünürlük, cetvel piksel koordinatı). "
                          "TURUNCU çizgi odanın giriş kapısı açıklığı (doğrulanmış; sınır buradan düz geçer).\n"
                          "ADIM 1: Odanın iç sınırını kelimelerle tarif et: saat yönünde hangi duvar iç yüzlerinden geçiyor, "
                          "nerede dönüyor (kolon, girinti, eğim), kapıyı nerede düz geçiyor. Yaklaşık cetvel değerlerini de yaz. "
                          "JSON: {\"loop\": \"...\"}")]
    log, t0 = {}, time.time()
    text, _ = ask(msgs, 700); msgs.append({"role": "assistant", "content": text}); log["loop"] = QA.action_of(text).get("loop", text[-500:])
    # 2: measure, budget 2 focus + 1 profile
    left = {"focus": 2, "profile": 1}
    log["measure"] = []
    for _ in range(3):
        msgs.append({"role": "user", "content":
            f"ADIM 2 (ölçüm, kalan hak: focus {left['focus']}, profile {left['profile']}): En belirsiz köşe ya da yüzü çöz. "
            "Seçenekler: {\"action\":\"focus\",\"box\":[x0,y0,x1,y1],\"scale\":1|2,\"resolve\":\"neyi çözüyorsun\"} | "
            "{\"action\":\"profile\",\"box\":[x0,y0,x1,y1],\"axis\":\"x\"|\"y\",\"resolve\":\"...\"} | {\"action\":\"done\"}"})
        text, _ = ask(msgs, 300); msgs.append({"role": "assistant", "content": text})
        act = QA.action_of(text); k = act.get("action")
        if k == "focus" and left["focus"] > 0:
            left["focus"] -= 1
            try:
                bx = [max(0, int(act["box"][0])), max(0, int(act["box"][1])), min(W, int(act["box"][2])), min(H, int(act["box"][3]))]
                sc = 2 if int(act.get("scale", 1)) >= 2 else 1
                bx[2] = min(bx[2], bx[0] + 1120 // sc); bx[3] = min(bx[3], bx[1] + 1000 // sc)
                v = QA.ruled_view(crop, bx, sc, hint); v.save(out / f"{name}_2_focus{2 - left['focus']}.png")
                msgs.append(img_msg(v, f"focus {bx} scale {sc} ({act.get('resolve', '')})."))
                log["measure"].append({"focus": bx, "scale": sc, "resolve": act.get("resolve")})
            except Exception as e:
                msgs.append({"role": "user", "content": f"Geçersiz kutu: {e}"})
        elif k == "profile" and left["profile"] > 0:
            left["profile"] -= 1
            try:
                pk = QA.profile(crop, act["box"], act.get("axis", "x"))
                msgs.append({"role": "user", "content": f"profile {act['box']} axis {act.get('axis')}: {json.dumps(pk)}"})
                log["measure"].append({"profile": act["box"], "axis": act.get("axis"), "result": pk, "resolve": act.get("resolve")})
            except Exception as e:
                msgs.append({"role": "user", "content": f"Geçersiz profil: {e}"})
        else:
            break
    # 3: draft, tools withdrawn
    msgs.append({"role": "user", "content":
        "ADIM 3 (taslak, artık ölçüm aracı yok): Odanın iç sınır köşelerini saat yönünde, sırayla, ilk görüntüdeki cetvel "
        "koordinatlarıyla yaz. Her yön değişimi bir köşe; kolon omuzlarını ayrı köşe olarak ver; kapıyı düz geç. "
        "Ölçtüğün çizgi konumlarını kullan. JSON: {\"points\": [[x,y], ...]} ya da emin olamıyorsan "
        "{\"unresolved\": \"sorunlu kenar\"}"})
    text, tm = ask(msgs, 900); msgs.append({"role": "assistant", "content": text})
    act = QA.action_of(text); draft = as_points(act.get("points") or [])
    log["draft"] = draft if draft else {"unresolved": act.get("unresolved", text[-300:])}
    res = {"target": target, "loop": log["loop"], "measure": log["measure"]}
    res["draft_score"] = QT.brief(QT.score(draft, ref, hint)) if len(draft) >= 3 else {"valid": False, "reason": "no draft"}
    final = draft
    # 4: one correction
    if len(draft) >= 3:
        dv = draft_view(crop, hint, draft); dv.save(out / f"{name}_3_draft.png")
        msgs.append(img_msg(dv, "ADIM 4 (bir kez düzelt): İnce turuncu çizgi senin taslağın, numaralı pembe halkalar köşeler. "
                                "Çizgi her yerde duvarın ODAYA BAKAN iç yüzünde mi? Kolon omzu düz geçilmiş mi, ölçü/aks çizgisine "
                                "kaymış mı, komşu odaya taşmış mı, kapı düz geçilmiş mi? Yalnız şu JSON'lardan biri: "
                                "{\"decision\":\"keep\"} | {\"decision\":\"replace_vertex\",\"changes\":[{\"i\":k,\"xy\":[x,y]}, ...]} | "
                                "{\"decision\":\"replace_all\",\"points\":[[x,y],...]} | {\"decision\":\"unresolved\",\"edge\":\"...\"}"))
        text, _ = ask(msgs, 900)
        act = QA.action_of(text); log["correction"] = act or text[-300:]
        dec = act.get("decision")
        if dec == "replace_vertex":
            final = [list(p) for p in draft]
            for c in act.get("changes", []):
                try:
                    i = int(c["i"]); final[i] = [float(c["xy"][0]), float(c["xy"][1])]
                except Exception:
                    pass
        elif dec == "replace_all":
            final = as_points(act.get("points") or []) or draft
        res["correction"] = act
        res["final_score"] = QT.brief(QT.score(final, ref, hint)) if len(final) >= 3 else {"valid": False}
        draft_view(crop, hint, final).save(out / f"{name}_4_final.png")
    res.update({"draft": draft, "final": final, "seconds": round(time.time() - t0, 1)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--cases", default="")
    ap.add_argument("--out", default=str(Path(__file__).with_name("runs") / "qwen_agent2_a"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want = set(a.cases.split(",")) if a.cases else None
    allr = {}
    for name, crop, target, hint, ref in QT.cases():
        if want and name not in want:
            continue
        r = run_case(name, crop, target, hint, ref, a, out)
        allr[name] = r
        print(name, "draft", r["draft_score"], "| final", r.get("final_score"), "|", r["seconds"], "s", flush=True)
        json.dump(allr, open(out / "RESULT.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
