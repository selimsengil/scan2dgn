"""Training-data registry: which project/floor is used how by the boundary networks.
Writes kat_test/EGITIM_KAYDI.json + EGITIM_KAYDI.md and copies the md to cizim_projeler.
Roles: egitim = in the DGN training set (cizgi_ag c1 / bolut_dgn, --skip Kisla);
       test = measured with the line net; kor = blind floor (never extracted); kullanilmadi = extracted, unused.
usage: python egitim_kaydi.py"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

H = config.KT
TEST = {"Susurluk_Kisla_143_85__1_54883", "Susurluk_Kisla_143_84__1_55569", "Susurluk_Kisla_143_85__Z_55052", "Susurluk_Kisla_143_84__Z_55390"}
EL = {"Gonen_Malkoc_361_79__K2_56155": "kat_test/el_etiket/malkoc_k2/ETIKET.json (21.11, Claude elle: oda, kapi/pencere + etiket metni, bolum, balkon, kat alani)"}
OLD = {"001": "Gonen Gundogdu 771 11 / 1. kat", "002": "Gonen Gundogdu 771 11 / Zemin", "007": "Gonen Plevne 228 28 / 1. kat",
       "010": "Havran Ebubekir 271 172 / Bodrum", "013": "Marmara Turkeli 0 8480 / 1. kat", "016": "Susurluk Kisla 143 83 / Bodrum",
       "017": "Susurluk Kisla 143 84 / 1. kat", "019": "Susurluk Kisla 143 85 / 1. kat"}


def main():
    idx = json.load(open(H / "dgn_katlar" / "INDEX.json", encoding="utf-8"))
    aud = json.load(open(H / "gt_denetim" / "DENETIM.json", encoding="utf-8"))
    vo = H / "aciklik_ag" / "VERI_OZET.json"; ac = json.load(open(vo, encoding="utf-8")) if vo.exists() else {}
    rows = []
    for f in idx:
        fid = f["floor"]; rooms = f.get("room", 0); a = aud.get(fid, {}).get("rooms", [])
        role = "el_test" if fid in EL else "kullanilmadi" if rooms == 0 else ("test" if fid in TEST else ("kullanilmadi" if "kisla" in fid.lower() else "egitim"))
        rows.append({"kat": fid, "proje": fid.split("__")[0], "kat_no": f.get("kat_no"), "rol": role, "oda": rooms,
                     "temiz_oda": sum(r["oturma"] >= 0.95 for r in a), "kapi": f.get("door", 0), "pencere": f.get("window", 0),
                     "balkon": f.get("balcony", 0), "px_m": round(f.get("px_per_m", 0), 1),
                     "aciklik_ag": ac.get(fid), "el_etiket": EL.get(fid)})
    rows.append({"kat": "Susurluk_Kisla_143_83__1 (eski 015)", "proje": "Susurluk_Kisla_143_83", "kat_no": "1", "rol": "kor",
                 "not": "kör test katı; dgn_katlar'a hiç yazılmadı (dgn_kat.py --blind olmadan atlanır)"})
    out = {"tarih": str(date.today()), "kaynak": "kat_test/dgn_katlar (dgn_kat.py), kaynak DGN kopyaları kat_test/dgn/<proje>/source",
           "temiz_esik": "gt_denetim oturma >= 0.95 (oda sınırının taramadaki çizgiye 2.5 cm içinde oturan payı)",
           "katlar": rows,
           "eski_parca_verisi": {"yer": "kat_test/parca_veri (001..019 numaralı eski set; özgün sam_etiket_hassas_v3 klasörü artık yok)", "katlar": OLD}}
    (H / "EGITIM_KAYDI.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    by = defaultdict(list)
    for r in rows: by[r["proje"]].append(r)
    L = [f"# Eğitim verisi kaydı ({out['tarih']})", "", "Roller: **egitim** = sınır ağlarının eğitim setinde (cizgi_ag c1, bolut_dgn; Kışla dışarıda), "
         "**test** = ölçüm katı, **el_test** = Claude'un elle tam etiketlediği test katı (eğitimde KULLANILMAZ; aynı planın K1/K3 tekrarları eğitimde, sızıntıya dikkat), **kor** = kör test, **kullanilmadi** = çıkarıldı ama kullanılmıyor. "
         "`temiz` = DGN sınırı taramadaki çizgiye oturan oda (oturma ≥ 0.95); kirli odaların sınırı eğitimde yok sayılır.", "",
         "Açıklık ağı (aciklik_ag a1, 21.11): aynı roller. `güv. kapı/pencere` = kasası taramada oturan ya da motorla oturtulan DGN açıklığı; "
         "kalanlar eğitimde yok sayılır. Oda girdisi = temiz ya da `oturt` ile temizlenen odalar (kat_test/aciklik_ag/veri).", "",
         "| Proje | Kat | Rol | Oda (temiz) | Kapı | Pencere | güv. kapı | güv. pencere | px/m |", "|---|---|---|---|---|---|---|---|---|"]
    tot = defaultdict(int)
    for p in sorted(by):
        for r in by[p]:
            if r["rol"] == "kor": L.append(f"| {p} | 1 | kor | – | – | – | – | – | – |"); continue
            q = r.get("aciklik_ag") or {}
            L.append(f"| {p} | {r['kat_no']} | {r['rol']} | {r['oda']} ({r['temiz_oda']}) | {r['kapi']} | {r['pencere']} | {q.get('guvenilir_kapi', '–')} | {q.get('guvenilir_pencere', '–')} | {r['px_m']} |")
            tot[r["rol"] + "_gkapi"] += q.get("guvenilir_kapi", 0); tot[r["rol"] + "_gpen"] += q.get("guvenilir_pencere", 0)
            tot[r["rol"] + "_kat"] += 1; tot[r["rol"] + "_oda"] += r["oda"]; tot[r["rol"] + "_temiz"] += r["temiz_oda"]
    L += ["", f"Eğitim: {tot['egitim_kat']} kat, {tot['egitim_oda']} oda ({tot['egitim_temiz']} temiz). "
          f"Açıklık ağı eğitim: {tot['egitim_gkapi']} kapı, {tot['egitim_gpen']} pencere (güvenilir). "
          f"Test: {tot['test_kat']} kat, {tot['test_oda']} oda. Kullanılmayan: {tot['kullanilmadi_kat']} kat.", "",
          "Not: üst normal katlar çoğunlukla aynı planın tekrarıdır (benzersiz plan ~50). Tekrar eden katlarda DGN daha çok kaymış (ör. Malkoç Z %100, K2 %32 temiz).",
          "Marmara Türkeli satırları ofisin eski çizimidir (15.11 çıkarımı); 20.11'de asistanın çizdiği Marmara katları henüz veri değildir.", "",
          "Eski parça seti (kat_test/parca_veri): " + ", ".join(f"{k} = {v}" for k, v in OLD.items()) + "."]
    md = "\n".join(L) + "\n"; (H / "EGITIM_KAYDI.md").write_text(md, encoding="utf-8")
    shutil.copy2(H / "EGITIM_KAYDI.md", Path(r"C:\Users\Selim\Documents\cizim_projeler\EGITIM_KAYDI.md"))
    print(md[-700:])


if __name__ == "__main__":
    main()
