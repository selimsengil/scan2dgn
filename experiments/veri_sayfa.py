"""Builds kat_test/egitim_veri/EGITIM_VERISI.html: what is available for training the piece classifier (one file, pictures inside)."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import base64, io, json, sys
from collections import Counter
from pathlib import Path
from PIL import Image

import v0_watershed as V0  # noqa: E402

OUT = config.KT / "egitim_veri"; ROOT = V0.B.parent
scope = json.load(open(ROOT / "USER_LABEL_SCOPE_20261024.json", encoding="utf-8"))
stats = json.load(open(OUT / "STATS.json", encoding="utf-8"))
ROLE = {"001": "eğitim", "002": "eğitim", "010": "eğitim", "013": "eğitim", "016": "eğitim", "011": "eğitim (yalnız yapı)", "012": "eğitim (yalnız yapı)",
        "019": "geliştirme testi (oda sınırı)", "017": "ek test (oda sınırı)", "007": "ek test (oda sınırı)", "015": "kör son test, dokunulmuyor"}


def b64(name, width=1500):
    im = Image.open(OUT / name); s = width / im.width
    if s < 1: im = im.resize((width, int(im.height * s)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=78); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


rows = []; tot = Counter()
for d in sorted(p for p in V0.B.iterdir() if p.is_dir()):
    L = json.load(open(d / "LABELS_COMMON.json", encoding="utf-8")); c = Counter(s["kind"] for s in L["structures"]); f = d.name
    sc = "tam onaylı" if f in scope["full"] else "yalnız yapı onaylı" if f in scope["structure_only"] else "yalnız oda sınırı onaylı" if f in scope["boundary_only"] else "kapsam dışı"
    rows.append((f, L["project"].replace("Balikesir_", "").replace("%29", ")").replace("28", "(", 1) if "%29" in L["project"] else L["project"].replace("Balikesir_", ""), len(L["rooms"]), c["wall"], c["column"], c["door"], c["window"], sc, ROLE.get(f, "kullanılmıyor")))
    for k in ("wall", "column", "door", "window"): tot[k] += c[k]
    tot["room"] += len(L["rooms"])
tr = "".join(f"<tr class='{'use' if r[8] != 'kullanılmıyor' else 'off'}'><td>{r[0]}</td><td>{r[1]}</td><td>{r[2] or '–'}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td><td>{r[7]}</td><td>{r[8]}</td></tr>" for r in rows)
NAME = {"oda": "oda zemini", "duvar": "duvar", "kolon": "kolon/beton", "kapi": "kapı açıklığı", "pencere": "pencere", "bilinmiyor": "etiketsiz (eğitime girmez)"}
ps = "".join("<tr><td>%s</td>%s</tr>" % (f, "".join(f"<td>{stats[f]['count'].get(k, 0):,}</td>".replace(",", ".") for k in NAME) + f"<td>{stats[f]['pieces']:,}</td>".replace(",", ".")) for f in stats)


def viewer(f, title):
    return f"""<section><h3>{title}</h3><div class="tabs" data-f="{f}"><button class="on" data-v="kaynak">Tarama</button><button data-v="etiket">Etiketler</button><button data-v="parca">Parçalar (eğitim hedefi)</button></div>
<div class="pic"><img id="img{f}" src="{b64(f + '_kaynak.jpg')}" data-kaynak="" alt="kat {f}"></div></section>
<script>(function(){{const S={{kaynak:"{b64(f + '_kaynak.jpg')}",etiket:"{b64(f + '_etiket.jpg')}",parca:"{b64(f + '_parca.jpg')}"}};const im=document.getElementById('img{f}');
document.querySelectorAll('.tabs[data-f="{f}"] button').forEach(b=>b.onclick=()=>{{im.src=S[b.dataset.v];b.parentNode.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));}});}})();</script>"""


html = f"""<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Eğitim verisi</title><style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#17212d;--mut:#5b6877;--line:#d7dee6;--acc:#0758a0}}
body{{font:16px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:16px}}main{{max-width:1100px;margin:auto}}
h1{{font-size:26px;margin:8px 0}}h2{{font-size:20px;margin:28px 0 8px;border-bottom:2px solid var(--line);padding-bottom:4px}}h3{{font-size:17px;margin:14px 0 6px}}
section,.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}}p{{margin:6px 0}}.mut{{color:var(--mut)}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}}th{{background:#eef2f6}}
tr.off td{{color:#9aa5b1}}.pic{{overflow:auto;max-height:80vh;border:1px solid var(--line)}}.pic img{{width:100%;min-width:900px;display:block}}
.tabs button{{font:inherit;padding:6px 12px;margin:0 4px 8px 0;border:1px solid var(--line);border-radius:8px;background:#eef2f6;cursor:pointer}}.tabs button.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}
.leg span{{display:inline-block;margin:2px 12px 2px 0}}.leg i{{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:5px;vertical-align:-2px}}
.kpi{{display:flex;flex-wrap:wrap;gap:10px}}.kpi div{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:120px}}.kpi b{{display:block;font-size:22px}}
ol,ul{{margin:6px 0 6px 20px;padding:0}}li{{margin:3px 0}}</style><main>
<h1>Eğitimde kullanılabilecek veri</h1><p class="mut">Henüz hiçbir eğitim başlamadı. Bu sayfa elde ne olduğunu ve sınıflandırıcının neyi öğreneceğini gösterir.</p>
<div class="kpi"><div><b>19</b>etiketli kat</div><div><b>{tot['room']}</b>senin çizdiğin oda sınırı (9 kat)</div><div><b>{tot['wall']}</b>duvar</div><div><b>{tot['column']}</b>kolon/beton</div><div><b>{tot['door']}</b>kapı açıklığı</div><div><b>{tot['window']}</b>pencere</div></div>

<h2>1. Katlar ve hangi iş için kullanılacakları</h2>
<div class="card"><p>Oda sınırları senin DGN çizimlerinden geliyor. Duvar, kolon, kapı ve pencere çizimleri taramadan elle izlenmiş taslaklar; hangilerinin kullanılabileceğini 24 Ekim'de sen belirlemiştin (“kapsam” sütunu). Gri satırlar o kapsamın dışında, kullanılmıyor.</p>
<div class="scroll"><table><tr><th>Kat</th><th>Proje</th><th>Oda</th><th>Duvar</th><th>Kolon</th><th>Kapı</th><th>Pencere</th><th>Senin kapsamın</th><th>Önerilen rol</th></tr>{tr}</table></div></div>

<h2>2. Veri neye benziyor</h2>
<p class="leg"><span><i style="background:#3cb44b"></i>oda zemini</span><span><i style="background:#006edc"></i>duvar</span><span><i style="background:#f5821e"></i>kolon/beton</span><span><i style="background:#e11e3c"></i>kapı açıklığı</span><span><i style="background:#00bec8"></i>pencere</span><span><i style="background:#969696"></i>etiketsiz</span></p>
<p>Üç görünüm arasında geç: <b>Tarama</b> modelin gördüğü resim, <b>Etiketler</b> elle çizilmiş doğrular, <b>Parçalar</b> kodlamanın çıkardığı parçaların her birine o etiketlerden düşen sınıf. Sınıflandırıcı tam olarak bu son görünümü öğrenir.</p>
{viewer('013', 'Kat 013 (Marmara Türkeli) · bir bölümü · tam onaylı')}
{viewer('019', 'Kat 019 (Susurluk Kışla 85) · tamamı · geliştirme testi')}
{viewer('016', 'Kat 016 (Susurluk Kışla 83, bodrum) · tamamı · tam onaylı')}

<h2>3. Parça sayıları (gösterilen bölümlerde)</h2>
<div class="card"><div class="scroll"><table><tr><th>Kat</th>{''.join(f'<th>{v}</th>' for v in NAME.values())}<th>toplam parça</th></tr>{ps}</table></div>
<p class="mut">Tek bir katın bir bölümü bile binlerce etiketli parça veriyor. Yedi eğitim katının tamamında bu sayı yüz binleri bulur.</p></div>

<h2>4. Eğitim nasıl yapılacak</h2>
<div class="card"><p><b>Yöntem: etiketli (denetimli) eğitim, parça düzeyinde.</b> Etiketsiz öğrenmeye ilk aşamada gerek yok, çünkü her parçanın doğru sınıfı elimizde.</p>
<ol><li><b>Girdi:</b> kodlamanın çıkardığı her parça için şekli (alan, en, boy, incelik), kenarlarındaki çizgilerin koyuluğu ve kalınlığı, içindeki mürekkep oranı, komşu parçaların aynı özellikleri ve parçanın çevresinden küçük bir resim kesiti.</li>
<li><b>Hedef:</b> altı sınıftan biri: oda zemini, duvar, kolon, kapı açıklığı, pencere, hiçbiri. Etiketsiz parçalar “yok” sayılmaz, eğitime hiç girmez.</li>
<li><b>Model:</b> önce karar ağaçları topluluğu (hızlı, az veriyle iyi çalışır, saniyeler içinde eğitilir). Yetmezse resim kesitini okuyan küçük bir sinir ağı eklenir. İkisi de yerelde, ekran kartsız çalışır.</li>
<li><b>Odaya dönüş:</b> oda adının durduğu noktadan başlanır; duvar, kolon, kapı ve pencere parçalarına gelince durulur. Sınır, parçaların kendi çizgisidir.</li>
<li><b>Ölçüm:</b> oda sınırında bugüne kadar kullandığımız çizgi tutma skoru (019, sonra 017 ve 007), ayrıca kapı ve pencere için bulunan/kaçan sayısı. 015 en sonda bir kez.</li></ol>
<p><b>Genelleme sınaması:</b> 015, 016, 017 ve 019 aynı çizim ailesinden (Kışla). 016 eğitimdeyken 019'da iyi çıkmak “yeni bir büroda da çalışır” demek değildir. Bu yüzden iki ölçüm yapılır: Kışla ailesi eğitimde varken ve tamamen çıkarılmışken.</p>
<p><b>Etiketsiz veri ikinci aşamada:</b> genelleme zayıf çıkarsa, etiketlenmemiş planlarda modelin emin olduğu parçalar eğitime eklenir ve senin tıklayarak düzelttiğin parçalar yeni etiket olur.</p></div>

<h2>5. Kapı ve pencere ayrı bir iş mi?</h2>
<div class="card"><p><b>Yerlerini bulmak ayrı bir iş değil.</b> Kapı açıklığı ve pencere bu veri setinde zaten etiketli ({tot['door']} kapı, {tot['window']} pencere) ve aynı sınıflandırıcının iki sınıfı. Odayı kapatmak için de gerekliler: kapı açıklığı parçası odanın bittiği yerdir.</p>
<p><b>Ölçüler iki yoldan gelir:</b> genişlik, bulunan açıklığın boyundan ve planın ölçeğinden hesaplanır. Yazılı etiket (K3-90/210, P2-150/140) ise bir okuma işidir; açıklığın yanındaki yazı okunup ona bağlanır. Bu okuma adımı için eğitim gerekmiyor, ama ayrı yazılıp ayrı ölçülmesi gerekiyor.</p></div>

<h2>6. Bilinen eksikler</h2>
<div class="card"><ul><li>Duvar, kolon, kapı ve pencere çizimlerinin hiçbiri “insan onaylı” olarak işaretli değil; taslak olarak duruyorlar. Senin kapsam kararın dışındaki katları kullanmıyorum.</li>
<li>Balkon, merdiven, asansör gibi oda olmayan alanlar etiketsiz. Model bunları “oda zemini değil” diye öğrenemez; oda adı noktası olmayan alanlar zaten oda sayılmayacağı için ilk aşamada sorun beklemiyorum, ama ölçmeden emin değilim.</li>
<li>Tam onaylı kat sayısı az (altı, biri kör test). Beş ayrı proje var; yeni bir büro stilinde ne olacağını ancak aile dışı sınama gösterir.</li>
<li>Kodlamanın kendi tavanı 019'da çizgi tutmada 0,987; 35 odanın 25'i kabul sınırını geçiyor. Sınıflandırma kusursuz olsa bile kalan odalar için kodlayıcıda düzeltme gerekecek.</li></ul></div>
</main></html>"""
(OUT / "EGITIM_VERISI.html").write_text(html, encoding="utf-8"); print(OUT / "EGITIM_VERISI.html", round(len(html) / 1e6, 2), "MB")
