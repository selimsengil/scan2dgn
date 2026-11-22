"""Citymax çizim motoru v2 (2026-11-19).

Tek koordinat çerçevesi: bağlı TIF'in özgün piksel merkezleri (x sağa, y aşağı). Her kat bir raster kopyasına
(1 tabanlı raster indeksi) yazılır; dünya koordinatı = o rasterin kayıtlı dönüşümü. Görüntü ölçeği/kenar
boşluğu dönüşümü yoktur: cetvelde okunan sayı doğrudan komuta yazılır.

Zekâ gerektirmeyen her şey burada yapılır:
  - kenarları mürekkep yüzüne oturtma (önerir ve raporlar, sınır ±R px; belirsizse oynatmaz)
  - eksene yakın kenarları tam dik yapma, köşeleri komşu kenar doğrularının kesişiminden hesaplama
  - kapı/pencere kutusu: iki kasa noktası verilir, iki duvar yüzü çizili odalardan/kat alanından bulunur
  - F4 / oda / açıklık etiketlerinin konumu, kapı/pencere yüksekliğinin etiket yazısından çıkarılması
  - merdiven (F6) köşelerinin oda sınırına yakalanması
  - F11 uçlarının kat alanı dışına uzatılması, F12/Ortak alan tıklama noktalarının hesabı
  - balkon sınırı uçlarının kat alanına yakalanması, F10 native komutu
  - kat kopyalama (native seçim + COPY/MOVE), Kat Kontrol / Yükseltme / Birleştirme kopyada, log çözümü
  - topoloji denetimi (örtüşme, taşma, iki yüz kesme, kapı-pencere çakışması, F11 oda içinden geçmesin ...)
Mimari karar (hangi çizgi oda sınırı, oda adı, BB aidiyeti) planı yazan modelindir.
"""
from __future__ import annotations

import json, math, re, shutil, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point, Polygon, box as sbox
from shapely.ops import nearest_points, polygonize, unary_union, polylabel

from .native import RasterMap, app, save, snapshot
from .native_batch_writer import _set_object_property, _tag_state
from .native_schema import TAGS

Image.MAX_IMAGE_PIXELS = None

LEVEL = {'frame': ('KatPlaniSiniri', 23), 'axis': ('BinaAksi', 1), 'floor': ('KatAlani', 85), 'building': ('BinaAlani', 2),
         'room': ('OdaAlani', 9), 'door': ('KapiAcikligi', 6), 'window': ('PencereAcikligi', 6), 'stair': ('OdaIcYapi', 183),
         'balcony_boundary': ('BalkonSiniri', 4), 'bb_common': ('BagimsizBolumOrtakAlani', 150)}
SCHEMA = {**TAGS, 'stair': ('Oda İç Yapı Bilgisi', [('Oda İç Yapı Adı', 1, 'OdaIcYapiAdi', 134)]),
          'building': ('Bina Bilgisi', [('Bina Çatı Kotu', 3, 'BinaCatiKotu', 128), ('Asansör Sayısı', 3, 'AsansorSayisi', 129),
                                        ('Proje Onay Tarihi', 1, 'ProjeOnayTarihi', 130), ('MAKS Kimlik No', 1, 'MAKSKimlikNo', 138)])}
HEIGHT = {'KatNo': .35, 'KatTabanKotu': .25, 'KatTavanKotu': .25, 'OdaAdi': .22, 'OdaM2': .19, 'KapiYuksekligi': .16,
          'PencereYuksekligi': .16, 'ZemindenYuksekligi': .16, 'OdaIcYapiAdi': .2, 'BinaCatiKotu': .3, 'AsansorSayisi': .3,
          'ProjeOnayTarihi': .3, 'MAKSKimlikNo': .3}
ODA_ADLARI = ('ODA', 'BANYO', 'SALON', 'MUTFAK', 'KİLER', 'TUVALET', 'BALKON', 'HOL', 'DÜKKAN', 'SIĞINAK', 'DANIŞMA',
              'KALORİFER DAİRESİ', 'KAPICI DAİRESİ', 'ELEKTRİK MERKEZİ', 'ISI MERKEZİ', 'OTOPARK', 'GÜVENLİK ODASI', 'HAVUZ',
              'SOSYAL TESİS', 'SPOR SALONU', 'ÇÖP ODASI', 'N.O.A', 'MERDİVEN', 'SU MERKEZİ', 'DİNİ TESİS', 'MÜŞTEMİLAT', 'TEKNİK HACİM')
# Citymax 6.6 Oda Adı listesi (kullanıcı ekranı, 2026-11-19). Paftadaki ad kılavuz tablosuyla bunlardan birine çevrilir
# (ör. Mağaza/Ofis/Büro -> DÜKKAN, Yatak odası -> ODA, Wc -> TUVALET, Asansör/Merdiven holü -> N.O.A).
CODES = {'bagimsizbolumsiniri': 111, 'bagimsizbolumalani': 104, 'ortakalan': 113, 'balkonduvari': 107}


def rd(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


# ----------------------------------------------------------------------------- görüntü
class Kaynak:
    """Bağlı TIF; bir kez açılır, gri tonlu numpy olarak bellekte tutulur."""
    _cache = {}

    def __init__(self, tif):
        tif = str(tif)
        if tif not in Kaynak._cache:
            Kaynak._cache[tif] = np.asarray(Image.open(tif).convert('L'))
        self.g = Kaynak._cache[tif]
        self.ink = None

    def crop(self, b):
        x0, y0, x1, y1 = map(int, b)
        return self.g[max(0, y0):y1, max(0, x0):x1]

    def inkmap(self):
        if self.ink is None: self.ink = self.g < 128
        return self.ink


def gorunum(kaynak, b, scale=1.0, overlays=(), out=None, grid=100):
    """b = [x0,y0,x1,y1] TIF pikseli. Cetvel değerleri TIF pikselidir; overlay: [(points, kapalı, renk, etiket)]."""
    x0, y0, x1, y1 = map(int, b); g = kaynak.crop(b)
    im = Image.fromarray(g).convert('RGB')
    if scale != 1: im = im.resize((round(im.width * scale), round(im.height * scale)), Image.NEAREST if scale > 1 else Image.LANCZOS)
    M = 44; pan = Image.new('RGB', (im.width + 2 * M, im.height + 2 * M), '#eef1f5'); pan.paste(im, (M, M)); d = ImageDraw.Draw(pan)
    try: f = ImageFont.truetype('arial.ttf', 12)
    except OSError: f = ImageFont.load_default()
    T = lambda p: (M + (p[0] - x0) * scale, M + (p[1] - y0) * scale)  # piksel j = [j, j+1)
    step = grid if scale >= .5 else grid * 5
    for v in range(math.ceil(x0 / step) * step, x1, step):
        X = T((v, y0))[0]; d.line((X, M - 6, X, M), fill='#222'); d.line((X, M + im.height, X, M + im.height + 6), fill='#222')
        d.text((X - 14, 6), str(v), font=f, fill='#123'); d.text((X - 14, M + im.height + 10), str(v), font=f, fill='#123')
    for v in range(math.ceil(y0 / step) * step, y1, step):
        Y = T((x0, v))[1]; d.line((M - 6, Y, M, Y), fill='#222'); d.line((M + im.width, Y, M + im.width + 6, Y), fill='#222')
        d.text((2, Y - 7), str(v), font=f, fill='#123'); d.text((M + im.width + 8, Y - 7), str(v), font=f, fill='#123')
    for pts, closed, col, lab in overlays:
        q = [T(p) for p in pts] + ([T(pts[0])] if closed and pts else [])
        if len(q) > 1: d.line(q, fill=col, width=1 if scale < 2 else 2)
        if lab and pts:
            c = Polygon(pts).representative_point() if closed and len(pts) > 2 else Point(pts[0])
            d.text(T((c.x, c.y)), str(lab), font=f, fill=col)
    if out: pan.save(out)
    return pan


def pano(kaynak, boxes, scale=2.0, overlays=(), out=None, cols=4, grid=25):
    """Birçok küçük ayrıntıyı (köşe, kolon, kasa) tek resimde, her biri kendi TIF cetveliyle gösterir."""
    tiles = []
    for b in boxes:
        ov = [(pts, c, col, lab) for pts, c, col, lab in overlays]
        t = gorunum(kaynak, b, scale, ov, None, grid); d = ImageDraw.Draw(t); d.text((48, t.height - 16), f'{int(b[0])},{int(b[1])}', fill='#a00')
        tiles.append(t)
    if not tiles: return None
    cw = max(t.width for t in tiles); ch = max(t.height for t in tiles); rows = math.ceil(len(tiles) / cols)
    sheet = Image.new('RGB', (cw * min(cols, len(tiles)), ch * rows), 'white')
    for k, t in enumerate(tiles): sheet.paste(t, ((k % cols) * cw, (k // cols) * ch))
    if out: sheet.save(out)
    return sheet


def aks_temizle(ink, pafta, bina, egim=None, kalin=5, oran=.7):
    """Aks (grid) çizgilerini siler: bina kutusunun iki yanındaki boş pafta şeritlerinde (üst/alt ya da sol/sağ)
    kesintisiz uzanan ince çizgiler akstır. Yalnız ince (<= kalin px) kısımları silinir; duvar/kolon üstündeki
    kalın mürekkep korunur. pafta, bina: [x0,y0,x1,y1]. Dönüş: (temiz ink, silinen aks listesi)."""
    H_ = np.array((egim or {}).get('yatay', [1.0, 0.0])); V_ = np.array((egim or {}).get('dikey', [0.0, 1.0]))
    out = ink.copy(); found = []
    px0, py0, px1, py1 = map(int, pafta); bx0, by0, bx1, by1 = map(int, bina)
    def line_pts(c, vert, a0, a1):
        t = np.arange(a0, a1)
        if vert: return np.floor(c + (t - by0) * V_[0] / V_[1]).astype(int), t
        return t, np.floor(c + (t - bx0) * H_[1] / H_[0]).astype(int)
    for vert in (True, False):
        lo, hi = (bx0 - 300, bx1 + 300) if vert else (by0 - 300, by1 + 300)
        for c in range(lo, hi):
            bands = [(max(py0, by0 - 340), by0 - 40), (by1 + 40, min(py1, by1 + 340))] if vert else [(max(px0, bx0 - 340), bx0 - 40), (bx1 + 40, min(px1, bx1 + 340))]
            fr = []
            for a0, a1 in bands:
                if a1 - a0 < 100: fr.append(0); continue
                xs, ys = line_pts(c, vert, a0, a1); fr.append(ink[ys, xs].mean())
            if min(fr) < oran: continue
            found.append(('dikey' if vert else 'yatay', c))
    # bitişik konumları tek aksa indir, ince kısmı sil
    merged = []
    for kind, c in found:
        if merged and merged[-1][0] == kind and c - merged[-1][2] <= 1: merged[-1][2] = c
        else: merged.append([kind, c, c])
    for kind, c0, c1 in merged:
        vert = kind == 'dikey'; a0, a1 = (py0, py1) if vert else (px0, px1)
        for c in range(c0 - 1, c1 + 2):
            xs_, ys_ = line_pts(c, vert, a0, a1)
            if vert:
                wide = sum(ink[ys_, np.clip(xs_ + d, 0, ink.shape[1] - 1)] for d in range(-kalin, kalin + 1))
            else:
                wide = sum(ink[np.clip(ys_ + d, 0, ink.shape[0] - 1), xs_] for d in range(-kalin, kalin + 1))
            thin = wide <= kalin
            out[ys_[thin], xs_[thin]] = False
    return out, [(k, (a + b) / 2) for k, a, b in merged]


def kolon_bul(ink, box, egim=None, min_px=18, max_px=160):
    """Taralı (hatch) dikdörtgen kolonları bulur. Dönüş: [{'kutu': 4 köşe (TIF px), 'gen_px', 'boy_px'}].
    Yoğunluk eşiği + morfolojik açma ince çizgileri ve yazıyı eler; kenarlar oturt(dis=True) ile dış çizgiye oturur."""
    import cv2
    x0, y0, x1, y1 = map(int, box); sub = ink[y0:y1, x0:x1].astype(np.float32)
    # 45° tarama süzgeci: yalnız çapraz çizgi parçaları kalır (duvar/aks yatay-dikey, yazıda uzun çapraz yok)
    u8 = sub.astype(np.uint8); K = np.fliplr(np.eye(9, dtype=np.uint8)); K2 = np.eye(9, dtype=np.uint8)
    hatch = np.maximum(cv2.morphologyEx(u8, cv2.MORPH_OPEN, K), cv2.morphologyEx(u8, cv2.MORPH_OPEN, K2)).astype(np.float32)
    d = cv2.blur(hatch, (21, 21)); m = (d > .04).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); out = []
    def side_cov(rect):
        cov = []
        for i in range(4):
            a, b = np.array(rect[i], float), np.array(rect[(i + 1) % 4], float); L = np.linalg.norm(b - a)
            if L < 5: return 0
            u = (b - a) / L; nn = np.array([-u[1], u[0]]); c = (a + b) / 2
            inward = nn if Polygon(rect).contains(Point(*(c + 3 * nn))) else -nn; best = 0
            for off in (-1, 0, 1, 2, 3):
                ts = np.arange(.1 * L, .9 * L); P = a[None] + ts[:, None] * u[None] + off * inward[None]
                xi = np.clip(np.floor(P[:, 0]).astype(int), 0, ink.shape[1] - 1); yi = np.clip(np.floor(P[:, 1]).astype(int), 0, ink.shape[0] - 1)
                best = max(best, float(ink[yi, xi].mean()))
            cov.append(best)
        return min(cov)
    for k in range(1, n):
        bx, by, bw, bh, area = st[k]
        if not (min_px <= bw <= max_px and min_px <= bh <= max_px) or max(bw, bh) / min(bw, bh) > 4.5: continue
        if area / (bw * bh) < .8: continue
        X0, Y0, X1, Y1 = x0 + bx - 3, y0 + by - 3, x0 + bx + bw + 3, y0 + by + bh + 3
        blk = ink[Y0:Y1, X0:X1]
        if blk.size == 0: continue
        cols_ = blk.mean(0); rows_ = blk.mean(1)
        def first(v, rev=False):
            idx = range(len(v) - 1, -1, -1) if rev else range(len(v))
            for j in idx:
                if v[j] >= .5: return j
            return None
        l, r_, t_, b_ = first(cols_[:20]), first(cols_[-20:], True), first(rows_[:20]), first(rows_[-20:], True)
        if None in (l, r_, t_, b_): continue
        rect = [[X0 + l, Y0 + t_], [X1 - 20 + r_ + 1, Y0 + t_], [X1 - 20 + r_ + 1, Y1 - 20 + b_ + 1], [X0 + l, Y1 - 20 + b_ + 1]]
        P_ = Polygon(rect)
        if not P_.is_valid or P_.area < min_px * min_px * .8: continue
        cov = side_cov(rect)
        if cov < .6: continue  # dört kenarı da çerçeve çizgisi olmayan (yazı, leke)
        xs = [p[0] for p in rect]; ys = [p[1] for p in rect]
        inner = ink[int(min(ys)) + 5:int(max(ys)) - 5, int(min(xs)) + 5:int(max(xs)) - 5]
        if inner.size == 0 or not (.08 <= inner.mean() <= .6): continue  # içi taralı olmalı
        out.append({'kutu': rect, 'gen_px': round(max(xs) - min(xs), 1), 'boy_px': round(max(ys) - min(ys), 1), 'cerceve': round(cov, 2)})
    return out


def kutu_oturt(ink, box, pay=12):
    """Modelin yaklaşık verdiği kolon kutusunu çerçeve çizgisine oturtur. Her kenarda ±pay içinde, kutunun orta
    %80'i boyunca mürekkebi >= %50 olan satır/sütunlardan verilen kenara EN YAKIN olanı seçilir (dıştan ilk değil);
    böylece hemen yandaki kapı kasası ya da duvar çizgisi yakalanmaz. Bulunamayan kenar verilen değerde kalır."""
    x0, y0, x1, y1 = [float(v) for v in box]; X0, Y0, X1, Y1 = int(x0 - pay), int(y0 - pay), int(x1 + pay), int(y1 + pay)
    blk = ink[Y0:Y1, X0:X1]; h, w = blk.shape
    cy = slice(int(.1 * h + pay * .8), int(.9 * h - pay * .8)); cx = slice(int(.1 * w + pay * .8), int(.9 * w - pay * .8))
    cols_ = blk[cy, :].mean(0); rows_ = blk[:, cx].mean(1); res = [x0, y0, x1, y1]
    def near(prof, base, given, outer_sign):
        best = None
        for j in range(len(prof)):
            if prof[j] < .5: continue
            pos = base + j
            # kenar: dış yüz = çizginin dış ucu; ardışık mürekkep dizisinin dış tarafı
            if outer_sign < 0 and j > 0 and prof[j - 1] >= .5: continue
            if outer_sign > 0 and j + 1 < len(prof) and prof[j + 1] >= .5: continue
            if abs(pos - given) <= pay and (best is None or abs(pos - given) < abs(best - given)): best = pos
        return best
    for idx, prof, base, given, sg in ((0, cols_, X0, x0, -1), (2, cols_, X0, x1, 1), (1, rows_, Y0, y0, -1), (3, rows_, Y0, y1, 1)):
        v = near(prof, base, given, sg)
        if v is not None: res[idx] = v + (1 if sg > 0 else 0)
    return [[res[0], res[1]], [res[2], res[1]], [res[2], res[3]], [res[0], res[3]]]


def kolon_cikar(room, kolonlar, ppm, temas_px=4, min_alan_px=60):
    """Oda poligonundan, sınırına değen/taşan kolonların kesişimini çıkarır (kolon dönüşleri). Odanın tamamen
    içindeki serbest kolonlar ayrı döner (BB'de F6 SÜTUN; ortak alanda çizilmez)."""
    R = Polygon(room); attached, free = [], []
    for c in kolonlar:
        K = Polygon(c['kutu'])
        if not K.is_valid or K.intersection(R).area < min_alan_px: continue
        if R.buffer(-temas_px).contains(K): free.append(c['kutu']); continue
        I = K.intersection(R); bx = I.bounds; depth = min(bx[2] - bx[0], bx[3] - bx[1]) if not I.is_empty else 0
        if depth < 3: continue  # duvar yüzüyle aynı hizada kolon: dönüş yok
        attached.append(c['kutu'])
    if attached:
        G = R.difference(unary_union([Polygon(k) for k in attached]).buffer(.01, join_style=2))
        if G.geom_type == 'MultiPolygon': G = max(G.geoms, key=lambda g: g.area)
        G = G.simplify(.3)
        R = Polygon([[round(x, 2), round(y, 2)] for x, y in list(G.exterior.coords)[:-1]])
    return [list(p) for p in R.exterior.coords][:-1], attached, free


def _duvar_yuzu(hit, arka=70, bosluk=3):
    """Işın boyunca mürekkep dizisinde ilk 'duvar yüzü': mürekkep başlar ve arkasındaki `arka` px içinde ya ikinci
    bir çizgi (duvarın öbür yüzü) ya da sürekli bant (tarama/kolon) vardır. Arkası boş tek ince çizgi (aks, ölçü,
    kapı yayı) atlanır."""
    on = np.flatnonzero(hit)
    if not len(on): return None
    runs = []; st = on[0]; pv = on[0]
    for k in on[1:]:
        if k - pv > 1: runs.append((st, pv)); st = k
        pv = k
    runs.append((st, pv))
    for j, (s0, s1) in enumerate(runs):
        if s1 - s0 + 1 < 2: continue
        if s1 - s0 + 1 >= 10: return float(s0)               # kalın bant: tarama/kolon/dolu duvar
        nxt = [r for r in runs[j + 1:] if r[0] - s1 >= bosluk and r[0] - s0 <= arka]
        if nxt: return float(s0)                               # çift çizgili duvar
    return None


def isin(ink, seed, egim=None, genislik=40, n=21, azami=3000, min_pay=.3):
    """Oda içindeki tohumdan dört yöne paralel ışınlar; her yönde ilk mürekkebe uzaklıkların en büyük tutarlı
    kümesi (>= min_pay ışın, 3 px içinde) iç duvar yüzü kabul edilir. Dönüş: yön -> {uzaklik, pay, nokta}."""
    H_ = np.array((egim or {}).get('yatay', [1.0, 0.0])); V_ = np.array((egim or {}).get('dikey', [0.0, 1.0]))
    s = np.array(seed, float); Hh, Ww = ink.shape; out = {}
    for name, d in (('sag', H_), ('sol', -H_), ('alt', V_), ('ust', -V_)):
        p = np.array([-d[1], d[0]]); dist = []
        for o in np.linspace(-genislik, genislik, n):
            st = s + o * p; t = np.arange(0, azami); P = st[None, :] + t[:, None] * d[None, :]
            xi = np.floor(P[:, 0]).astype(int); yi = np.floor(P[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < Ww) & (yi >= 0) & (yi < Hh); hit = np.zeros(len(t), bool); hit[ok] = ink[yi[ok], xi[ok]]
            dist.append(_duvar_yuzu(hit))
        vals = sorted(v for v in dist if v is not None); best = None
        for i in range(len(vals)):
            grp = [v for v in vals if vals[i] <= v <= vals[i] + 3]
            if len(grp) >= min_pay * n and (best is None or vals[i] > best[0]): best = (vals[i], len(grp), float(np.median(grp)))
        if best: out[name] = {'uzaklik': round(best[2], 1), 'pay': round(best[1] / n, 2), 'nokta': [round(float(x), 1) for x in s + best[2] * d], 'en_yakin': vals[0] if vals else None}
        else: out[name] = {'uzaklik': None, 'pay': 0}
    return out


def profile_face(ink, a, b, inside_normal, R=8, frac=.4):
    """Kenar a-b boyunca, normal yönünde -R..R kaydırılmış örnek doğrularda mürekkep oranı.
    Oda içinden dışa doğru giderken ilk 'kağıt -> mürekkep' geçişi iç duvar yüzüdür.
    Dönüş: (kayma px, güven, profil) ; kayma pozitif = dışa."""
    a = np.array(a, float); b = np.array(b, float); L = np.linalg.norm(b - a)
    if L < 6: return None
    u = (b - a) / L; n = -np.array(inside_normal, float)  # dışa doğru
    ts = np.arange(.12 * L, .88 * L, 1.0)
    H, W = ink.shape; prof = []
    for dd in range(-R, R + 1):
        P = a[None, :] + ts[:, None] * u[None, :] + (dd) * n[None, :]
        xi = np.clip(np.floor(P[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.floor(P[:, 1]).astype(int), 0, H - 1)  # RasterMap: piksel j = [j, j+1)
        prof.append(float(ink[yi, xi].mean()))
    prof = np.array(prof); cands = []
    for i in range(1, len(prof)):
        if prof[i] >= frac and prof[i - 1] < frac:
            cands.append(i - R)
    if not cands: return {'kayma': None, 'guven': 0.0, 'profil': prof.round(2).tolist(), 'not': 'yüz bulunamadı'}
    k = min(cands, key=abs); i = k + R
    contrast = float(prof[min(len(prof) - 1, i + 1)] - prof[max(0, i - 2)])
    # alt-piksel: kağıt->mürekkep sınırı, profilin 0.5'i kestiği yer (örnek merkezleri arası doğrusal)
    p0, p1 = prof[i - 1], prof[i]; t = (.5 - p0) / (p1 - p0) if p1 != p0 else .5
    sub = (i - 1 - R) + min(1.0, max(0.0, t))
    return {'kayma': round(float(sub), 2), 'guven': round(contrast, 2), 'aday': cands, 'profil': prof.round(2).tolist()}


def cizgi_uydur(g, yaklasik, ekseni, bas, son, adim=100, ara=25):
    """Uzun ince bir aks çizgisini (dikey: x≈yaklasik, y bas..son) mürekkep örneklerinden doğruya uydurur.
    Dönüş: birim yön vektörü (görüntü çerçevesi) ve örnek sayısı."""
    pts = []
    for t in range(int(bas), int(son), adim):
        if ekseni == 'dikey': seg = g[t, int(yaklasik - ara):int(yaklasik + ara)] < 128
        else: seg = g[int(yaklasik - ara):int(yaklasik + ara), t] < 128
        idx = np.flatnonzero(seg)
        if 0 < len(idx) <= 6 and idx.max() - idx.min() <= 6:  # yalnız ince tek çizgi
            c = yaklasik - ara + idx.mean(); pts.append((c, t) if ekseni == 'dikey' else (t, c))
    if len(pts) < 5: raise ValueError('Aks çizgisi bulunamadı')
    P = np.array(pts)
    for _ in range(3):  # aykırı örnekleri at (yazı, kesişen çizgi)
        c = P.mean(0); _, _, vt = np.linalg.svd(P - c); d = vt[0]; r_ = np.abs((P - c) @ np.array([-d[1], d[0]]))
        if r_.max() <= 2 or len(P) <= 5: break
        P = P[r_ <= max(2, np.median(r_) * 3)]
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c); d = vt[0]
    if (ekseni == 'dikey' and d[1] < 0) or (ekseni == 'yatay' and d[0] < 0): d = -d
    res = np.abs((P - c) @ np.array([-d[1], d[0]])).max()
    return [float(d[0]), float(d[1])], len(pts), float(res), [float(c[0]), float(c[1])]


def oturt(ink, pts, R=6, frac=.4, min_guven=.35, dik_aci=.4, dis=False, egim=None):
    """Kaba poligonun her kenarını iç duvar yüzüne oturtur (dis=True: kat alanı gibi dış yüze). Köşeler komşu
    kenar doğrularının kesişimidir. Eksene dik_aci dereceden yakın kenar tam yatay/dikey yapılır.
    Yüz bulunamayan ya da zayıf kontrastlı kenar yerinde kalır ve 'belirsiz' raporlanır."""
    pts = [list(map(float, p)) for p in pts]
    if Polygon(pts).area <= 0: raise ValueError('Poligon alanı sıfır')
    n = len(pts); lines = []; rep = []
    for i in range(n):
        a, b = np.array(pts[i]), np.array(pts[(i + 1) % n]); v = b - a; L = np.linalg.norm(v)
        u = v / L; nrm = np.array([-u[1], u[0]])  # sola dik (görüntü çerçevesinde)
        mid = (a + b) / 2
        inside = nrm if Polygon(pts).contains(Point(*(mid + 2 * nrm))) else -nrm
        if dis: inside = -inside
        # dik kenarlar: paftada ölçülen dikey/yatay yönler (tarama kayması); yoksa piksel eksenleri
        H_ = np.array((egim or {}).get('yatay', [1.0, 0.0])); V_ = np.array((egim or {}).get('dikey', [0.0, 1.0]))
        ortho = None
        for name, d_ in (('h', H_), ('v', V_)):
            if math.degrees(math.acos(min(1.0, abs(float(np.dot(u, d_)))))) < dik_aci:
                ortho = name; du = d_ * np.sign(np.dot(u, d_)); mid_ = (a + b) / 2
                a = mid_ - du * L / 2; b = mid_ + du * L / 2; u = du
                nn = np.array([-u[1], u[0]]); inside = nn if np.dot(nn, inside) > 0 else -nn
                break
        r = profile_face(ink, a, b, inside, R, frac) if ink is not None else None
        shift = 0
        if r and r['kayma'] is not None and r['guven'] >= min_guven: shift = r['kayma']
        out = -inside; fitted = False
        if ortho is None and L > 160 and ink is not None:
            # eğik uzun kenar: yüzü 4 istasyonda ölç, doğru uydur (açı da düzelir)
            st_ = []
            for q0 in range(4):
                sa = a + u * L * q0 / 4; sb = a + u * L * (q0 + 1) / 4; rr = profile_face(ink, sa, sb, inside, R + 2, frac)
                if rr and rr['kayma'] is not None and rr['guven'] >= min_guven: st_.append((L * (q0 + .5) / 4, rr['kayma']))
            if len(st_) >= 3:
                T = np.array(st_); bfit, afit = np.polyfit(T[:, 0], T[:, 1], 1); res_ = np.abs(T[:, 1] - (afit + bfit * T[:, 0])).max()
                if res_ <= 1.5 and abs(bfit) < .035:
                    p0 = a + afit * out; p1 = a + u * L + (afit + bfit * L) * out; u2 = (p1 - p0) / np.linalg.norm(p1 - p0)
                    lines.append((p0, u2)); fitted = True; shift = round(float(afit), 2)
        if not fitted: lines.append((a + shift * out, u))
        rep.append({'kenar': i, 'dik': ortho, 'kayma_px': shift, 'uydurma': fitted, 'oneri_px': None if not r else r['kayma'], 'guven': None if not r else r['guven'],
                    'durum': 'oturtuldu' if shift else ('yerinde' if r and r['kayma'] == 0 else 'belirsiz')})
    new = []
    for i in range(n):
        (p1, u1), (p2, u2) = lines[i - 1], lines[i]
        A = np.array([u1, -u2]).T
        if abs(np.linalg.det(A)) < math.sin(math.radians(3)):  # neredeyse paralel: kesişim kararsız
            v0 = np.array(pts[i]); q1 = p1 + np.dot(v0 - p1, u1) * u1; q2 = p2 + np.dot(v0 - p2, u2) * u2
            new.append([round(float(x), 2) for x in (q1 + q2) / 2]); continue
        t = np.linalg.solve(A, p2 - p1); new.append([round(float(x), 2) for x in p1 + t[0] * u1])
    moved = max(math.dist(p, q) for p, q in zip(pts, new))
    return new, rep, moved


def koordinat_birlestir(polys, tol=2.0, egim=None):
    """Aynı kattaki odalarda birbirine tol px'ten yakın dik kenarları (aynı duvar yüzü) tek doğruya indirger.
    Eğim verilirse iş, dikey/yatay yönlerin eksen olduğu düzeltilmiş çerçevede yapılır ve geri çevrilir."""
    A = np.array([(egim or {}).get('yatay', [1.0, 0.0]), (egim or {}).get('dikey', [0.0, 1.0])], float).T; Ai = np.linalg.inv(A)
    polys = [[list(Ai @ np.array(p, float)) for p in P] for P in polys]
    xs, ys = [], []
    for P in polys:
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            if abs(a[0] - b[0]) < 1e-3: xs.append(a[0]); a[0] = b[0] = (a[0] + b[0]) / 2
            if abs(a[1] - b[1]) < 1e-3: ys.append(a[1]); a[1] = b[1] = (a[1] + b[1]) / 2
    def clusters(vals):
        vals = sorted(vals); out = {}; grp = []
        for v in vals:
            if grp and v - grp[-1] > tol:
                m = float(np.mean(grp)); out.update({g: m for g in grp}); grp = []
            grp.append(v)
        if grp: m = float(np.mean(grp)); out.update({g: m for g in grp})
        return out
    cx, cy = clusters(xs), clusters(ys)
    near = lambda c, v: next((c[k] for k in c if abs(k - v) < 1e-6), v)
    return [[[round(float(x), 2) for x in A @ np.array([near(cx, p[0]), near(cy, p[1])])] for p in P] for P in polys]


# ----------------------------------------------------------------------------- açıklıklar
def acilik_kutusu(jambs, faces, pad_px, max_kalinlik_px, uc_pay=2.0, ink=None, kasa_ara=0, cam_ara=0, etiket_gen_px=0):
    """jambs: iki kasa noktası (TIF px). faces: [(LineString, ad)] çizili oda/kat kenarları.
    Kasa doğrusuna paralel ve aralığı örten iki yüz aranır (biri kasalara yakın olan 'ev sahibi', diğeri karşı yüz)."""
    a, b = np.array(jambs[0], float), np.array(jambs[1], float); L = np.linalg.norm(b - a)
    if L < 8: raise ValueError('Kasa aralığı çok küçük')
    u = (b - a) / L; n = np.array([-u[1], u[0]]); mid = (a + b) / 2
    cand = []
    for ls, name in faces:
        cs = np.array(ls.coords)
        for p, q in zip(cs[:-1], cs[1:]):
            v = q - p; l = np.linalg.norm(v)
            if l < 4: continue
            if abs(abs(np.dot(v / l, u)) - 1) > .02: continue  # paralel değil (~11°)
            off = float(np.dot(p - mid, n))
            s = sorted([float(np.dot(p - a, u)), float(np.dot(q - a, u))])
            ov = min(s[1], L) - max(s[0], 0)
            if ov < .5 * L: continue
            if abs(off) > max_kalinlik_px: continue
            cand.append({'off': off, 'span': s, 'ad': name})
    if not cand: raise ValueError('Kasaların yanında çizili duvar yüzü yok')
    host = min(cand, key=lambda c: abs(c['off']))
    opp = [c for c in cand if (c['off'] - host['off']) * (1 if host['off'] <= 0 else -1) > 4 or abs(c['off'] - host['off']) > 4]
    opp = [c for c in opp if abs(c['off'] - host['off']) >= 6]
    if not opp: raise ValueError('Karşı duvar yüzü yok (karşı oda/kat alanı çizilmemiş ya da duvar max kalınlıktan kalın)')
    o = min(opp, key=lambda c: abs(c['off'] - host['off']))
    lo, hi = sorted([host['off'], o['off']])
    kasa_rap = None
    if ink is not None and kasa_ara > 0:
        # kasa blokları: duvar orta çizgisi boyunca kısa mürekkep dizileri; verilen uçlara en yakın olanların dış kenarı
        mid_off = (lo + hi) / 2; ts = np.arange(-kasa_ara, L + kasa_ara); H_, W_ = ink.shape; hits = np.zeros(len(ts), bool)
        for dd in (-2, 0, 2):
            P = a[None] + ts[:, None] * u[None] + (mid_off + dd + float(np.dot(mid - a, n))) * n[None]
            xi = np.clip(np.floor(P[:, 0]).astype(int), 0, W_ - 1); yi = np.clip(np.floor(P[:, 1]).astype(int), 0, H_ - 1); hits |= ink[yi, xi]
        runs = []; st = None
        for j, h in enumerate(hits):
            if h and st is None: st = j
            if not h and st is not None: runs.append((ts[st], ts[j - 1])); st = None
        if st is not None: runs.append((ts[st], ts[-1]))
        runs = [r for r in runs if 2 <= r[1] - r[0] + 1 <= 22]
        left = [r for r in runs if abs(r[0] - 0) <= kasa_ara or abs(r[1] - 0) <= kasa_ara]
        right = [r for r in runs if abs(r[0] - L) <= kasa_ara or abs(r[1] - L) <= kasa_ara]
        if left and right:
            pairs = [(l_, r_) for l_ in left for r_ in right if r_[1] > l_[0] and l_ != r_]
            if etiket_gen_px > 0 and pairs:  # etiketteki genişliğe en yakın kasa çifti (±%20 içinde)
                lr, rr = min(pairs, key=lambda q: abs((q[1][1] + 1 - q[0][0]) - etiket_gen_px))
                if abs((rr[1] + 1 - lr[0]) - etiket_gen_px) > .2 * etiket_gen_px:
                    lr = min(left, key=lambda r: abs((r[0] + r[1]) / 2)); rr = min(right, key=lambda r: abs((r[0] + r[1]) / 2 - L))
            else:
                lr = min(left, key=lambda r: abs((r[0] + r[1]) / 2)); rr = min(right, key=lambda r: abs((r[0] + r[1]) / 2 - L))
            n0, n1 = float(lr[0]), float(rr[1] + 1)
            if n1 - n0 > .5 * L and lr != rr:
                kasa_rap = {'eski_px': [0, round(L, 1)], 'yeni_px': [round(n0, 1), round(n1, 1)]}
                a = a + n0 * u; L = n1 - n0; b = a + L * u; mid = (a + b) / 2
                for c in (host, o): c['span'] = [c['span'][0] - n0, c['span'][1] - n0]
    if ink is not None and cam_ara > 0 and kasa_rap is None:
        # pencere: duvar orta çizgisinde camın boyuna çizgileri sürekli; ortadan iki yana sürekliliği izle
        mid_off = (lo + hi) / 2; ts = np.arange(-cam_ara, L + cam_ara); H_, W_ = ink.shape
        offs = np.linspace(lo + .25 * (hi - lo), hi - .25 * (hi - lo), 5) + float(np.dot(mid - a, n))
        rows = []
        for dd in offs:
            P = a[None] + ts[:, None] * u[None] + dd * n[None]
            xi = np.clip(np.floor(P[:, 0]).astype(int), 0, W_ - 1); yi = np.clip(np.floor(P[:, 1]).astype(int), 0, H_ - 1); rows.append(ink[yi, xi])
        on = np.vstack(rows).any(0); c0 = int(np.argmin(np.abs(ts - L / 2)))
        if on[c0]:
            l = c0; gap = 0
            while l > 0 and gap <= 8: l -= 1; gap = 0 if on[l] else gap + 1
            r_ = c0; gap = 0
            while r_ < len(ts) - 1 and gap <= 8: r_ += 1; gap = 0 if on[r_] else gap + 1
            n0 = 0.0 if l == 0 else float(ts[l] + 9); n1 = L if r_ == len(ts) - 1 else float(ts[r_] - 8)  # sınıra dayandıysa verilen uç
            if .5 * L <= n1 - n0 <= 2.2 * L:
                kasa_rap = {'eski_px': [0, round(L, 1)], 'yeni_px': [round(n0, 1), round(n1, 1)], 'cam': True}
                a = a + n0 * u; L = n1 - n0; b = a + L * u; mid = (a + b) / 2
                for c in (host, o): c['span'] = [c['span'][0] - n0, c['span'][1] - n0]
    lo -= pad_px; hi += pad_px
    s0 = max(0.0, host['span'][0], o['span'][0]); s1 = min(L, host['span'][1], o['span'][1])
    # yüz ucunda (köşe/kolon dönüşü) bitiyorsa pay bırak: Citymax 'birden fazla oda yüzeyine değiyor' / vertex hatası
    if s0 > 0: s0 += uc_pay
    if s1 < L: s1 -= uc_pay
    if s1 - s0 < .8 * L: raise ValueError(f'İki yüzün ortak aralığı kasa aralığının %{100*(s1-s0)/L:.0f}\'i; kasa köşeye/kolona taşıyor')
    P = lambda s, t: [round(float(x), 2) for x in a + s * u + t * n + n * float(np.dot(mid - a, n))]
    box = [P(s0, lo), P(s1, lo), P(s1, hi), P(s0, hi)]
    return box, {'ev_sahibi': host['ad'], 'karsi': o['ad'], 'duvar_px': round(abs(o['off'] - host['off']), 1), 'kirpma_px': round(L - (s1 - s0), 1), 'kasa_oturtma': kasa_rap}


def etiket_olcu(tag):
    m = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', str(tag or ''))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _balkon_poligon(pts, FP):
    """Açık balkon sınırı (2+ nokta) ile KatAlani (FP) dış sınırının kapattığı balkon poligonunu döner."""
    from shapely.ops import snap as _snap
    if not pts or len(pts) < 2: return None
    q = [list(map(float, p)) for p in pts]
    ex = FP.exterior
    for end in (0, -1):
        np_ = nearest_points(ex, Point(q[end]))[0]
        if np_.distance(Point(q[end])) < 40: q[end] = [np_.x, np_.y]
    ls0 = LineString(q)
    # Hem doğrudan snapli birleşimi hem de 2 px uzatılmış birleşimi dene (köşe ve kenar durumları için)
    ex_s = _snap(ex, ls0, 1e-3)
    cand = [r for r in polygonize(unary_union([ex_s, ls0])) if r.difference(FP).area > 0.5 * r.area]
    if not cand:
        q_ext = [list(p) for p in q]
        for end, nxt in ((0, 1), (-1, -2)):
            v = np.subtract(q_ext[end], q_ext[nxt]); L = float(np.linalg.norm(v))
            if L > 1e-6: q_ext[end] = (np.array(q_ext[end]) + (v / L) * 2.0).tolist()
        cand = [r for r in polygonize(unary_union([ex, LineString(q_ext)])) if r.difference(FP).area > 0.5 * r.area]
    if cand:
        hit = [r for r in cand if r.area < FP.area * 0.5 and r.exterior.intersection(ls0.buffer(2.5)).length > 0.4 * ls0.length]
        if hit: return max(hit, key=lambda r: r.area)
        small = [r for r in cand if r.area < FP.area * 0.5]
        if small: return max(small, key=lambda r: r.area)
    if len(q) >= 3:
        g_ = Polygon(q).buffer(0)
        return g_ if g_.area > 0 else None
    return None


# ----------------------------------------------------------------------------- motor
class Motor:
    def __init__(self, work):
        self.work = Path(work).resolve(); self.work.mkdir(parents=True, exist_ok=True)
        self.a = app(); self.f = self.a.ActiveDesignFile; self.m = self.a.ActiveModelReference; self.q = self.a.CadInputQueue
        self.pt = self.a.Point3dFromXYZ; self.s = snapshot(self.a)
        self.levels = {l.Name: l for l in self.f.Levels}

    # ---------------- temel
    def rmap(self, i): return RasterMap(self.s['rasters'][i - 1])
    def kaynak(self, i): return Kaynak(self.s['rasters'][i - 1]['tif'])
    def ppm(self, i):
        r = self.s['rasters'][i - 1]; return r['pixels'][0] / r['extent'][0]

    def refresh(self): self.s = snapshot(self.a); return self.s

    def ink(self, i, P=None):
        """Plan 'pafta' ve 'bina' kutusu veriyorsa aks çizgileri silinmiş mürekkep, yoksa ham mürekkep."""
        k = self.kaynak(i); P = P or {}
        if not (P.get('pafta') and P.get('bina')): return k.inkmap()
        key = (i, tuple(P['pafta']), tuple(P['bina']))
        if not hasattr(self, '_temiz'): self._temiz = {}
        if key not in self._temiz:
            x0, y0, x1, y1 = map(int, P['pafta']); full = k.inkmap(); sub = full[y0:y1, x0:x1]
            b = P['bina']; clean, axes = aks_temizle(sub, [0, 0, x1 - x0, y1 - y0], [b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0], P.get('egim'))
            out = full.copy(); out[y0:y1, x0:x1] = clean; self._temiz[key] = (out, [(a, c + (x0 if a == 'dikey' else y0)) for a, c in axes])
        return self._temiz[key][0]

    def elements_on(self, i, levels=None):
        """i. raster çerçevesine düşen çizim öğeleri (TIF pikseline çevrilmiş)."""
        rm = self.rmap(i); r = self.s['rasters'][i - 1]; W, H = r['pixels']; out = []
        for e in self.s['elements']:
            if not e.get('points') or (levels and e['level'] not in levels): continue
            px = [rm.pixels(*p[:2]) for p in e['points']]
            if all(-50 <= x <= W + 50 and -50 <= y <= H + 50 for x, y in px):
                out.append({**e, 'px': px})
        return out

    def tags_of(self, eid): return {t['tag']: t['value'] for t in self.s['elements'] if t.get('base') == eid}

    def add(self, kind, world, closed=True, style=None):
        name, color = LEVEL[kind]; v = [self.pt(*p, 0) for p in world]
        e = (self.a.CreateShapeElement1(None, v, 0) if closed else self.a.CreateLineElement1(None, v))[-1]
        _set_object_property(e, 'Level', self.levels[name]); e.Color = color; e.LineWeight = 1
        if style: _set_object_property(e, 'LineStyle', self.f.LineStyles.Item(style))
        self.m.AddElement(e); return e

    def put_tags(self, e, kind, values, xy, color=None):
        title, spec = SCHEMA[kind]
        try:
            ts = self.f.TagSets.Item(title)
        except Exception:
            ts = self.f.TagSets.Add(title)
            for label, typ, _, _ in spec:
                td = ts.TagDefinitions.Add(label, typ)
                td.IsConstant = False; td.IsHidden = False
        font = self.f.Fonts.Item('Arial')
        for j, ((label, typ, lev, _), val) in enumerate(zip(spec, values, strict=True)):
            t = e.AddTag(ts.TagDefinitions.Item(label)); t.Value = -1.0 if val is None and typ == 4 else ('-1' if val is None and typ == 1 else (-1 if val is None else val))
            _set_object_property(t, 'Level', self.levels[lev]); _set_object_property(t, 'Font', font)
            h = HEIGHT.get(lev, .2); t.Height = h; t.Width = h; t.Color = color if color is not None else e.Color
            t.IsHidden = val is None or val in ('', '-1', -1)
            o, d = t.Origin, t.Offset; dy = sum(HEIGHT.get(spec[k][2], .2) * 1.25 for k in range(j))
            t.Move(self.pt(xy[0] - o.X - d.X, xy[1] - dy - o.Y - d.Y, -o.Z - d.Z)); t.Rewrite()

    def city(self, key, options=None, params=None):
        self.a.CommandState.StartDefaultCommand(); self.m.UnselectAllElements()
        self.q.SendKeyin('citymax islem cizim ' + key); time.sleep(.15)
        code = int(self.a.GetCExpressionValue('gCityMax.Islem', 'CITYMAX'))
        if key in CODES and code != CODES[key]: raise RuntimeError(f'Citymax işlem kodu {code} != {CODES[key]} ({key})')
        for k, v in (options or {}).items(): self.a.SetCExpressionValue('gCityMax.' + k, v, 'CITYMAX')
        for i, v in enumerate(params or [], 1): self.a.SetCExpressionValue(f'gCityMax.Parametre{i}', str(v), 'CITYMAX')
        self.q.SendKeyin('citymax baslat'); time.sleep(.15)

    def click(self, xy): self.q.SendDataPoint(self.pt(*xy, 0), 1); time.sleep(.12)
    def stop(self): self.q.SendReset(); self.a.CommandState.StartDefaultCommand(); time.sleep(.08)

    def view(self, xy, span=30):
        v = self.f.Views(1); getattr(v, '__View__com__set_Center')(self.pt(*xy, 0)); getattr(v, '__View__com__set_Extents')(self.pt(span, span, 24)); v.Redraw()

    def backup(self, name):
        self.f.Save(); p = self.work / f'{name}.dgn'
        if not p.exists(): shutil.copy2(self.s['dgn'], p)
        return p

    def rollback(self, before_ids):
        now = snapshot(self.a)
        for row in sorted([r for r in now['elements'] if r['id'] not in before_ids], key=lambda r: r['type'] != 37):
            try: self.m.RemoveElement(self.m.GetElementByID64(row['id']))
            except Exception: pass
        self.f.Save()

    # ---------------- kat yazımı
    def kat_yaz(self, plan_path, adimlar=('cerceve', 'kat', 'oda', 'merdiven', 'acilik'), kuru=False):
        """Plan JSON'u (TIF pikseli) doğrular, düzeltir, kuru=False ise DGN'e yazar. Rapor döner."""
        P = rd(plan_path); i = P['raster']; rm = self.rmap(i); ppm = self.ppm(i); W = lambda p: rm.xy(*p)
        k = self.kaynak(i); ink = self.ink(i, P); rep = {'plan': str(plan_path), 'raster': i, 'uyari': [], 'hata': []}
        # 1) odaları oturt ve birleştir
        rooms = P.get('odalar', [])
        bad = [f"{r['anahtar']}: '{r['ad']}' Citymax oda adı değil" for r in rooms if r['ad'] not in ODA_ADLARI]
        if bad: rep['hata'] += bad; rep['yazildi'] = False; return rep
        for r in rooms:
            if r.get('oturt', True) and not r.get('kesin'):
                new, er, mv = oturt(ink, r['nokta'], R=P.get('oturt_px', 6), egim=P.get('egim'))
                r['_ham'] = r['nokta']; r['nokta'] = new; r['_oturt'] = er
                if mv > 10: rep['uyari'].append(f"{r['anahtar']}: oturtma köşeyi {mv:.1f} px oynattı, kontrol et")
        if P.get('birlestir', True) and rooms:
            for r, q in zip(rooms, koordinat_birlestir([r['nokta'] for r in rooms], egim=P.get('egim'))): r['nokta'] = q
        if P.get('kolon', True) and rooms:
            allp = np.array([p for r in rooms for p in r['nokta']]); bb = [allp[:, 0].min() - 80, allp[:, 1].min() - 80, allp[:, 0].max() + 80, allp[:, 1].max() + 80]
            cols = [{'kutu': kutu_oturt(ink, b)} for b in P.get('kolonlar', [])]; rep['kolon'] = len(cols); P['_kolonlar'] = cols; frees = []
            # denetim: bulucunun gördüğü ama planda olmayan, bir odanın sınırına >= 3 px giren kolon -> hata (sessiz eksik olmasın)
            given = [Polygon(c['kutu']) for c in cols]
            for c in kolon_bul(ink, bb, P.get('egim')):
                K = Polygon(c['kutu'])
                if any(K.intersection(g).area > .3 * K.area for g in given): continue
                bx_ = K.bounds
                if min(bx_[2] - bx_[0], bx_[3] - bx_[1]) < 28: continue  # kasa bloğu / ölçü yazısı boyutu
                for r in rooms:
                    R_ = Polygon(r['nokta'])
                    if not R_.is_valid or R_.buffer(-4).contains(K): continue
                    I = K.intersection(R_)
                    if not I.is_empty and I.area > 9 and min(I.bounds[2] - I.bounds[0], I.bounds[3] - I.bounds[1]) >= 3:  # yalnız kenardan değme (çizgi) sayılmaz
                        rep['hata'].append(f"planda olmayan kolon {[round(v) for v in K.bounds]} -> {r['anahtar']} sınırına giriyor"); break
            for r in rooms:
                if r.get('kolonsuz'): continue
                new, att, free = kolon_cikar(r['nokta'], cols, ppm)
                if att: r['nokta'] = new; r['_kolon_donus'] = len(att)
                if free: frees += [{'oda': r['anahtar'], 'kutu': f} for f in free]
        floor = P.get('kat_alani')
        if floor and P.get('kat_oturt', True) and not P.get('kat_kesin'):
            newf, er, mv = oturt(ink, floor, R=P.get('oturt_px', 6), dis=True, egim=P.get('egim'))
            P['_kat_ham'] = floor; floor = P['kat_alani'] = newf
            rep['kat_oturt'] = {'oynama_px': round(mv, 1), 'belirsiz_kenar': [r['kenar'] for r in er if r['durum'] == 'belirsiz']}
        # 2) topoloji
        FP = Polygon(floor) if floor else None
        polys = {r['anahtar']: Polygon(r['nokta']) for r in rooms}
        for kk, g in polys.items():
            if not g.is_valid: rep['hata'].append(f'{kk}: geçersiz poligon (kendini kesiyor)')
            if FP is not None and g.difference(FP.buffer(1.0)).area > 4: rep['hata'].append(f'{kk}: kat alanı dışına taşıyor ({g.difference(FP).area:.0f} px²)')
        keys = list(polys)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                A = polys[keys[x]].intersection(polys[keys[y]]).area
                if A > 4: rep['hata'].append(f'{keys[x]} ile {keys[y]} örtüşüyor ({A:.0f} px²)')
        for r in rooms:
            r['etiket'] = r.get('etiket') or list(polylabel(polys[r['anahtar']], 1.0).coords)[0]
            if not polys[r['anahtar']].covers(Point(r['etiket'])): rep['hata'].append(f"{r['anahtar']}: etiket noktası oda dışında")
            if r.get('m2') and polys[r['anahtar']].is_valid:
                a2 = polys[r['anahtar']].area / ppm / ppm
                if abs(a2 - r['m2']) / r['m2'] > .08: rep['uyari'].append(f"{r['anahtar']} {r['ad']}: pafta {r['m2']} m², çizim {a2:.2f} m²")
        # 3) merdiven yakalama
        for s in P.get('merdivenler', []):
            host = polys[s['oda']]; pts = []; snapped = 0
            for p in s['nokta']:
                q = nearest_points(host.exterior, Point(p))[0]
                if q.distance(Point(p)) <= 4: pts.append([round(q.x, 2), round(q.y, 2)]); snapped += 1
                else: pts.append(p)
            s['nokta'] = pts
            if snapped < 2: rep['hata'].append(f"merdiven {s.get('anahtar')}: oda sınırına en az iki noktadan değmiyor")
            if not host.buffer(1).covers(Polygon(pts)): rep['hata'].append(f"merdiven {s.get('anahtar')}: odanın dışına taşıyor")
        # 4) açıklıklar
        faces = [(LineString(list(g.exterior.coords)), k_) for k_, g in polys.items()]
        if FP is not None: faces.append((LineString(list(FP.exterior.coords)), 'KAT'))
        recp = self.work / (Path(plan_path).stem + '_yazildi.json'); own = set()
        if recp.exists():
            def _w(v):
                if isinstance(v, dict): [_w(x) for x in v.values()]
                elif isinstance(v, int): own.add(v)
            _w(rd(recp)['idler'])
        faces += [(LineString(e['px']), 'KAT' if e['level'] == 'KatAlani' else f"DGN{e['id']}") for e in self.elements_on(i, {'OdaAlani', 'KatAlani'}) if FP is not None and Polygon(e['px']).intersects(FP) and e['id'] not in own]
        ops = []
        for o in P.get('acikliklar', []):
            try:
                bx, info = acilik_kutusu(o['kasa'], faces, pad_px=o.get('pay_cm', 2) / 100 * ppm, max_kalinlik_px=P.get('max_duvar_cm', 60) / 100 * ppm, ink=ink, kasa_ara=(o.get('kasa_ara', 25) if o['tip'] == 'kapi' else 0), cam_ara=(o.get('cam_ara', 40) if o['tip'] == 'pencere' else 0), etiket_gen_px=((etiket_olcu(o.get('etiket'))[0] or 0) / 100 * ppm))
                if info.get('kasa_oturtma'): o = {**o, 'kasa': [list(map(float, bx[0])), list(map(float, bx[1]))]}
            except ValueError as e:
                rep['hata'].append(f"{o.get('anahtar', o['kasa'])}: {e}"); continue
            w_cm, h_cm = etiket_olcu(o.get('etiket'))
            span = math.dist(bx[0], bx[1]) / ppm * 100
            if w_cm and abs(span - w_cm) > max(6, .06 * w_cm): rep['uyari'].append(f"{o.get('anahtar')}: kasa aralığı {span:.0f} cm, etiket {w_cm} cm")
            if o['tip'] == 'pencere' and 'KAT' not in (info['ev_sahibi'], info['karsi']): rep['uyari'].append(f"{o.get('anahtar')}: pencere dış yüzü (kat alanı) kesmiyor")
            ops.append({**o, 'kutu': bx, 'bilgi': info, 'genislik_cm': round(span, 1), 'yukseklik': o.get('yukseklik') or h_cm or (210 if o['tip'] == 'kapi' else 130)})
        # aynı duvarda kapı-pencere çakışması: kapı kasası öncelikli, pencere kapının bitiminden 2 px sonra başlar
        for w in [o for o in ops if o['tip'] == 'pencere']:
            for d in [o for o in ops if o['tip'] == 'kapi']:
                if Polygon(w['kutu']).intersection(Polygon(d['kutu'])).area <= 1: continue
                A0, A1, A2, A3 = [np.array(q, float) for q in w['kutu']]; u_ = (A1 - A0) / np.linalg.norm(A1 - A0)
                proj = [float(np.dot(np.array(q) - A0, u_)) for q in d['kutu']]; Lw = float(np.linalg.norm(A1 - A0))
                lo_, hi_ = min(proj), max(proj); s0_, s1_ = 0.0, Lw
                if lo_ <= 0 < hi_ or (0 < lo_ and hi_ - 0 < Lw - lo_ and lo_ < Lw / 2): s0_ = hi_ + 2
                else: s1_ = lo_ - 2
                if s1_ - s0_ < 10: continue
                w['kutu'] = [[round(float(v), 2) for v in A0 + s0_ * u_], [round(float(v), 2) for v in A0 + s1_ * u_], [round(float(v), 2) for v in A3 + s1_ * u_], [round(float(v), 2) for v in A3 + s0_ * u_]]
                w['genislik_cm'] = round((s1_ - s0_) / ppm * 100, 1); w.setdefault('not_', []).append(f"{d.get('anahtar')} kapısına göre kırpıldı")
        for x in range(len(ops)):
            for y in range(x + 1, len(ops)):
                if Polygon(ops[x]['kutu']).intersection(Polygon(ops[y]['kutu'])).area > 1: rep['hata'].append(f"{ops[x].get('anahtar')} ile {ops[y].get('anahtar')} kesişiyor")
        doored = {k_ for o in ops if o['tip'] == 'kapi' for k_ in (o['bilgi']['ev_sahibi'], o['bilgi']['karsi'])}
        for r in rooms:
            # kullanıcı kuralı 2026-11-19: havalandırma şaftı N.O.A çizilir (havalandirma: true), kapısı olmaz;
            # F11 onun etrafından dolaşır (oda içinden geçme denetimi şaftı da kapsar)
            if r.get('havalandirma'):
                if r['ad'] != 'N.O.A': rep['hata'].append(f"{r['anahtar']}: havalandırma şaftı N.O.A olmalı")
                continue
            if r['anahtar'] not in doored and not r.get('kapisiz'): rep['uyari'].append(f"{r['anahtar']} {r['ad']}: kapısı yok")
        rep['oda'] = len(rooms); rep['acilik'] = len(ops)
        P['_odalar'] = rooms; P['_acikliklar'] = ops
        save(self.work / (Path(plan_path).stem + '_hazir.json'), P)
        if kuru or rep['hata']:
            rep['yazildi'] = False; return rep
        # 5) yazım
        self.refresh(); s0 = {e['id']: e for e in self.s['elements']}
        before = set(s0); self.backup(Path(plan_path).stem + '_ONCE'); made = {}
        try:
            if 'cerceve' in adimlar and P.get('cerceve'):
                made['cerceve'] = int(self.add('frame', [W(p) for p in P['cerceve']]).ID64)
                made['aks'] = int(self.add('axis', [W(p) for p in P['aks']]).ID64)
            if 'kat' in adimlar and floor:
                kind = 'building' if P.get('bina_alani') else 'floor'
                e = self.add(kind, [W(p) for p in floor]); made['kat'] = int(e.ID64)
                fr = P.get('cerceve') or [[min(p[0] for p in floor), min(p[1] for p in floor)]]
                tl = [min(p[0] for p in fr) + .6 * ppm, min(p[1] for p in fr) + .7 * ppm]
                vals = [P['bina']['cati_kotu'], P['bina']['asansor'], P['bina']['onay'], None] if kind == 'building' else [P['kat_no'], P['taban'], P['tavan']]
                self.put_tags(e, kind, vals, W(tl))
            if 'oda' in adimlar:
                made['odalar'] = {}
                for r in rooms:
                    e = self.add('room', [W(p) for p in r['nokta']]); lab = W(r['etiket'])
                    self.put_tags(e, 'room', [r['ad'], r.get('m2')], (lab[0] - .5, lab[1] + .15)); made['odalar'][r['anahtar']] = int(e.ID64)
            if 'merdiven' in adimlar and P.get('merdivenler'):
                made['merdivenler'] = {}
                for s in P['merdivenler']:
                    e = self.add('stair', [W(p) for p in s['nokta']]); c = Polygon(s['nokta']).representative_point()
                    self.put_tags(e, 'stair', [s.get('ad', 'MERDİVEN')], W((c.x, c.y))); made['merdivenler'][s.get('anahtar', str(len(made['merdivenler'])))] = int(e.ID64)
            if 'acilik' in adimlar:
                made['acikliklar'] = {}
                for o in ops:
                    kind = 'door' if o['tip'] == 'kapi' else 'window'
                    e = self.add(kind, [W(p) for p in o['kutu']], style=None if kind == 'door' else '2'); c = Polygon(o['kutu']).centroid
                    # eşik: pafta verirse o; yoksa pencere üstü tavandan 50 cm aşağıda (Citymax eksi kabul etmez)
                    esik = o.get('esik') if o.get('esik') is not None else max(0, int(P['tavan'] - P['taban'] - 50 - o['yukseklik']))
                    vals = [o['yukseklik']] if kind == 'door' else [o['yukseklik'], esik]
                    self.put_tags(e, kind, vals, W((c.x, c.y))); made['acikliklar'][o.get('anahtar', str(len(made['acikliklar'])))] = int(e.ID64)
            self.a.CommandState.StartDefaultCommand(); self.f.Save(); after = self.refresh()
            changed = [e['id'] for e in after['elements'] if e['id'] in s0 and s0[e['id']] != e] + [k_ for k_ in s0 if k_ not in {e['id'] for e in after['elements']}]
            if changed: raise RuntimeError(f'Mevcut {len(changed)} öğe değişti')
            rep['etiket'] = self.etiket_sigdir({e['id'] for e in after['elements'] if e['id'] not in before})
            rep['yazildi'] = True; rep['yeni_oge'] = len(after['elements']) - len(before); rep['idler'] = made
            recp = self.work / (Path(plan_path).stem + '_yazildi.json'); prev = rd(recp)['idler'] if recp.exists() else {}
            prev.update(made); save(recp, {'idler': prev})
        except Exception as exc:
            self.stop(); self.rollback(before); self.refresh(); rep['hata'].append('yazım geri alındı: ' + repr(exc)); rep['yazildi'] = False
        return rep

    def kat_sil(self, plan_path, adim=None):
        """Yalnız bu planın yazdığı kimlikleri siler (kullanıcı çizimine dokunmaz). adim: 'acikliklar', 'odalar' ..."""
        stem = Path(plan_path).stem
        if adim in ('bolumler', 'balkonlar'):
            path = self.work / (stem + ('_bolum_yazildi.json' if adim == 'bolumler' else '_balkon_yazildi.json')); rec = rd(path); unlink_after = path
        else:
            path = self.work / (stem + '_yazildi.json'); rec = rd(path)
            if adim:
                part = rec['idler'].pop(adim); save(path, rec); rec = {'idler': part}
        ids = set()
        def walk(v):
            if isinstance(v, dict): [walk(x) for x in v.values()]
            elif isinstance(v, (list, tuple)): [walk(x) for x in v]
            elif isinstance(v, int): ids.add(v)
        walk(rec['idler']); now = self.refresh()
        tags = [e['id'] for e in now['elements'] if e.get('base') in ids]
        for eid in tags + sorted(ids):
            try: self.m.RemoveElement(self.m.GetElementByID64(eid))
            except Exception: pass
        self.f.Save(); self.refresh()
        if adim in ('bolumler', 'balkonlar'): unlink_after.unlink()   # kayıt ancak silme bittikten sonra kalkar
        return {'silinen': len(ids) + len(tags)}

    # ---------------- etiketler alanın içinde
    def etiket_sigdir(self, ids=None, levels=('OdaAlani', 'BagimsizBolumAlani', 'OrtakAlan', 'BagimsizBolumOrtakAlani', 'BalkonAlani', 'KapiAcikligi', 'PencereAcikligi', 'OdaIcYapi'), pay=.02, min_boy=.06):
        """Her alanın görünür Tag'lerini alanın içine alır. Önce boy varsayılana döner; oda/alan etiketleri alt alta,
        açıklık etiketleri uzun kenar boyunca yan yana dizilir; blok alanın en iç noktasına taşınır; sığmazsa boy
        %85 adımlarla küçülür. Kimlik, değer ve stil korunur. Range her ölçümde yeniden okunur."""
        s = self.refresh(); out = {'yerlesti': 0, 'kucultuldu': 0, 'sigmadi': []}
        bases = [e for e in s['elements'] if e.get('points') and e['level'] in levels and (ids is None or e['id'] in ids)]
        G = lambda i: self.m.GetElementByID64(i)
        for b in bases:
            poly = Polygon([p[:2] for p in b['points']]).buffer(0)
            if poly.area <= 0: continue
            tids = [t['id'] for t in s['elements'] if t.get('base') == b['id'] and not G(t['id']).AsTagElement.IsHidden]
            if not tids: continue
            opening = b['level'] in ('KapiAcikligi', 'PencereAcikligi')
            mr = poly.minimum_rotated_rectangle; cs = list(mr.exterior.coords)
            long_ = max(((cs[k], cs[k + 1]) for k in range(4)), key=lambda e: math.dist(*e)); along = np.subtract(long_[1], long_[0]); along = along / np.linalg.norm(along)
            if along[0] < 0: along = -along
            inner = poly.buffer(-(.004 if opening else pay))
            if inner.is_empty or inner.geom_type != 'Polygon': inner = poly
            mb = .045 if opening else min_boy
            h0 = {i: HEIGHT.get(str(G(i).Level.Name), .2) * (.75 if opening else 1) for i in tids}
            for k in range(14):
                f = .85 ** k
                if k and min(h0[i] * f for i in tids) < mb: break
                for i in tids:
                    te = G(i).AsTagElement; te.Height = h0[i] * f; te.Width = h0[i] * f; te.Rewrite()
                c = polylabel(inner, .005); cur = (c.x, c.y); boxes = []
                for j, i in enumerate(tids):  # yerleşim: önce hepsini tek noktaya topla
                    e = G(i); r = e.Range; w_ = r.High.X - r.Low.X; h_ = r.High.Y - r.Low.Y; boxes.append((i, w_, h_))
                if opening:
                    total = sum(w_ for _, w_, _ in boxes) * 1.1; pos = -total / 2
                    targets = []
                    for i, w_, h_ in boxes: targets.append((i, cur[0] + along[0] * (pos + w_ * .55), cur[1] + along[1] * (pos + w_ * .55))); pos += w_ * 1.1
                else:
                    total = sum(h_ for _, _, h_ in boxes) * 1.05; pos = total / 2
                    targets = []
                    for i, w_, h_ in boxes: targets.append((i, cur[0], cur[1] + pos - h_ * .525)); pos -= h_ * 1.05
                for i, x, y in targets:
                    e = G(i); r = e.Range; dx = x - (r.Low.X + r.High.X) / 2; dy = y - (r.Low.Y + r.High.Y) / 2
                    e.Move(self.pt(dx, dy, 0)); e.Rewrite()
                ok = all(inner.covers(sbox(G(i).Range.Low.X, G(i).Range.Low.Y, G(i).Range.High.X, G(i).Range.High.Y)) for i in tids)
                if ok:
                    out['yerlesti'] += 1; out['kucultuldu'] += int(k > 0); break
            else:
                pass
            if not all(inner.covers(sbox(G(i).Range.Low.X, G(i).Range.Low.Y, G(i).Range.High.X, G(i).Range.High.Y)) for i in tids): out['sigmadi'].append([b['id'], b['level']])
        self.f.Save(); self.refresh(); return out

    # ---------------- F11 / F12 / ortak alan
    def bolumler(self, plan_path, kuru=False):
        P = rd(plan_path); i = P['raster']; rm = self.rmap(i); W = lambda p: rm.xy(*p); ppm = self.ppm(i); self.refresh()
        mine = self.elements_on(i)
        FP = Polygon(P['kat_alani']) if P.get('kat_alani') else None
        fl = [e for e in mine if e['level'] == 'KatAlani' and (FP is None or Polygon(e['px']).intersects(FP))]
        if len(fl) != 1: raise RuntimeError(f'Bu katta {len(fl)} kat alanı var')
        FP = Polygon(fl[0]['px']); rooms = [Polygon(e['px']) for e in mine if e['level'] == 'OdaAlani' and FP.covers(Polygon(e['px']).representative_point())]
        ops = [Polygon(e['px']) for e in mine if e['level'] in ('KapiAcikligi', 'PencereAcikligi') and FP.buffer(5).covers(Polygon(e['px']))]
        rep = {'hata': [], 'uyari': []}; divs = []; d_egik = {i: True for i in P.get('egik_ayirici', [])}
        for d in P.get('ayiricilar', []):
            pts = [list(map(float, p)) for p in d]
            for end, nxt in ((0, 1), (-1, -2)):  # kat alanına yakın uç dışarı uzatılır
                p = np.array(pts[end]); q = np.array(pts[nxt]); v = p - q; v /= np.linalg.norm(v)
                if FP.exterior.distance(Point(p)) < .25 * ppm or not FP.contains(Point(p)):
                    k = 0
                    while FP.buffer(.05 * ppm).contains(Point(*(p + k * v))) and k < 2 * ppm: k += 1
                    pts[end] = list((p + (k + .3 * ppm) * v).round(2))
            ls = LineString(pts)
            if not ls.is_simple: rep['hata'].append(f'ayırıcı {len(divs)} kendini kesiyor')
            # kural: parçalar paftanın dikey/yatay yönünde (eğik duvar izlenmiyorsa); duvar ortasından; balkondan geçmez
            Hd = np.array((P.get('egim') or {}).get('yatay', [1, 0])); Vd = np.array((P.get('egim') or {}).get('dikey', [0, 1]))
            faces_ = unary_union([g.exterior for g in rooms])
            for si, (q0, q1) in enumerate(zip(pts, pts[1:])):
                v = np.subtract(q1, q0); Lq = float(np.linalg.norm(v))
                if Lq < 1: continue
                u_ = v / Lq; ang = min(math.degrees(math.acos(min(1, abs(float(np.dot(u_, Hd)))))), math.degrees(math.acos(min(1, abs(float(np.dot(u_, Vd)))))))
                on_edge = FP.exterior.distance(Point(*(np.add(q0, q1) / 2))) < 2.5 or not FP.contains(Point(*(np.add(q0, q1) / 2)))
                if ang > 3 and not d_egik.get(len(divs)) and not on_edge: rep['hata'].append(f'ayırıcı {len(divs)} parça {si}: {ang:.0f}° eğik (dik ya da duvar boyunca olmalı)')
                samp = [Point(*(np.array(q0) + u_ * t)) for t in np.arange(0, Lq, 4)]
                inside = [pt for pt in samp if FP.contains(pt)]
                if inside:
                    def tek_yuz(pt):  # bir yüze yakın, karşı yüz uzak -> duvar ortası değil (ince bölmede iki yüz de yakın)
                        ds = sorted(g.exterior.distance(pt) for g in rooms)
                        return len(ds) > 0 and ds[0] < 4 and (len(ds) < 2 or ds[1] > 10)
                    close = sum(tek_yuz(pt) for pt in inside)
                    if close > .3 * len(inside) and close * 4 > .3 * ppm: rep['hata'].append(f'ayırıcı {len(divs)} parça {si}: duvar ortasından değil, oda yüzünün {4} px yakınından geçiyor')
            for bk in P.get('balkonlar', []):
                bp = _balkon_poligon(bk['nokta'], FP)
                if bp is not None and bp.is_valid and ls.intersection(bp.buffer(-2)).length > 2: rep['hata'].append(f"ayırıcı {len(divs)} balkon {bk['anahtar']} içinden geçiyor")
            for g in rooms:
                if g.buffer(-.03 * ppm).intersects(ls): rep['hata'].append(f'ayırıcı {len(divs)} bir odanın içinden geçiyor')
            for o in ops:
                if any(o.buffer(.5).contains(Point(p)) for p in pts): rep['hata'].append(f'ayırıcı {len(divs)} açıklıkta köşe/uç yapıyor')
            divs.append(pts)
        regions = [r for r in polygonize(unary_union([FP.exterior] + [LineString(d) for d in divs])) if FP.buffer(.5).covers(r) and r.area > .2 * ppm * ppm]
        seeds = P.get('bolumler', [])
        assign = {}
        for s in seeds:
            hit = [j for j, r in enumerate(regions) if r.contains(Point(s['nokta']))]
            if len(hit) != 1: rep['hata'].append(f"{s.get('no', s.get('tip'))}: tohum bir bölgeye düşmüyor"); continue
            if hit[0] in assign: rep['hata'].append(f"{s.get('no')}: bölge zaten {assign[hit[0]].get('no')} için"); continue
            assign[hit[0]] = s
        empty = [j for j in range(len(regions)) if j not in assign]
        if empty: rep['hata'].append(f'{len(empty)} bölge atanmadı: ' + str([list(regions[j].representative_point().coords)[0] for j in empty]))
        rep['bolge'] = len(regions); rep['ayirici'] = len(divs)
        save(self.work / (Path(plan_path).stem + '_bolum_hazir.json'), {'ayiricilar': divs, 'bolgeler': [list(r.exterior.coords) for r in regions]})
        if kuru or rep['hata']: rep['yazildi'] = False; return rep
        before = {e['id'] for e in self.s['elements']}; self.backup(Path(plan_path).stem + '_BOLUM_ONCE')
        try:
            self.view(W(list(FP.centroid.coords)[0]), 40)
            for d in divs:
                self.city('bagimsizbolumsiniri')
                for p in d: self.click(W(p))
                self.stop()
            for j, s in assign.items():
                c = list(polylabel(regions[j], 1.0).coords)[0]
                if s.get('tip', 'bb') == 'bb':
                    blok = s.get('blok', P.get('blok', '')); giris = s.get('giris', P.get('giris', ''))
                    self.city('bagimsizbolumalani', params=[blok, giris, str(s['no']), s.get('brut', ''), s.get('net', ''), ''])
                    self.click(W(c)); self.click(W(s.get('etiket', c))); self.stop()
                else:
                    self.city('ortakalan'); self.click(W(c)); self.click(W(c)); self.stop()
            self.f.Save(); after = self.refresh(); new = [e for e in after['elements'] if e['id'] not in before]
            areas = [Polygon([p[:2] for p in e['points']]).buffer(0) for e in new if e['level'] in ('BagimsizBolumAlani', 'OrtakAlan', 'BagimsizBolumOrtakAlani')]
            if len(areas) != len(assign): raise RuntimeError(f'{len(areas)} alan oluştu, beklenen {len(assign)}')
            fw = Polygon([rm.xy(*p) for p in FP.exterior.coords])
            u = unary_union(areas); gap = u.symmetric_difference(fw).area; ov = sum(a.area for a in areas) - u.area
            if gap > .05 or ov > .05: raise RuntimeError(f'Alanlar katı bölmüyor: boşluk/fazla {gap:.3f} m², örtüşme {ov:.3f} m²')
            for e in new:
                if e['level'] == 'BagimsizBolumNo':
                    base = next(x for x in after['elements'] if x['id'] == e['base']); pol = Polygon([p[:2] for p in base['points']])
                    st = _tag_state(self.m, e['id'])['placement']
                    if not pol.buffer(-.1).covers(Point(*st[:2])):
                        c = pol.representative_point(); t = self.m.GetElementByID64(e['id']).AsTagElement; t.Move(self.pt(c.x - st[0], c.y - st[1], 0)); t.Rewrite()
            rep['etiket'] = self.etiket_sigdir({e['id'] for e in new})
            self.f.Save(); self.refresh(); rep['yazildi'] = True; rep['yeni_oge'] = len(new)
            save(self.work / (Path(plan_path).stem + '_bolum_yazildi.json'), {'idler': [e['id'] for e in new]})
        except Exception as exc:
            self.stop(); self.rollback(before); self.refresh(); rep['hata'].append('geri alındı: ' + repr(exc)); rep['yazildi'] = False
        return rep

    def bolum_turet(self, plan_path, raster, sahip, cikti, uzat_m=.3, sadelestir=1.5):
        """F11/F12 tohumlarını oda sahipliğinden HESAPLAR (model çizgi çizmez).
        sahip: {oda anahtarı ön eki ya da tam anahtar: BB no | 'ortak'}. Kat alanının her pikseli en yakın odanın
        sahibine verilir; farklı sahiplerin sınırı duvar ortasından geçer -> F11. Aynı sahibin bitişik odaları tek bölge,
        her bölgeye tek tohum (BB no ya da ortak). Şaftlar (N.O.A) ayrı ortak bölge olur, F11 etrafından dolaşır.
        Sınıra değen çizgiler kat dışına uzatılır. Çıktı: bolumler() ile yazılabilen plan dosyası."""
        import cv2
        from scipy import ndimage
        from skimage.measure import find_contours
        from shapely.ops import linemerge
        self.refresh(); ppm = self.ppm(raster)
        # oda kimliği -> anahtar: bu klasördeki bütün yazım kayıtları
        keyof = {}
        for f in self.work.glob('*_yazildi.json'):
            if f.name.endswith(('_bolum_yazildi.json', '_balkon_yazildi.json')): continue
            for k, v in (rd(f)['idler'].get('odalar') or {}).items(): keyof[v] = k
        E = self.elements_on(raster)
        fl = [e for e in E if e['level'] == 'KatAlani']
        if len(fl) != 1: raise RuntimeError('Kat alanı tek değil')
        FP = Polygon(fl[0]['px']); x0, y0, x1, y1 = [int(v) for v in FP.bounds]; x0 -= 5; y0 -= 5; W_, H_ = x1 - x0 + 6, y1 - y0 + 6
        def owner(k):
            if k in sahip: return sahip[k]
            pre = max((p for p in sahip if k.startswith(p)), key=len, default=None)
            return sahip[pre] if pre else None
        lab = np.zeros((H_, W_), np.int32); owners = {}; missing = []
        rooms = [e for e in E if e['level'] == 'OdaAlani' and FP.buffer(2).covers(Polygon(e['px']).representative_point())]
        for e in rooms:
            k = keyof.get(e['id']); o = owner(k) if k else None
            if o is None: missing.append(k or e['id']); continue
            code = owners.setdefault(o, len(owners) + 1)
            cv2.fillPoly(lab, [np.round(np.array(e['px']) - [x0, y0]).astype(np.int32)], int(code))
        if missing: raise RuntimeError(f'Sahibi verilmemiş odalar: {missing}')
        inside = np.zeros((H_, W_), np.uint8); cv2.fillPoly(inside, [np.round(np.array(fl[0]['px']) - [x0, y0]).astype(np.int32)], 1)
        _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        near = lab[iy, ix] * inside
        inv = {v: k for k, v in owners.items()}; regions = []
        reg = np.zeros_like(near); rid = 0; rowner = {}
        for o_, code in owners.items():
            n, cc = cv2.connectedComponents((near == code).astype(np.uint8), connectivity=4)
            for c in range(1, n):
                m = cc == c
                if m.sum() < .1 * ppm * ppm: reg[m] = -1; continue   # çok küçük parça: komşuya katılır
                rid += 1; reg[m] = rid; rowner[rid] = o_
        if (reg == -1).any():  # küçük parçaları en yakın bölgeye ver
            _, (jy, jx) = ndimage.distance_transform_edt(reg <= 0, return_indices=True); fill = reg[jy, jx]; reg = np.where(reg == -1, fill, reg)
        reg = reg * inside; self._reg = (reg, x0, y0, rowner)
        for r_ in range(1, rid + 1):
            m = (reg == r_).astype(np.uint8)
            if not m.any(): continue
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c = max(cs, key=cv2.contourArea).reshape(-1, 2)
            ring = Polygon(c + [x0, y0]).buffer(0)
            if ring.geom_type == 'MultiPolygon': ring = max(ring.geoms, key=lambda g: g.area)
            regions.append((rowner[r_], ring))
        # sınır: farklı bölge etiketli komşu pikseller arasındaki birim kenarlar (doğal olarak yatay/dikey)
        edges = []
        dx = (reg[:, :-1] != reg[:, 1:]) & (reg[:, :-1] > 0) & (reg[:, 1:] > 0)
        for yy, xx in zip(*np.nonzero(dx)): edges.append(((x0 + xx + 1, y0 + yy), (x0 + xx + 1, y0 + yy + 1)))
        dy = (reg[:-1, :] != reg[1:, :]) & (reg[:-1, :] > 0) & (reg[1:, :] > 0)
        for yy, xx in zip(*np.nonzero(dy)): edges.append(((x0 + xx, y0 + yy + 1), (x0 + xx + 1, y0 + yy + 1)))
        # kat alanı sınırına değen sahip sınırları da (dış uç) için: dış (0) ile iç arasındaki kenarlar kullanılmaz
        net = linemerge(unary_union([LineString(e) for e in edges]))
        segs = list(net.geoms) if hasattr(net, 'geoms') else [net]
        P0 = rd(plan_path)
        def rsimp(c, step=4):
            c = [list(map(float, q)) for q in c]
            def collinear(c):
                out = [c[0]]
                for q in c[1:]:
                    if len(out) >= 2 and ((abs(out[-2][0] - out[-1][0]) < 1e-9 and abs(out[-1][0] - q[0]) < 1e-9) or (abs(out[-2][1] - out[-1][1]) < 1e-9 and abs(out[-1][1] - q[1]) < 1e-9)): out[-1] = q
                    else: out.append(q)
                return out
            c = collinear(c)
            for _ in range(2000):
                if len(c) < 3: break
                L_ = [abs(c[i + 1][0] - c[i][0]) + abs(c[i + 1][1] - c[i][1]) for i in range(len(c) - 1)]
                i = int(np.argmin(L_))
                if L_[i] > step: break
                if 0 < i < len(c) - 2:  # iç basamak: kısa komşuyu uzun komşunun hizasına çek
                    hor = abs(c[i][1] - c[i + 1][1]) < 1e-9  # basamak yatay mı
                    li, ln = L_[i - 1], L_[i + 1]
                    if hor:  # basamak yatay -> komşular dikey: x'leri eşitle
                        if li >= ln: c[i + 1][0] = c[i][0]; c[i + 2][0] = c[i][0]
                        else: c[i][0] = c[i + 1][0]; c[i - 1][0] = c[i + 1][0]
                    else:
                        if li >= ln: c[i + 1][1] = c[i][1]; c[i + 2][1] = c[i][1]
                        else: c[i][1] = c[i + 1][1]; c[i - 1][1] = c[i + 1][1]
                elif i == 0: c = c[1:]
                else: c = c[:-1]
                c = collinear(c)
            return [tuple(q) for q in c]
        raw = [rsimp(list(g.coords)) for g in segs if g.length >= 6]
        # düğüm yakalama: 6 px içindeki uçlar tek noktada, kalan serbest uç en yakın çizgiye
        ends = [(i, j) for i in range(len(raw)) for j in (0, -1)]
        for a_i, (i, j) in enumerate(ends):
            p0 = np.array(raw[i][j]); grp = [(k, l) for k, l in ends if np.linalg.norm(np.array(raw[k][l]) - p0) < 6]
            if len(grp) > 1:
                m_ = np.mean([raw[k][l] for k, l in grp], 0)
                for k, l in grp: raw[k][l] = tuple(m_)
        for i, j in ends:
            p0 = Point(raw[i][j])
            if FP.exterior.distance(p0) < 3: continue
            others = [LineString(raw[k]) for k in range(len(raw)) if k != i]
            if not others or min(o.distance(p0) for o in others) < .05: continue
            o = min(others, key=lambda o: o.distance(p0))
            # önce kendi doğrultusunda uzat (dik kalsın); bulunamazsa 32 px içindeki en yakın noktaya
            a_ = np.array(raw[i][j]); b_ = np.array(raw[i][1] if j == 0 else raw[i][-2]); v_ = (a_ - b_) / max(np.linalg.norm(a_ - b_), 1e-9)
            ray = LineString([tuple(a_ - v_ * 2), tuple(a_ + v_ * 400)]); hits = []
            for g_ in others + [FP.exterior]:
                x_ = ray.intersection(g_)
                if not x_.is_empty: hits += [pt for pt in (x_.geoms if hasattr(x_, 'geoms') else [x_]) if pt.geom_type == 'Point']
            hits = [h_ for h_ in hits if h_.distance(Point(a_)) > .05]
            if hits and min(h_.distance(Point(a_)) for h_ in hits) < 400: h_ = min(hits, key=lambda pt: pt.distance(Point(a_))); raw[i][j] = (h_.x, h_.y)
            elif o.distance(p0) < 32: q = nearest_points(o, p0)[0]; raw[i][j] = (q.x, q.y)
        # kavşak hizası: uçtaki parça yatayımsıysa önceki köşenin y'si, dikeyimsiyse x'i kavşağa eşitlenir (tam dik kalır)
        for i in range(len(raw)):
            raw[i] = [list(q) for q in raw[i]]
            for j, k in ((0, 1), (-1, -2)):
                if len(raw[i]) < 2: continue
                e_, q_ = raw[i][j], raw[i][k]; dx_, dy_ = abs(e_[0] - q_[0]), abs(e_[1] - q_[1])
                if 0 < min(dx_, dy_) <= 12 and len(raw[i]) >= 3:
                    if dx_ >= dy_: raw[i][k][1] = e_[1]
                    else: raw[i][k][0] = e_[0]
            raw[i] = [tuple(q) for q in raw[i]]
        # balkonlar: plandaki açık balkon çizgileri kat kenarıyla kapatılır; F11 ucu balkona girmez
        balk = [Polygon(e['px']).buffer(0) for e in E if e['level'] == 'BalkonAlani']   # DGN'deki gerçek balkon alanları
        if not balk:
            for bk in P0.get('balkonlar', []):
                g_ = _balkon_poligon(bk['nokta'], FP)
                if g_ is not None and g_.area > 0: balk.append(g_)
        Bu = unary_union([g_.buffer(-.3) for g_ in balk]) if balk else None   # yalnız İÇ alanlar: iki balkonun arası serbest
        def balkon_yonlendir(c, end):
            p = Point(c[end])
            if Bu is None or Bu.is_empty or FP.exterior.distance(p) > 3: return c
            q = np.array(c[end]); r = np.array(c[1] if end == 0 else c[-2]); v = (q - r) / np.linalg.norm(q - r)
            if not any(Bu.contains(Point(*(q + k * v))) for k in (4, 10, 20, 40)): return c
            ex = FP.exterior; t0 = ex.project(p); best = None; Lx = ex.length
            for dt in np.arange(2, 600, 2):
                for sgn in (1, -1):
                    pp = ex.interpolate((t0 + sgn * dt) % Lx)
                    if not any(Bu.contains(Point(pp.x + k * v[0], pp.y + k * v[1])) for k in (4, 10, 20, 40)):
                        best = sgn * dt; break
                if best is not None: break
            if best is None: return c
            pts_ = [ex.interpolate((t0 + best * k / 8) % Lx) for k in range(1, 9)]; pp = pts_[-1]
            out_pt = (pp.x + 3 * v[0], pp.y + 3 * v[1])  # balkonların arasından dışarı
            path = [(pt.x, pt.y) for pt in pts_] + [out_pt]
            return (path[::-1] + c) if end == 0 else (c + path)
        divs = []
        for c in raw:
            c = list(c)
            c = balkon_yonlendir(balkon_yonlendir(c, 0), -1)
            for end, nxt in ((0, 1), (-1, -2)):
                p = np.array(c[end]); q = np.array(c[nxt]); v = p - q; v = v / np.linalg.norm(v)
                if FP.exterior.distance(Point(p)) < 3:
                    k = 0
                    while FP.buffer(1).contains(Point(*(p + k * v))) and k < 200: k += 1
                    c[end] = tuple(p + (k + uzat_m * ppm) * v)
            cc_ = []
            for a, b in c:
                q_ = [round(float(a), 1), round(float(b), 1)]
                if not cc_ or math.dist(q_, cc_[-1]) >= 1.5: cc_.append(q_)
            if len(cc_) >= 3:  # doğrusal ara köşeleri at
                k2 = [cc_[0]]
                for i_ in range(1, len(cc_) - 1):
                    a_, b_, d_ = np.array(k2[-1]), np.array(cc_[i_]), np.array(cc_[i_ + 1])
                    if LineString([a_, d_]).distance(Point(b_)) > .6: k2.append(cc_[i_])
                k2.append(cc_[-1]); cc_ = k2
            divs.append(cc_)
        seeds = []
        for o, ring in regions:
            pt = list(polylabel(ring, 1.0).coords)[0]
            seeds.append({'tip': 'ortak', 'nokta': [round(pt[0]), round(pt[1])]} if o == 'ortak' else {'tip': 'bb', 'no': o, 'nokta': [round(pt[0]), round(pt[1])]})
        P = rd(plan_path); P['ayiricilar'] = divs; P['bolumler'] = seeds; P['_turetildi'] = {'bolge': len(regions), 'ayirici': len(divs)}
        save(cikti, P); return {'bolge': len(regions), 'ayirici': len(divs), 'cikti': str(cikti), 'sahipsiz': missing}

    # ---------------- balkonlar (F9/F10)
    def balkonlar(self, plan_path, kuru=False):
        P = rd(plan_path); i = P['raster']; rm = self.rmap(i); W = lambda p: rm.xy(*p); ppm = self.ppm(i); self.refresh()
        fl = [e for e in self.elements_on(i, {'KatAlani'}) if Polygon(e['px']).intersects(Polygon(P['kat_alani']))]
        if len(fl) != 1: raise RuntimeError('Kat alanı bulunamadı')
        FP = Polygon(fl[0]['px']); rep = {'hata': [], 'uyari': [], 'balkon': []}
        for b in P.get('balkonlar', []):
            pts = [list(map(float, p)) for p in b['nokta']]
            for end in (0, -1):
                q = nearest_points(FP.exterior, Point(pts[end]))[0]
                if q.distance(Point(pts[end])) > .15 * ppm: rep['hata'].append(f"{b['anahtar']}: uç kat alanına {q.distance(Point(pts[end]))/ppm*100:.0f} cm uzak"); continue
                # kat alanı köşesine 3 px'ten yakınsa köşeyi al
                cs = [Point(c) for c in FP.exterior.coords]; cn = min(cs, key=lambda c: c.distance(Point(pts[end])))
                q = cn if cn.distance(Point(pts[end])) < 3 else q
                pts[end] = [q.x, q.y]
            ring = _balkon_poligon(pts, FP)
            if ring is None or ring.area <= 1: rep['hata'].append(f"{b['anahtar']}: balkon kapalı alan oluşturmuyor"); continue
            if ring.intersection(FP.buffer(-1.5)).area > 4: rep['hata'].append(f"{b['anahtar']}: balkon kat alanına giriyor ({ring.intersection(FP).area:.0f} px²)")
            m2 = b.get('m2') or round(ring.area / ppm / ppm, 2)
            c_pt = list(polylabel(ring, 1.0).coords)[0]
            rep['balkon'].append({'anahtar': b['anahtar'], 'nokta': pts, 'm2': m2, 'cizim_m2': round(ring.area / ppm / ppm, 2), 'ic_nokta': [round(c_pt[0], 2), round(c_pt[1], 2)]})
        if kuru or rep['hata']: rep['yazildi'] = False; return rep
        before = {e['id'] for e in self.s['elements']}; self.backup(Path(plan_path).stem + '_BALKON_ONCE')
        try:
            for b, spec in zip(rep['balkon'], P['balkonlar']):
                world = [W(p) for p in b['nokta']]; e = self.add('balcony_boundary', world, closed=False)
                c = Point(*b['ic_nokta']); self.view(W((c.x, c.y)), 16)
                self.q.SendKeyin('citymax islem cizim balkonduvari'); time.sleep(.15)
                if int(self.a.GetCExpressionValue('gCityMax.Islem', 'CITYMAX')) != 107: raise RuntimeError('F10 kodu değil')
                for k_, v in {'Secenek1': 5, 'Secenek2': 1, 'Parametre1': str(b['m2']), 'Parametre2': str(spec.get('yukseklik', 110)), 'Parametre3': str(spec.get('kalinlik', 10))}.items():
                    self.a.SetCExpressionValue('gCityMax.' + k_, v, 'CITYMAX')
                self.q.SendKeyin('citymax baslat'); time.sleep(.15)
                mid = ((world[1][0] + world[2][0]) / 2, (world[1][1] + world[2][1]) / 2) if len(world) > 2 else ((world[0][0] + world[1][0]) / 2, (world[0][1] + world[1][1]) / 2)
                self.click(mid); self.click(W((c.x, c.y))); self.stop()
                if spec.get('ad', 'BALKON') != 'BALKON':
                    nw = self.refresh()
                    for t in nw['elements']:
                        if t['id'] not in before and t.get('level') == 'BalkonAdi': tt = self.m.GetElementByID64(t['id']).AsTagElement; tt.Value = spec['ad']; tt.Rewrite()
            self.f.Save(); after = self.refresh(); new = [e for e in after['elements'] if e['id'] not in before]
            na = sum(e['level'] == 'BalkonAlani' for e in new); nw_ = sum(e['level'] == 'BalkonDuvari' for e in new)
            if na != len(rep['balkon']) or nw_ != len(rep['balkon']): raise RuntimeError(f'{na} balkon alanı / {nw_} duvar oluştu, beklenen {len(rep["balkon"])}')
            rep['etiket'] = self.etiket_sigdir({e['id'] for e in new}); rep['yazildi'] = True
            recb = self.work / (Path(plan_path).stem + '_balkon_yazildi.json'); prev = rd(recb)['idler'] if recb.exists() else []
            save(recb, {'idler': prev + [e['id'] for e in new if e['type'] != 37]})
        except Exception as exc:
            self.stop(); self.rollback(before); self.refresh(); rep['hata'].append('geri alındı: ' + repr(exc)); rep['yazildi'] = False
        return rep

    # ---------------- kat kopyalama (native seçim + COPY / MOVE)
    def kopyala(self, ids, vector, move=False):
        """ids: kopyalanacak öğe kimlikleri (raster eleman kimlikleri dahil olabilir). vector: dünya metre.
        Tag'ler sahip öğeyle birlikte gider. Native COPY ICON / MOVE ICON kullanılır."""
        before = self.refresh(); old = {e['id'] for e in before['elements']}
        self.a.CommandState.StartDefaultCommand(); self.m.UnselectAllElements()
        for eid in ids: self.m.SelectElement(self.m.GetElementByID64(eid))
        self.q.SendKeyin('move icon' if move else 'copy icon'); time.sleep(.2)
        self.q.SendDataPoint(self.pt(0, 0, 0), 1); time.sleep(.15); self.q.SendDataPoint(self.pt(vector[0], vector[1], 0), 1); time.sleep(.3)
        self.stop(); self.m.UnselectAllElements(); self.f.Save(); after = self.refresh()
        new = [e for e in after['elements'] if e['id'] not in old]
        return {'yeni': [e['id'] for e in new], 'sayi': len(new)}

    def raster_eleman(self, i):
        """i. raster bağlantısının (tip 94) öğe kimliği: aralığı rasterin kayıtlı başlangıcıyla eşleşen öğe."""
        o = self.s['rasters'][i - 1]['origin']
        for e in self.s['elements']:
            if e['type'] == 94:
                r = self.m.GetElementByID64(e['id']).Range
                if abs(r.Low.X - o[0]) < .05 and abs(r.Low.Y - o[1]) < .05: return e['id']
        raise RuntimeError(f'{i}. rasterin öğesi bulunamadı')

    def pafta_kopyala(self, kaynak_raster, ids, aks_id, hedef_kolon_px, dy=-42.0, ad='kopya'):
        """Kılavuz yöntemi (24.10.2026): kaynak katın rasteri + verilen çizimler TEK seçimle COPY ICON ile dy kadar
        aşağı kopyalanır (raster kopyası sabit kalır). Sonra yalnız kopyalanan çizimler, kopyalanan bina aksının
        merkezinden hedef paftadaki aynı kolonun merkezine (yeni rasterde, kutu_oturt ile ölçülen) tek MOVE ICON ile
        taşınır. Geri okuma: kaynak değişmedi, raster kayıtlı dönüşümü korundu, bütün öğeler ve Tag'ler aynı
        ötelemeyi aldı. Dönüş: işlem kaydı (vektörler, kimlikler, artıklar)."""
        self.refresh(); s0 = {e['id']: e for e in self.s['elements']}; self.backup(f'{ad}_ONCE')
        rid = self.raster_eleman(kaynak_raster); src_r = self.s['rasters'][kaynak_raster - 1]
        tag_ids = [e['id'] for e in self.s['elements'] if e.get('base') in set(ids)]
        log = {'ad': ad, 'kaynak_raster': kaynak_raster, 'raster_oge': rid, 'secim': len(ids) + 1, 'tag': len(tag_ids), 'kopya_vektor_m': [0.0, dy]}
        c = self.kopyala([rid] + list(ids), [0.0, dy]); new = set(c['yeni'])
        s1 = self.refresh()
        if any(s0[k] != e for e in s1['elements'] if (k := e['id']) in s0): raise RuntimeError('Kopyalama kaynak öğeleri değiştirdi')
        ni = next(i for i, r in enumerate(s1['rasters'], 1) if abs(r['origin'][1] - (src_r['origin'][1] + dy)) < .01 and abs(r['origin'][0] - src_r['origin'][0]) < .01)
        nr = s1['rasters'][ni - 1]
        if [round(v, 6) for v in nr['extent']] != [round(v, 6) for v in src_r['extent']] or abs(nr['rotation'] - src_r['rotation']) > 1e-9: raise RuntimeError('Raster ölçeği/dönüşü değişti')
        log['yeni_raster'] = ni; log['kopyalanan'] = len(new)
        # kopyalanan aks: kaynak aksla aynı geometri + dy
        ax0 = s0[aks_id]['points']; cand = [e for e in s1['elements'] if e['id'] in new and e['level'] == 'BinaAksi']
        ax = next(e for e in cand if all(abs(p[0] - q[0]) < 1e-6 and abs(p[1] + dy - q[1]) < 1e-6 for p, q in zip(ax0, e['points'])))
        axc = np.mean(np.array([p[:2] for p in ax['points'][:4]]), 0)
        if len(hedef_kolon_px) == 2:  # hedefte aks kolonu görünmüyorsa (ör. ÇK terası): başka kolonlardan hesaplanmış merkez
            kc = np.array(hedef_kolon_px, float); kb = [list(kc)]
        else:
            ink = self.kaynak(ni).inkmap(); kb = kutu_oturt(ink, hedef_kolon_px); kc = np.mean(np.array(kb), 0)
        tw = np.array(self.rmap(ni).xy(*kc)); mv = (tw - axc).tolist()
        log.update(hedef_kolon_px=[[round(v, 1) for v in p] for p in kb], hedef_kolon_merkez_px=[round(float(v), 2) for v in kc], tasima_vektor_m=[round(v, 4) for v in mv])
        draw = [k for k in new if s1 and next(e for e in s1['elements'] if e['id'] == k)['type'] != 94 and next(e for e in s1['elements'] if e['id'] == k)['type'] != 37]
        before_move = {e['id']: e for e in s1['elements']}
        self.kopyala(draw, mv, move=True); s2 = self.refresh()
        # doğrulama
        err = 0.0
        for k in draw:
            a_ = before_move[k]['points']; b_ = next(e for e in s2['elements'] if e['id'] == k)['points']
            err = max(err, max(math.dist([p[0] + mv[0], p[1] + mv[1]], q[:2]) for p, q in zip(a_, b_)))
        moved_tags = [e for e in s2['elements'] if e['id'] in new and e['type'] == 37]
        if err > 1e-6: raise RuntimeError(f'Taşıma tutarsız ({err} m)')
        if any(s0[k] != e for e in s2['elements'] if (k := e['id']) in s0): raise RuntimeError('Taşıma kaynak öğeleri değiştirdi')
        if s2['rasters'][ni - 1] != nr: raise RuntimeError('Yeni raster taşındı')
        ax2 = next(e for e in s2['elements'] if e['id'] == ax['id']); c2 = self.rmap(ni).pixels(*np.mean(np.array([p[:2] for p in ax2['points'][:4]]), 0))
        log.update(tasinan=len(draw), tasinan_tag=len(moved_tags), en_buyuk_oteleme_hatasi_m=err, aks_merkez_px=[round(v, 2) for v in c2],
                   aks_kolon_artik_cm=round(100 * math.dist(self.rmap(ni).xy(*c2), self.rmap(ni).xy(*kc)), 3), yeni_idler=sorted(new))
        save(self.work / f'{ad}_ISLEM.json', log); return log

    # ---------------- kapanış
    def kat_islem(self, run):
        """Yalnız Citymax Kat Kontrol, açık dosyanın kendisinde (kullanıcı tercihi, 2026-11-19/20). Kat Yükseltme ve
        Birleştirme'yi kullanıcı kendisi yapar; motorda yoktur.
        Not: HATA işaret katmanlarının ADLARI canlı oturumda kayık okunabiliyor (2026-11-20: dosya yeniden açılınca
        aynı sayılar başka adlarla geldi). İşaretin anlamını logdan ve konumundan doğrula, katman adına güvenme."""
        src = Path(self.s['dgn']).resolve(); self.f.Save(); out = {'dosya': str(src)}
        try:
            self.a.CommandState.StartDefaultCommand(); self.q.SendKeyin('citymax islem kat kontrol'); time.sleep(.3)
            if int(self.a.GetCExpressionValue('gCityMax.Islem', 'CITYMAX')) != 210: raise RuntimeError('kontrol kodu yanlış')
            for k_, v in {'Secenek1': 1, 'Secenek2': 1, 'Parametre1': '10', 'Parametre2': '10'}.items(): self.a.SetCExpressionValue('gCityMax.' + k_, v, 'CITYMAX')
            self.q.SendKeyin('citymax baslat'); time.sleep(1.5); self.q.SendReset(); self.a.CommandState.StartDefaultCommand()
            out['kontrol'] = True
            log = src.with_suffix('.log')
            if log.exists(): shutil.copy2(log, self.work / f'{run}.log'); out['log'] = str(self.work / f'{run}.log'); out['ozet'] = log_coz(log)
            after = snapshot(self.a); from collections import Counter
            out['hata_isaretleri'] = dict(Counter(e['level'] for e in after['elements'] if e['level'].upper().startswith('HATA')))
        finally:
            self.refresh()
        return out


def log_coz(path):
    """Citymax kat kontrol logunu kat > bölüm > oda > HATA/UYARI ağacına çevirir."""
    floor = None; unit = None; room = None; rows = []
    raw = Path(path).read_bytes()
    try: text = raw.decode('utf-8')
    except UnicodeDecodeError: text = raw.decode('cp1254', errors='replace')  # Citymax logu Windows Türkçe
    for line in text.splitlines():
        s = line.strip()
        if m := re.match(r'\[(\d+)\] kat', s): floor = {'kat': int(m.group(1))}; unit = room = None
        elif s.startswith('Kat No') and floor is not None: floor['ad'] = s.split('=')[1].strip()
        elif s.startswith('Bağımsız Bölüm No') or s.startswith('Ortak Alan Adı'): unit = s.split('=')[1].strip(); room = unit if s.startswith('Ortak') else None
        elif s.startswith('Oda Adı'): room = s.split('=')[1].strip()
        elif '>HATA' in s or '>UYARI' in s:
            rows.append({'kat': (floor or {}).get('ad'), 'bolum': unit, 'oda': room, 'tur': 'HATA' if '>HATA' in s else 'UYARI', 'mesaj': s.split(':', 1)[1].strip()})
    return {'hata': [r for r in rows if r['tur'] == 'HATA'], 'uyari_sayisi': sum(r['tur'] == 'UYARI' for r in rows), 'uyari_turleri': sorted({r['mesaj'] for r in rows if r['tur'] == 'UYARI'})}
