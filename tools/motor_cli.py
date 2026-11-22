"""Citymax motoru v2 komut satırı. Bütün koordinatlar bağlı TIF'in özgün pikselidir.
  durum   --is W                                  açık DGN, rasterler, kat başına öğe sayısı
  gor     --is W --raster i --kutu x0 y0 x1 y1 [--olcek 1] [--plan P] [--dgn] [--cikti f.png]
  olc     --is W --raster i --a x y --b x y        iki nokta arası cm
  oturt   --is W --raster i --plan P               odaları yüze oturtur, rapor (yazmaz)
  kat     --is W --plan P [--kuru] [--adim cerceve,kat,oda,merdiven,acilik]
  sil     --is W --plan P                          yalnız bu planın yazdığı kimlikler
  bolum   --is W --plan P [--kuru]                 F11 + F12 + ortak alan
  balkon  --is W --plan P [--kuru]                 F9 + F10
  kopyala --is W --ids ... --vektor dx dy [--tasi]
  kontrol --is W --ad RUN                         açık dosyada Kat Kontrol, log özeti (yükseltme/birleştirme kullanıcıda)
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, json, sys, time
from pathlib import Path
from citymax import motor as M  # noqa: E402

LOG = None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd'); p.add_argument('--is', dest='work', required=True); p.add_argument('--raster', type=int)
    p.add_argument('--kutu', nargs=4, type=float); p.add_argument('--olcek', type=float, default=1.0); p.add_argument('--plan')
    p.add_argument('--dgn', action='store_true'); p.add_argument('--cikti'); p.add_argument('--a', nargs=2, type=float); p.add_argument('--b', nargs=2, type=float)
    p.add_argument('--kuru', action='store_true'); p.add_argument('--adim'); p.add_argument('--ids', nargs='+', type=int); p.add_argument('--vektor', nargs=2, type=float)
    p.add_argument('--tasi', action='store_true'); p.add_argument('--ad'); p.add_argument('--grid', type=int, default=100)
    p.add_argument('--kutular'); p.add_argument('--merkezler'); p.add_argument('--yaricap', type=float, default=60); p.add_argument('--sutun', type=int, default=4)
    p.add_argument('--satir'); p.add_argument('--sutun-x', dest='sutunx'); p.add_argument('--aralik', nargs=2, type=float); p.add_argument('--bant', type=int, default=0)
    p.add_argument('--tohumlar'); p.add_argument('--genislik', type=float, default=40)
    p.add_argument('--dikey', nargs=3, type=float); p.add_argument('--yatay', nargs=3, type=float)
    a = p.parse_args(); t0 = time.time(); mo = M.Motor(a.work)
    if a.cmd == 'durum':
        s = mo.s; r = {'dgn': s['dgn'], 'model': s['model_name'], 'oge': len(s['elements']), 'rasterler': []}
        for i in range(1, len(s['rasters']) + 1):
            from collections import Counter
            r['rasterler'].append({'i': i, 'y0': round(s['rasters'][i - 1]['origin'][1], 2), 'px_m': round(mo.ppm(i), 3), 'ogeler': dict(Counter(e['level'] for e in mo.elements_on(i)))})
        try: r['citymax_islem'] = mo.a.GetCExpressionValue('gCityMax.Islem', 'CITYMAX')
        except Exception: r['citymax_islem'] = 'YÜKLÜ DEĞİL (mdl load CITYMAX)'
    elif a.cmd == 'gor':
        k = mo.kaynak(a.raster); ov = []
        if a.dgn:
            col = {'OdaAlani': '#0050ff', 'KatAlani': '#0a9a3c', 'KapiAcikligi': '#ff6a00', 'PencereAcikligi': '#00a5b5', 'KatPlaniSiniri': '#999999',
                   'BagimsizBolumSiniri': '#c000c0', 'BalkonSiniri': '#8a5a00', 'OdaIcYapi': '#aa0000', 'BagimsizBolumAlani': '#c000c0', 'OrtakAlan': '#0a7a6a'}
            ov += [(e['px'], e['type'] == 6, col.get(e['level'], '#666'), None) for e in mo.elements_on(a.raster) if e['level'] in col]
        if a.plan:
            P = M.rd(a.plan)
            for r in P.get('odalar', []): ov.append((r['nokta'], True, '#e0005a', r['anahtar']))
            if P.get('kat_alani'): ov.append((P['kat_alani'], True, '#00a000', None))
            for o in P.get('acikliklar', []): ov.append((o['kasa'], False, '#ff00ff', None))
            for d in P.get('ayiricilar', []): ov.append((d, False, '#c000c0', None))
            for b in P.get('balkonlar', []): ov.append((b['nokta'], False, '#8a5a00', b['anahtar']))
        out = a.cikti or str(Path(a.work) / 'gor' / f"{a.raster}_{'_'.join(str(int(v)) for v in a.kutu)}_{a.olcek}.png")
        Path(out).parent.mkdir(parents=True, exist_ok=True); M.gorunum(k, a.kutu, a.olcek, ov, out, a.grid); r = {'resim': out}
    elif a.cmd == 'pano':
        # --kutular "x0,y0,x1,y1;x0,y0,x1,y1;..." veya --merkezler "x,y;x,y" (--yaricap r)
        boxes = []
        if a.kutular: boxes = [list(map(float, s.split(','))) for s in a.kutular.split(';') if s]
        if a.merkezler: boxes += [[x - a.yaricap, y - a.yaricap, x + a.yaricap, y + a.yaricap] for x, y in (map(float, s.split(',')) for s in a.merkezler.split(';') if s)]
        k = mo.kaynak(a.raster); ov = []
        if a.plan:
            P = M.rd(a.plan)
            for r in P.get('odalar', []): ov.append((r['nokta'], True, '#e0005a', None))
            if P.get('kat_alani'): ov.append((P['kat_alani'], True, '#00a000', None))
        if a.dgn: ov += [(e['px'], e['type'] == 6, '#0050ff', None) for e in mo.elements_on(a.raster) if e['level'] in ('OdaAlani', 'KatAlani', 'KapiAcikligi', 'PencereAcikligi')]
        out = a.cikti or str(Path(a.work) / 'gor' / f'pano_{int(time.time())}.png'); Path(out).parent.mkdir(parents=True, exist_ok=True)
        M.pano(k, boxes, a.olcek if a.olcek != 1.0 else 2.0, ov, out, a.sutun); r = {'resim': out, 'kutu': len(boxes)}
    elif a.cmd == 'kesit':
        # --satir y --aralik x0 x1   ya da   --sutun-x x --aralik y0 y1 ; birden çok değer: --satir 3500,3600
        g = mo.kaynak(a.raster).g; r = {}; band = a.bant
        for v in (a.satir or a.sutunx).split(','):
            v = int(float(v)); lo, hi = map(int, a.aralik)
            line = (g[v - band:v + band + 1, lo:hi] < 128).mean(0) if a.satir else (g[lo:hi, v - band:v + band + 1] < 128).mean(1)
            runs = []; st = None
            for j, on in enumerate(line >= .5):
                if on and st is None: st = j
                if not on and st is not None: runs.append([lo + st, lo + j - 1]); st = None
            if st is not None: runs.append([lo + st, hi - 1])
            r[('y' if a.satir else 'x') + str(v)] = runs
    elif a.cmd == 'egim':
        # --dikey X y0 y1  --yatay Y x0 x1  : paftadaki uzun aks çizgilerine yakın değerler
        g = mo.kaynak(a.raster).g; r = {}
        if a.dikey: d, n, res, cv = M.cizgi_uydur(g, a.dikey[0], 'dikey', a.dikey[1], a.dikey[2]); r['dikey'] = d; r['dikey_ornek'] = n; r['dikey_sapma_px'] = round(res, 2)
        if a.yatay: d, n, res, ch = M.cizgi_uydur(g, a.yatay[0], 'yatay', a.yatay[1], a.yatay[2]); r['yatay'] = d; r['yatay_ornek'] = n; r['yatay_sapma_px'] = round(res, 2)
        if a.dikey and a.yatay:
            import numpy as np
            A = np.array([r['dikey'], [-r['yatay'][0], -r['yatay'][1]]]).T; t = np.linalg.solve(A, np.array(ch) - np.array(cv))
            r['kesisim'] = [round(float(cv[0] + t[0] * r['dikey'][0]), 2), round(float(cv[1] + t[0] * r['dikey'][1]), 2)]
        import math
        if 'dikey' in r: r['dikey_derece'] = round(math.degrees(math.atan2(r['dikey'][0], r['dikey'][1])), 3)
        if 'yatay' in r: r['yatay_derece'] = round(math.degrees(math.atan2(r['yatay'][1], r['yatay'][0])), 3)
    elif a.cmd == 'isin':
        # --tohumlar "x,y;x,y" [--egim-plan P] : her tohum için dört iç yüz
        PP = M.rd(a.plan) if a.plan else {}; ink = mo.ink(a.raster, PP); eg = PP.get('egim'); r = {}
        if PP.get('pafta'): r['silinen_aks'] = mo._temiz[next(iter(mo._temiz))][1]
        for s in a.tohumlar.split(';'):
            x, y = map(float, s.split(',')); res = M.isin(ink, (x, y), eg, a.genislik)
            r[s] = {k: (v.get('nokta'), v['pay'], v.get('en_yakin')) for k, v in res.items()}
    elif a.cmd == 'kolon':
        P = M.rd(a.plan) if a.plan else {}; ink = mo.ink(a.raster, P); cols = M.kolon_bul(ink, a.kutu, P.get('egim'))
        out = a.cikti or str(Path(a.work) / 'gor' / f'kolon_{a.raster}_{int(a.kutu[0])}.png'); Path(out).parent.mkdir(parents=True, exist_ok=True)
        M.gorunum(mo.kaynak(a.raster), a.kutu, a.olcek, [(c['kutu'], True, '#ff0000', str(i)) for i, c in enumerate(cols)], out, a.grid)
        r = {'kolon': len(cols), 'resim': out, 'kutular': {i: [round(c['kutu'][0][0]), round(c['kutu'][0][1]), round(c['kutu'][2][0]), round(c['kutu'][2][1])] for i, c in enumerate(cols)}}
    elif a.cmd == 'olc':
        rm = mo.rmap(a.raster); import math
        r = {'cm': round(100 * math.dist(rm.xy(*a.a), rm.xy(*a.b)), 2), 'px': round(math.dist(a.a, a.b), 2)}
    elif a.cmd == 'oturt':
        P = M.rd(a.plan); ink = mo.ink(P['raster'], P); r = {}
        for o in P.get('odalar', []):
            new, er, mv = M.oturt(ink, o['nokta'], R=P.get('oturt_px', 6)); r[o['anahtar']] = {'oynama_px': round(mv, 1), 'yeni': new, 'kenar': [(e['kenar'], e['durum'], e['kayma_px'], e['guven']) for e in er]}
    elif a.cmd == 'kat': r = mo.kat_yaz(a.plan, tuple(a.adim.split(',')) if a.adim else ('cerceve', 'kat', 'oda', 'merdiven', 'acilik'), a.kuru)
    elif a.cmd == 'sil': r = mo.kat_sil(a.plan, a.adim)
    elif a.cmd == 'bolum': r = mo.bolumler(a.plan, a.kuru)
    elif a.cmd == 'balkon': r = mo.balkonlar(a.plan, a.kuru)
    elif a.cmd == 'etiket': r = mo.etiket_sigdir()
    elif a.cmd == 'kopyala': r = mo.kopyala(a.ids, a.vektor, a.tasi)
    elif a.cmd == 'kontrol': r = mo.kat_islem(a.ad)
    else: p.error('bilinmeyen komut')
    r = {'komut': a.cmd, 'sure_s': round(time.time() - t0, 2), **(r if isinstance(r, dict) else {'sonuc': r})}
    with open(Path(a.work) / 'MOTOR_LOG.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'zaman': time.strftime('%H:%M:%S'), 'argv': sys.argv[1:], 'sonuc': {k: v for k, v in r.items() if k not in ('yeni',)}}, ensure_ascii=False, default=str) + '\n')
    out = json.dumps(r, ensure_ascii=False, default=str)
    (Path(a.work) / 'SON_CIKTI.json').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
