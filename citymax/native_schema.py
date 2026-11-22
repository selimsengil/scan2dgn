"""Fixed Citymax native format mapping; the caller chooses every component type."""
LEVELS={'frame':('KatPlaniSiniri',100,23),'axis':('BinaAksi',101,1),
        'floor':('KatAlani',102,85),'room':('OdaAlani',112,9),
        'door':('KapiAcikligi',122,6),'window':('PencereAcikligi',124,6),
        'balcony_boundary':('BalkonSiniri',116,4),'balcony':('BalkonAlani',117,4),'balcony_wall':('BalkonDuvari',120,4),
        'common_area':('OrtakAlan',131,85),'bb_boundary':('BagimsizBolumSiniri',106,3),
        'bb_area':('BagimsizBolumAlani',107,3)}
OPEN_KINDS={'balcony_boundary','bb_boundary'}
TAGS={'room':('Oda Bilgisi',[('Oda Adı',1,'OdaAdi',113),('Oda M2',4,'OdaM2',114)]),
      'floor':('Kat Bilgisi',[('Kat No',1,'KatNo',103),('Kat Taban Kotu',3,'KatTabanKotu',104),('Kat Tavan Kotu',3,'KatTavanKotu',105)]),
      'door':('Kapı Bilgisi',[('Kapı Yüksekliği',3,'KapiYuksekligi',123)]),
      'window':('Pencere Bilgisi',[('Pencere Yüksekliği',3,'PencereYuksekligi',125),('Zeminden Yüksekliği',3,'ZemindenYuksekligi',126)]),
      'balcony':('Balkon Bilgisi',[('Balkon Adı',1,'BalkonAdi',118),('Balkon M2',4,'BalkonM2',119)]),
      'balcony_wall':('Balkon Bilgisi',[('Balkon Yüksekliği',3,'BalkonYuksekligi',121)]),
      'bb_area':('Bağımsız Bölüm Bilgisi',[('Blok No',1,'BlokNo',108),('Giriş No',1,'GirisNo',109),
          ('Bağımsız Bölüm No',1,'BagimsizBolumNo',110),('Bağımsız Bölüm M2',4,'BagimsizBolumM2',111),
          ('Bağımsız Bölüm Net M2',4,'BagimsizBolumNetM2',135),('MAKS Kimlik No',1,'MAKSKimlikNo',138),
          ('MAKS Numarataj No',1,'MAKSNumaratajNo',141)])}
