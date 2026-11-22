"""A: RapidOCR (4 rotations, tiles) = the candidates already made for labelling (ADAY_OCR.json)."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
from ortak import *
reads = json.load(open(H.parent / "el_etiket" / "malkoc_k2" / "ADAY_OCR.json", encoding="utf-8"))
r = score_spot(reads, "RapidOCR 4 yon"); json.dump(r, open(H / "A_rapid.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({k: r[k] for k in ("door", "window", "okuma_sayisi")})
