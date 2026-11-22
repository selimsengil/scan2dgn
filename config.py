"""Where the data, models and tools live. Nothing here is committed except defaults.

Set the roots in `config.local.json` (git-ignored) next to this file, or with environment variables:

    SCAN2DGN_DATA      folder that holds kat_test/ (extracted floors, labels, outputs)
    SCAN2DGN_MODELS    downloaded OCR/VLM weights
    SCAN2DGN_PROJECTS  original Citymax projects (DGN + TIF), read-only
    SCAN2DGN_USTATION  MicroStation V8i ustation.exe

Every script calls `config.setup()`, which also puts each pipeline folder on sys.path so the stages can import
each other by module name."""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_LOCAL = ROOT / "config.local.json"
_cfg = json.loads(_LOCAL.read_text(encoding="utf-8")) if _LOCAL.exists() else {}


def _get(key, env, default):
    return Path(os.environ.get(env) or _cfg.get(key) or default)


DATA = _get("data", "SCAN2DGN_DATA", ROOT / "data")
KT = DATA / "kat_test"
MODELS = _get("models", "SCAN2DGN_MODELS", ROOT / "models")
PROJECTS = _get("projects", "SCAN2DGN_PROJECTS", ROOT / "projects")
USTATION = _get("ustation", "SCAN2DGN_USTATION", r"C:\Program Files (x86)\Bentley\MicroStation V8i (SELECTseries)\MicroStation\ustation.exe")

STAGES = ("extract", "rooms", "openings", "tags", "tags/ocr_bench", "tools", "experiments")


def setup():
    for p in [ROOT, *(ROOT / s for s in STAGES)]:
        s = str(p)
        if s not in sys.path: sys.path.insert(1, s)
