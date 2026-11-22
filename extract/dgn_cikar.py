"""Reads the user's finished drawings out of their DGN files (MicroStation V8i through its document API).
The original DGN is never opened: the DGN and its TIF are COPIED to kat_test/dgn/<project>/source, MicroStation is
started on the copy, every graphical element (level, vertices, tags) and the raster placement are read, MicroStation
is closed. Nothing is saved, no key-in is sent.   Output: kat_test/dgn/<project>/native_snapshot.json
usage (.venv, PYTHONPATH=src):  python dgn_cikar.py [--only Marmara] [--timeout 180]"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = config.PROJECTS; OUT = config.KT / "dgn"
EXE = str(config.USTATION)


def sha(p):
    with open(p, "rb") as h: return hashlib.file_digest(h, "sha256").hexdigest()


def sources():
    out = []
    for d in sorted(ROOT.glob("Balikesir_*")):
        if not d.is_dir(): continue
        for g in sorted(d.rglob("*.dgn")):
            if g.stem.endswith("_3d"): continue
            out.append(g)
    return out


def read(application, copy):
    import comtypes.client
    file = Path(application.ActiveDesignFile.FullName)
    assert file.resolve() == copy.resolve(), f"active file is {file}, not the copy"
    model = application.ActiveModelReference; rasters = []
    for i, r in enumerate(application.RasterManager.Rasters, 1):
        r = comtypes.client.GetBestInterface(r); info = comtypes.client.GetBestInterface(r.RasterInformation); geo = comtypes.client.GetBestInterface(r.GeoReferenceInformation)
        tif = Path(info.FullName)
        if not tif.is_absolute(): tif = file.parent / tif
        rasters.append({"index": i, "tif": str(tif), "tif_exists": tif.exists(), "pixels": [info.Extent.X, info.Extent.Y], "origin": [geo.Origin.X, geo.Origin.Y],
                        "extent": [geo.Extent.X, geo.Extent.Y], "rotation": geo.RotationAboutZ, "affinity": geo.Affinity})
    elements = []; scan = model.Scan()
    while scan.MoveNext():
        e = scan.Current
        if not e.IsGraphical: continue
        row = {"id": int(e.ID64), "type": int(e.Type), "level": str(e.Level.Name), "level_number": int(e.Level.Number)}
        if e.IsVertexList: row["points"] = [[p.X, p.Y, p.Z] for p in e.AsVertexList.GetVertices()]
        elif int(e.Type) == 37:
            t = e.AsTagElement; row.update(tag=str(t.TagDefinitionName), value=t.Value, base=int(t.BaseElement.ID64))
        elif int(e.Type) == 17:
            try: row["text"] = str(e.AsTextElement.Text); o = e.AsTextElement.Origin; row["origin"] = [o.X, o.Y, o.Z]
            except Exception: pass
        elements.append(row)
    return {"dgn_copy": str(file), "model_name": str(model.Name), "model_count": int(application.ActiveDesignFile.Models.Count), "unit": str(model.MasterUnit.Label), "rasters": rasters, "elements": elements}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--timeout", type=int, default=180); a = ap.parse_args()
    import comtypes.client
    try:
        comtypes.client.GetActiveObject("MicroStationDGN.Application"); print("MicroStation is already open: close it first (the user's own session is never touched)"); sys.exit(2)
    except OSError: pass
    for src in sources():
        name = src.parent.name if src.parent.parent == ROOT else src.parent.parent.name + "_" + src.parent.name
        if a.only and a.only.lower() not in name.lower(): continue
        d = OUT / name; (d / "source").mkdir(parents=True, exist_ok=True); before = sha(src); copy = d / "source" / src.name
        if (d / "native_snapshot.json").exists() and json.load(open(d / "native_snapshot.json", encoding="utf-8")).get("source_sha256") == before:
            print(name, "already read"); continue
        shutil.copy2(src, copy)
        for t in list(src.parent.glob("*.tif")) + list(src.parent.glob("*.tiff")):
            if not (d / "source" / t.name).exists(): shutil.copy2(t, d / "source" / t.name)
        proc = subprocess.Popen([EXE, str(copy)]); t0 = time.time(); data = None; err = None
        try:
            while time.time() - t0 < a.timeout:
                time.sleep(3)
                try:
                    application = comtypes.client.GetActiveObject("MicroStationDGN.Application")
                    if Path(application.ActiveDesignFile.FullName).resolve() == copy.resolve(): data = read(application, copy); break
                except Exception as e: err = repr(e)
            if data is None: print(name, "NOT READ within the time limit:", err)
            else:
                data.update(source_dgn=str(src), source_sha256=before, source_unchanged=sha(src) == before, read_at=time.strftime("%Y-%m-%d %H:%M"))
                json.dump(data, open(d / "native_snapshot.json", "w", encoding="utf-8"), ensure_ascii=False)
                from collections import Counter
                c = Counter(e["level"] for e in data["elements"]); print(name, "| elements", len(data["elements"]), "| rooms", c.get("OdaAlani", 0), "| floors", c.get("KatAlani", 0), "| frames", c.get("KatPlaniSiniri", 0),
                                                                    "| rasters", [(Path(r["tif"]).name[:22], r["tif_exists"], [int(v) for v in r["pixels"]]) for r in data["rasters"]], "| models", data["model_count"], flush=True)
        finally:
            try: comtypes.client.GetActiveObject("MicroStationDGN.Application").Quit()
            except Exception: pass
            try: proc.wait(30)
            except Exception: proc.kill()
            time.sleep(2)
