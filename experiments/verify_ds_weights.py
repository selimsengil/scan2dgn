"""Check that the local DeepSeek-V4.1-Flash-1017 files are byte-identical to Hugging Face's.

The reference is Hugging Face's own tree API (LFS sha256 / git blob of every file at the downloaded revision),
not the local DOWNLOAD_MANIFEST.json (which was written by the downloader); the manifest is compared too.
Hashes the 48 safetensors in parallel.

usage: python verify_ds_weights.py  ->  runs/ds_weights_verify.json
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import hashlib
import json
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

D = Path(r"F:\Models\DeepSeek-V4.1-Flash-1017")
REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
OUT = Path(__file__).with_name("runs") / "ds_weights_verify.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb", buffering=0) as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b)
    return str(p), h.hexdigest()


def git_blob_sha1(p):
    data = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    man = json.load(open(D / "DOWNLOAD_MANIFEST.json", encoding="utf-8"))
    rev = man["revision"]
    url = f"https://huggingface.co/api/models/{REPO}/tree/{rev}?recursive=1"
    tree = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "weights-verify"}), timeout=60))
    remote = {e["path"]: e for e in tree if e.get("type") == "file"}
    mf = {f["path"]: f["sha256"] for f in man["files"]}
    big = sorted(p for p in remote if "lfs" in remote[p])
    small = sorted(p for p in remote if "lfs" not in remote[p])
    res = {"revision": rev, "remote_files": len(remote), "lfs": {}, "small": {}, "missing": [], "extra": []}
    local = {str(p.relative_to(D)).replace("\\", "/") for p in D.rglob("*") if p.is_file()}
    res["missing"] = sorted(set(remote) - local)
    res["extra"] = sorted(local - set(remote))
    with ProcessPoolExecutor(8) as ex:
        for path, h in ex.map(sha256, [D / p for p in big if p in local]):
            rel = str(Path(path).relative_to(D)).replace("\\", "/")
            want = remote[rel]["lfs"]["oid"]
            res["lfs"][rel] = {"ok": h == want, "size_ok": Path(path).stat().st_size == remote[rel]["lfs"]["size"],
                               "manifest_ok": mf.get(rel) == h}
            print(rel, "OK" if h == want else "DIFFERENT", flush=True)
    for p in small:
        if p in local:
            res["small"][p] = git_blob_sha1(D / p) == remote[p]["oid"]
    res["all_lfs_ok"] = all(v["ok"] and v["size_ok"] for v in res["lfs"].values())
    res["all_small_ok"] = all(res["small"].values())
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("LFS files:", len(res["lfs"]), "all identical:", res["all_lfs_ok"])
    print("small files identical:", res["all_small_ok"], [p for p, ok in res["small"].items() if not ok])
    print("missing:", res["missing"]); print("extra (local only):", res["extra"])


if __name__ == "__main__":
    main()
