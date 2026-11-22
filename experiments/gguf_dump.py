"""Dump a GGUF's metadata keys and the tensors of chosen layers (header only, every shard).

usage: python gguf_dump.py SHARD1 [layers e.g. 0,2] [kv-filter]
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import glob
import re
import struct
import sys

from gguf_header import GGML, R


def header(path, nbytes=64 << 20):
    with open(path, "rb") as f:
        b = f.read(nbytes)
    r = R(b)
    assert r.u("<I", 4) == 0x46554747
    r.u("<I", 4); nt = r.u("<Q", 8); nkv = r.u("<Q", 8)
    kv = {}
    for _ in range(nkv):
        k = r.s(); t = r.u("<I", 4); kv[k] = r.val(t)
    ts = []
    for _ in range(nt):
        name = r.s(); nd = r.u("<I", 4); dims = [r.u("<Q", 8) for _ in range(nd)]; t = r.u("<I", 4); r.u("<Q", 8)
        ts.append((name, dims, GGML.get(t, str(t))))
    return kv, ts


def main():
    first = sys.argv[1]
    layers = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"0"}
    filt = sys.argv[3] if len(sys.argv) > 3 else "deepseek41|tokenizer.ggml.(model|pre)|general.arch"
    parts = sorted(glob.glob(re.sub(r"-\d{5}-of-", "-*-of-", first)))
    for i, p in enumerate(parts):
        kv, ts = header(p)
        if i == 0:
            for k in sorted(kv):
                if re.search(filt, k):
                    v = kv[k]
                    print("KV", k, "=", v if not isinstance(v, list) or len(v) < 12 else str(v)[:120])
        for name, dims, t in ts:
            m = re.match(r"blk\.(\d+)\.", name)
            if (m and m.group(1) in layers) or not m:
                print(f"{p[-20:]:>20} {name:45} {t:8} {dims}")


if __name__ == "__main__":
    main()
