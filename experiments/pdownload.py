"""Parallel range download (N connections) with resume and sha256 check against Hugging Face's LFS oid.

usage: python pdownload.py URL OUT_PATH [--parts 8] [--sha256 HEX]
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import argparse
import hashlib
import os
import threading
import time
import urllib.request


def head_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pdownload"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), r.geturl()


def fetch(url, path, start, end, prog, i):
    done = os.path.getsize(path) if os.path.exists(path) else 0
    while start + done <= end:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pdownload", "Range": f"bytes={start + done}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); done += len(b); prog[i] = done
        except Exception as e:
            time.sleep(3)
    prog[i] = done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--sha256", default="")
    a = ap.parse_args()
    size, final = head_size(a.url)
    n = a.parts
    step = size // n
    ranges = [(k * step, size - 1 if k == n - 1 else (k + 1) * step - 1) for k in range(n)]
    prog = [0] * n
    th = [threading.Thread(target=fetch, args=(final, f"{a.out}.part{k}", s, e, prog, k), daemon=True) for k, (s, e) in enumerate(ranges)]
    for t in th:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in th):
        time.sleep(15)
        got = sum(prog)
        print(f"{got / 1e9:.2f} / {size / 1e9:.2f} GB  {got / 1e6 / max(1, time.time() - t0):.0f} MB/s", flush=True)
    with open(a.out, "wb") as out:
        for k in range(n):
            with open(f"{a.out}.part{k}", "rb") as f:
                while True:
                    b = f.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
    for k in range(n):
        os.remove(f"{a.out}.part{k}")
    assert os.path.getsize(a.out) == size, "size mismatch"
    if a.sha256:
        h = hashlib.sha256()
        with open(a.out, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                h.update(b)
        print("sha256", "OK" if h.hexdigest() == a.sha256 else "MISMATCH " + h.hexdigest())
    print("done", a.out)


if __name__ == "__main__":
    main()
