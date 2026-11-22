"""Check the engine's DeepSeek-V4.1 vision tower against a numpy copy of the vendor's vision.py.

1. One small image goes through the engine (Colibri, V41_VISION_DUMP): the aligner's rows.
2. The same patches go through vision.py's ViT + Aligner in numpy, with the checkpoint's own
   weights (model-00001-of-00048.safetensors, bf16).
3. Per-row cosine and relative error between the two.

usage: python vision_check.py [--exe X.exe] [--image PNG] [--side 560] [--env K=V ...]
"""
from __future__ import annotations
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CK = Path(r"F:\Models\DeepSeek-V4.1-Flash-1017")
OUT = Path(__file__).with_name("runs") / "vision_check"


def st_tensors(names):
    idx = json.load(open(CK / "model.safetensors.index.json"))["weight_map"]
    by_file = {}
    for n in names:
        by_file.setdefault(idx[n], []).append(n)
    out = {}
    for fn, ns in by_file.items():
        with open(CK / fn, "rb") as f:
            hl = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(hl))
            for n in ns:
                m = h[n]
                a0, a1 = m["data_offsets"]
                f.seek(8 + hl + a0)
                b = f.read(a1 - a0)
                if m["dtype"] == "BF16":
                    a = (np.frombuffer(b, "<u2").astype(np.uint32) << 16).view(np.float32)
                elif m["dtype"] == "F32":
                    a = np.frombuffer(b, "<f4").copy()
                elif m["dtype"] == "F16":
                    a = np.frombuffer(b, "<f2").astype(np.float32)
                else:
                    raise ValueError(f"{n}: {m['dtype']}")
                out[n] = a.reshape(m["shape"])
    return out


def rms(x, w, eps=1e-6):
    return w * (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps))


def reference(patches, n_h, n_w, cfg):
    D, H, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_hidden_layers"]
    hd = D // H
    r = cfg["downsample_ratio"]
    names = ["vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight",
             "aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias"]
    for i in range(L):
        names += [f"vision.blocks.{i}.{s}" for s in ("norm1.weight", "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight",
                                                    "attn.wo.bias", "norm2.weight", "mlp.w1.weight", "mlp.w2.weight")]
    W = st_tensors(names)
    x = patches @ W["vision.patch_embed.proj.weight"].T + W["vision.patch_embed.proj.bias"]
    # get_vision_cos_sin(n_h, n_w, rope_dim = hd // 2, theta)
    dim = hd // 2
    inv = 1.0 / (cfg["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    hpos = np.repeat(np.arange(n_h)[:, None], n_w, 1).reshape(-1)
    wpos = np.repeat(np.arange(n_w)[None, :], n_h, 0).reshape(-1)
    freqs = np.concatenate([hpos[:, None] * inv[None], wpos[:, None] * inv[None]], 1)   # [n, dim]
    cos, sin = np.cos(freqs)[:, None, :], np.sin(freqs)[:, None, :]

    def rot(t):
        x1, x2 = t[..., :dim], t[..., dim:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    n = x.shape[0]
    for i in range(L):
        p = f"vision.blocks.{i}."
        hN = rms(x, W[p + "norm1.weight"])
        qkv = hN @ W[p + "attn.wqkv.weight"].T + W[p + "attn.wqkv.bias"]
        q, k, v = (t.reshape(n, H, hd) for t in np.split(qkv, 3, -1))
        q, k = rot(q), rot(k)
        q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(n, D)
        x = x + o @ W[p + "attn.wo.weight"].T + W[p + "attn.wo.bias"]
        hN = rms(x, W[p + "norm2.weight"])
        gu = hN @ W[p + "mlp.w1.weight"].T
        g, u = np.split(gu, 2, -1)
        x = x + ((g / (1 + np.exp(-g))) * u) @ W[p + "mlp.w2.weight"].T
        if i % 8 == 7:
            print(f"  reference block {i + 1}/{L}", flush=True)
    x = rms(x, W["vision.norm.weight"])
    # Aligner: view(n_h, n_w, C).permute(2,0,1), pad to r, unfold(r, stride r): [L, C*r*r] channel-major
    C = D
    g = x.reshape(n_h, n_w, C).transpose(2, 0, 1)
    ph, pw = (-n_h) % r, (-n_w) % r
    g = np.pad(g, ((0, 0), (0, ph), (0, pw)))
    bh, bw = g.shape[1] // r, g.shape[2] // r
    blocks = g.reshape(C, bh, r, bw, r).transpose(1, 3, 0, 2, 4).reshape(bh * bw, C * r * r)
    h1 = blocks @ W["aligner.w1.weight"].T + W["aligner.w1.bias"]
    from math import erf
    h1 = 0.5 * h1 * (1 + np.vectorize(erf)(h1 / math.sqrt(2)).astype(np.float32)) if h1.size < 2_000_000 else \
        0.5 * h1 * (1 + np.tanh(math.sqrt(2 / math.pi) * (h1 + 0.044715 * h1 ** 3)))
    return h1 @ W["aligner.w2.weight"].T + W["aligner.w2.bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io\deepseek_v41_vl.exe")
    ap.add_argument("--image", default=str(Path(__file__).with_name("runs") / "b132_ds500vl_sub22" / "001_12.png"))
    ap.add_argument("--side", type=int, default=560)
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.open(a.image).convert("RGB")
    s = a.side / max(img.size)
    img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.Resampling.LANCZOS)
    p = OUT / "probe.png"; img.save(p)

    sys.path.insert(0, r"C:\Users\Selim\Documents\Proje\third_party\colibri-custom-20261107-io")
    import deepseek_loop as L1
    L1.EXE = Path(a.exe)
    S = L1.S
    prepared = S.prepare_multimodal_payload(p, "Kısa cevap ver.", "Bu görüntüde ne var?", enable_thinking=False)
    gh, gw = prepared["grid_h"], prepared["grid_w"]
    patches = np.frombuffer(prepared["image_bytes"], np.float32).reshape(gh * gw, -1)
    print("grid", gh, "x", gw, "patches", patches.shape, flush=True)
    dump = OUT / "engine_rows.bin"
    if dump.exists():
        dump.unlink()
    env = {"COLI_VK_SHADERS": str(Path(a.exe).parent / "shaders"), "V41_VISION_DUMP": str(dump),
           "COLI_VK_CHAIN_ROWS": "1536", "COLI_VK_TIER_RESERVE_GB": "4.0"}
    env.update(dict(kv.split("=", 1) for kv in a.env))
    sess = L1.start_session(OUT, 900, env)
    try:
        r = sess.submit_multimodal(OUT / "req", prepared, generation_budget=24, timeout_s=1800)
        print("engine answer:", r["raw"][:200], flush=True)
    finally:
        sess.close()
    eng = np.fromfile(dump, np.float32)
    cfg = json.load(open(CK / "config.json"))["vision_config"]
    ref = reference(patches, gh, gw, cfg)
    eng = eng.reshape(ref.shape)
    cos = (eng * ref).sum(1) / (np.linalg.norm(eng, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
    rel = np.linalg.norm(eng - ref) / np.linalg.norm(ref)
    print(f"rows {ref.shape[0]} x {ref.shape[1]} | cosine per row: min {cos.min():.4f} mean {cos.mean():.4f} | "
          f"relative error {rel:.4f} | norms engine {np.linalg.norm(eng):.1f} reference {np.linalg.norm(ref):.1f}")
    np.save(OUT / "reference_rows.npy", ref)


if __name__ == "__main__":
    main()
