"""Compare GGUF tensors (dequantized) with the original DeepSeek-V4.1 checkpoint, tensor by tensor.

Finds layout changes the llama.cpp converter made (row permutations, transposes, sign/scale
changes) that an engine reading the GGUF under the checkpoint's names would get wrong.

usage: python gguf_vs_ckpt.py [layer]
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

from gguf_header import GGML, R

GG = Path(r"F:\Models\DeepSeek-V4.1-Flash-GSQ-RCO-IQ2_XXS")
CK = Path(r"F:\Models\DeepSeek-V4.1-Flash-1017")


def gguf_index():
    out = {}
    for p in sorted(GG.glob("DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000[1-5]-of-00006.gguf")):
        with open(p, "rb") as f:
            b = f.read(64 << 20)
        r = R(b)
        r.u("<I", 4); r.u("<I", 4); nt = r.u("<Q", 8); nkv = r.u("<Q", 8)
        align = 32
        for _ in range(nkv):
            k = r.s(); t = r.u("<I", 4); v = r.val(t)
            if k == "general.alignment":
                align = v
        ts = []
        for _ in range(nt):
            name = r.s(); nd = r.u("<I", 4); dims = [r.u("<Q", 8) for _ in range(nd)]; t = r.u("<I", 4); off = r.u("<Q", 8)
            ts.append((name, dims, t, off))
        start = (r.o + align - 1) // align * align
        for name, dims, t, off in ts:
            out[name] = (p, dims, t, start + off)
    return out


def gguf_read(ix, name):
    p, dims, t, off = ix[name]
    n = int(np.prod(dims))
    tn = GGML[t]
    with open(p, "rb") as f:
        f.seek(off)
        if tn == "F32":
            a = np.frombuffer(f.read(4 * n), "<f4")
        elif tn == "F16":
            a = np.frombuffer(f.read(2 * n), "<f2").astype(np.float32)
        elif tn == "BF16":
            a = (np.frombuffer(f.read(2 * n), "<u2").astype(np.uint32) << 16).view(np.float32)
        elif tn == "Q8_0":
            raw = np.frombuffer(f.read(34 * (n // 32)), np.uint8).reshape(-1, 34)
            d = raw[:, :2].copy().view("<f2").astype(np.float32)
            q = raw[:, 2:].view(np.int8).astype(np.float32)
            a = (q * d).reshape(-1)
        elif tn in ("Q5_K", "Q6_K"):
            a = dq_k(f.read((176 if tn == "Q5_K" else 210) * (n // 256)), tn)
        else:
            raise ValueError(tn)
    return a.reshape(list(reversed(dims)))   # row-major: ggml ne reversed


def _scale_min_k4(sc):
    """sc [nb,12] -> scales [nb,8], mins [nb,8] (ggml get_scale_min_k4)"""
    sc = sc.astype(np.int32)
    s = np.zeros((len(sc), 8), np.int32); m = np.zeros((len(sc), 8), np.int32)
    for j in range(8):
        if j < 4:
            s[:, j] = sc[:, j] & 63; m[:, j] = sc[:, j + 4] & 63
        else:
            s[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            m[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return s.astype(np.float32), m.astype(np.float32)


def dq_k(buf, tn):
    if tn == "Q5_K":
        raw = np.frombuffer(buf, np.uint8).reshape(-1, 176)
        d = raw[:, 0:2].copy().view("<f2").astype(np.float32); dmin = raw[:, 2:4].copy().view("<f2").astype(np.float32)
        sc, mn = _scale_min_k4(raw[:, 4:16]); qh = raw[:, 16:48].astype(np.int32); qs = raw[:, 48:176].astype(np.int32)
        y = np.zeros((len(raw), 256), np.float32)
        for j in range(4):                       # 64 values per j: two sub-blocks of 32
            ql = qs[:, 32 * j:32 * j + 32]
            lo = (ql & 0xF) + (((qh >> (2 * j)) & 1) << 4)
            hi = (ql >> 4) + (((qh >> (2 * j + 1)) & 1) << 4)
            y[:, 64 * j:64 * j + 32] = d * sc[:, 2 * j:2 * j + 1] * lo - dmin * mn[:, 2 * j:2 * j + 1]
            y[:, 64 * j + 32:64 * j + 64] = d * sc[:, 2 * j + 1:2 * j + 2] * hi - dmin * mn[:, 2 * j + 1:2 * j + 2]
        return y.reshape(-1)
    raw = np.frombuffer(buf, np.uint8).reshape(-1, 210)   # Q6_K: ql[128] qh[64] scales[16] d
    ql = raw[:, :128].astype(np.int32); qh = raw[:, 128:192].astype(np.int32)
    sc = raw[:, 192:208].view(np.int8).astype(np.float32); d = raw[:, 208:210].copy().view("<f2").astype(np.float32)
    y = np.zeros((len(raw), 256), np.float32)
    for h in range(2):
        L = ql[:, 64 * h:64 * h + 64]; H = qh[:, 32 * h:32 * h + 32]; S = sc[:, 8 * h:8 * h + 8]
        for l in range(32):
            q1 = ((L[:, l] & 0xF) | (((H[:, l] >> 0) & 3) << 4)) - 32
            q2 = ((L[:, l + 32] & 0xF) | (((H[:, l] >> 2) & 3) << 4)) - 32
            q3 = ((L[:, l] >> 4) | (((H[:, l] >> 4) & 3) << 4)) - 32
            q4 = ((L[:, l + 32] >> 4) | (((H[:, l] >> 6) & 3) << 4)) - 32
            is_ = l // 16
            base = 128 * h
            y[:, base + l] = d[:, 0] * S[:, is_] * q1
            y[:, base + l + 32] = d[:, 0] * S[:, is_ + 2] * q2
            y[:, base + l + 64] = d[:, 0] * S[:, is_ + 4] * q3
            y[:, base + l + 96] = d[:, 0] * S[:, is_ + 6] * q4
    return y.reshape(-1)


E4M3 = None


def e4m3_lut():
    lut = np.zeros(256, np.float32)
    for i in range(256):
        s = -1.0 if i & 0x80 else 1.0
        e = (i >> 3) & 0xF; m = i & 7
        if e == 15 and m == 7:
            lut[i] = np.nan
        elif e == 0:
            lut[i] = s * (m / 8) * 2 ** -6
        else:
            lut[i] = s * (1 + m / 8) * 2 ** (e - 7)
    return lut


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


CK_MAP = json.load(open(CK / "model.safetensors.index.json"))["weight_map"]


def ck_raw(name):
    path = CK / CK_MAP[name]
    h, base = st_header(path)
    m = h[name]
    a0, a1 = m["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a0)
        buf = f.read(a1 - a0)
    return m["dtype"], m["shape"], buf


def ck_read(name):
    global E4M3
    dt, shape, buf = ck_raw(name)
    if dt == "F32":
        return np.frombuffer(buf, "<f4").reshape(shape)
    if dt == "BF16":
        return (np.frombuffer(buf, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dt == "F16":
        return np.frombuffer(buf, "<f2").astype(np.float32).reshape(shape)
    if dt == "F8_E4M3":
        E4M3 = E4M3 if E4M3 is not None else e4m3_lut()
        w = E4M3[np.frombuffer(buf, np.uint8)].reshape(shape)
        sn = name.replace(".weight", ".scale")
        sdt, sshape, sbuf = ck_raw(sn)
        if sdt == "F8_E8M0":
            s = (2.0 ** (np.frombuffer(sbuf, np.uint8).astype(np.float32) - 127)).reshape(sshape)
        elif sdt == "F32":
            s = np.frombuffer(sbuf, "<f4").reshape(sshape)
        else:
            s = (np.frombuffer(sbuf, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(sshape)
        br, bc = -(-shape[0] // sshape[0]), -(-shape[1] // sshape[1])
        S = np.repeat(np.repeat(s, br, 0), bc, 1)[:shape[0], :shape[1]]
        return w * S
    raise ValueError(dt)


def corr(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    a -= a.mean(); b -= b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def compare(ix, gname, cname):
    g = gguf_read(ix, gname); c = ck_read(cname)
    line = f"{gname:32} {str(list(g.shape)):16} vs {cname:30} {str(list(c.shape)):16}"
    if g.shape != c.shape:
        if g.size == c.size:
            line += f" | reshaped corr {corr(g, c.reshape(g.shape)):.4f}"
        if g.ndim == 2 and g.shape == c.shape[::-1]:
            line += f" | TRANSPOSED corr {corr(g, c.T):.4f}"
        print(line); return g, c
    cr = corr(g, c)
    line += f" | corr {cr:.4f} scale {np.linalg.norm(g) / (np.linalg.norm(c) + 1e-30):.3f}"
    if g.ndim == 2 and cr < 0.98:
        # rows matched one by one: a row permutation shows as a high best match away from the diagonal
        gn = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-30); cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-30)
        M = gn @ cn.T
        best = M.argmax(1)
        bad = np.nonzero(best != np.arange(len(best)))[0]
        line += f" | rows off-diagonal {len(bad)}/{len(best)}"
        if len(bad):
            line += f" first {[(int(i), int(best[i])) for i in bad[:12]]}"
    print(line)
    return g, c


def main():
    L = sys.argv[1] if len(sys.argv) > 1 else "0"
    ix = gguf_index()
    pairs = [("attn_sinks.weight", "attn.attn_sink"), ("hc_attn_base.weight", "hc_attn_base"),
             ("hc_attn_scale.weight", "hc_attn_scale"), ("hc_attn_fn.weight", "hc_attn_fn"),
             ("attn_norm.weight", "attn_norm.weight"), ("attn_q_a_norm.weight", "attn.q_norm.weight"),
             ("attn_kv_a_norm.weight", "attn.kv_norm.weight"), ("attn_q_a.weight", "attn.wq_a.weight"),
             ("attn_kv.weight", "attn.wkv.weight"), ("ffn_gate_inp.weight", "ffn.gate.weight"),
             ("exp_probs_b.bias", "ffn.gate.bias"), ("exp_probs_b_vl.bias", "ffn.gate.bias_vl"),
             ("attn_q_b.weight", "attn.wq_b.weight"), ("attn_output_a.weight", "attn.wo_a.weight"),
             ("attn_output_b.weight", "attn.wo_b.weight"), ("ffn_gate_shexp.weight", "ffn.shared_experts.w1.weight"),
             ("ffn_down_shexp.weight", "ffn.shared_experts.w2.weight"),
             ("attn_compressor_kv.weight", "attn.compressor.wkv.weight"),
             ("attn_compressor_gate.weight", "attn.compressor.wgate.weight"),
             ("indexer.attn_q_b.weight", "attn.indexer.wq_b.weight"), ("indexer.attn_k.weight", "attn.indexer.wk.weight"),
             ("indexer.proj.weight", "attn.indexer.weights_proj.weight")]
    if len(sys.argv) > 2:
        pairs = [p for p in pairs if re.search(sys.argv[2], p[0])]
    for gg, ck in pairs:
        try:
            compare(ix, f"blk.{L}.{gg}", f"layers.{L}.{ck}")
        except Exception as e:
            print(gg, "ERR", e)
    for gg, ck in (("output_norm.weight", "norm.weight"),):
        compare(ix, gg, ck)


if __name__ == "__main__":
    main()
