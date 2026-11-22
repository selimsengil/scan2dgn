"""Read a GGUF header (metadata + tensor infos) from the first bytes of a file.

usage: python gguf_header.py FILE   -> tensor type census, routed-expert types, sizes
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import struct
import sys
from collections import Counter, defaultdict

GGML = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
        12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
        20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
        29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
# bits per weight of the block formats (for size estimates)
BPW = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5, "Q3_K": 3.4375,
       "Q2_K": 2.625, "IQ4_NL": 4.5, "IQ4_XS": 4.25, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "IQ2_S": 2.5,
       "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625, "IQ1_M": 1.75, "IQ1_S": 1.5625, "MXFP4": 4.25, "Q4_0": 4.5, "Q5_0": 5.5}


class R:
    def __init__(self, b): self.b, self.o = b, 0
    def u(self, f, n): v = struct.unpack_from(f, self.b, self.o); self.o += n; return v[0]
    def s(self): n = self.u("<Q", 8); v = self.b[self.o:self.o + n].decode("utf-8", "replace"); self.o += n; return v
    def val(self, t):
        f = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
             10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
        if t in f: return self.u(*f[t])
        if t == 8: return self.s()
        if t == 9:
            et = self.u("<I", 4); n = self.u("<Q", 8)
            return [self.val(et) for _ in range(n)] if n < 64 else (self.skip_arr(et, n) or f"[{n} items]")
        raise ValueError(t)
    def skip_arr(self, et, n):
        for _ in range(n): self.val(et)


def main():
    b = open(sys.argv[1], "rb").read()
    r = R(b)
    assert r.u("<I", 4) == 0x46554747, "not GGUF"
    ver = r.u("<I", 4); nt = r.u("<Q", 8); nkv = r.u("<Q", 8)
    kv = {}
    for _ in range(nkv):
        k = r.s(); t = r.u("<I", 4); kv[k] = r.val(t)
    print("version", ver, "tensors", nt, "kv", nkv)
    for k in sorted(kv):
        if any(x in k for x in ("general.", "split", "expert", "block_count", "quant")):
            print("  ", k, "=", kv[k] if not isinstance(kv[k], list) or len(kv[k]) < 8 else str(kv[k])[:80])
    census, byrole, size = Counter(), defaultdict(Counter), Counter()
    for _ in range(nt):
        name = r.s(); nd = r.u("<I", 4); dims = [r.u("<Q", 8) for _ in range(nd)]; t = r.u("<I", 4); r.u("<Q", 8)
        tn = GGML.get(t, str(t)); census[tn] += 1
        n = 1
        for d in dims: n *= d
        role = "routed_expert" if "_exps" in name else "other"
        byrole[role][tn] += 1
        size[role] += n * BPW.get(tn, 16) / 8
    print("type census:", dict(census))
    for role in byrole:
        print(f"{role}: types {dict(byrole[role])}, ~{size[role] / 1e9:.1f} GB in this shard")


if __name__ == "__main__":
    main()
