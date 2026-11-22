"""Static safety scan of a torch .pt checkpoint BEFORE loading it: list every global the pickle would import.
A plain weights file only references torch storage/tensor rebuild helpers and collections.OrderedDict;
anything else (os, subprocess, builtins.eval/exec, ...) means it can run code. Nothing is unpickled here.

usage: python scan_pt.py FILE.pt [FILE.pt ...]
"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import pickletools
import sys
import zipfile
from collections import Counter

SAFE_PREFIX = ("torch._utils", "torch.", "collections", "_codecs", "numpy.core.multiarray", "numpy.dtype", "numpy")
SAFE_NAMES = {("torch._utils", "_rebuild_tensor_v2"), ("collections", "OrderedDict"), ("torch", "FloatStorage"),
              ("torch", "HalfStorage"), ("torch", "BFloat16Storage"), ("torch", "LongStorage"), ("torch", "IntStorage"),
              ("torch", "BoolStorage"), ("torch", "ByteStorage"), ("torch", "DoubleStorage"), ("_codecs", "encode"),
              ("torch._utils", "_rebuild_parameter"), ("torch", "Size"), ("builtins", "set"), ("builtins", "frozenset"),
              ("torch", "float32"), ("torch", "bfloat16"), ("torch", "float16")}


def globals_of(pkl: bytes):
    found = Counter()
    strings = []
    for op, arg, _ in pickletools.genops(pkl):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
            strings.append(arg)
        if op.name == "GLOBAL":
            mod, name = arg.split(" ", 1)
            found[(mod, name)] += 1
        elif op.name == "STACK_GLOBAL":
            found[(strings[-2], strings[-1])] += 1
        elif op.name in ("REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD"):
            pass
    return found


def main():
    for f in sys.argv[1:]:
        with zipfile.ZipFile(f) as z:
            pk = [n for n in z.namelist() if n.endswith(".pkl")]
            print(f, "pickles:", pk, "entries:", len(z.namelist()))
            for n in pk:
                g = globals_of(z.read(n))
                bad = {k: v for k, v in g.items() if k not in SAFE_NAMES}
                for (m, nm), c in sorted(g.items()):
                    print(f"   {m}.{nm}  x{c}" + ("" if (m, nm) in SAFE_NAMES else "   <-- NOT IN SAFE LIST"))
                print("   VERDICT:", "only weight-rebuild globals" if not bad else f"{len(bad)} globals outside the safe list")


if __name__ == "__main__":
    main()
