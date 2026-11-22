"""Run the opening net a1 on a floor without room input (seeds only).  usage: python a1_calistir.py FLOOR   (.venv-spec, 5070 Ti)"""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, os, sys
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from pathlib import Path
import numpy as np, torch
import aciklik_ag as AA
fid = sys.argv[1]; dev = torch.device("cuda:0"); net = AA.Net().to(dev); net.load_state_dict(torch.load(AA.OUT / "a1" / "unet.pt", map_location=dev))
g, lab, rooms, ops, bad = AA.load(fid)
pr = AA.predict(net, np.stack([g.astype(np.float32) / 255.0, np.zeros(g.shape, np.float32)]), dev); pred = AA.boxes(pr)
json.dump(pred, open(AA.OUT / "a1" / f"{fid}_tahmin.json", "w")); print(len(pred), "kutu")
