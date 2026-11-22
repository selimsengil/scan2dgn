"""PP-OCRv5 server det+rec (PaddleOCR 3, CPU).  A: sheet tiles at 0 and 90 deg; B: level crops.   Run with .venv-ocr."""
import sys as _sys; from pathlib import Path as _P; _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
import config; config.setup()  # noqa: E402  repo paths: data/model roots come from config.local.json or env
import json, sys, time
from ortak import *
from paddleocr import PaddleOCR
D = str(config.MODELS)
ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_detection_model_dir=D + r"\PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec", text_recognition_model_dir=D + r"\PP-OCRv5_server_rec",
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device="cpu", enable_mkldnn=False)
def read(img):
    r = ocr.predict(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[0]
    return [(np.asarray(p, float), t) for p, t in zip(r["rec_polys"], r["rec_texts"])]
mode = sys.argv[1]
if mode == "B":
    res = {}; t0 = time.time()
    for oid, kind, tag, ims in crops():
        res[oid] = [" ".join(t for _, t in read(cv2.resize(im, None, fx=1.5, fy=1.5))) for im in ims]
    r = score_crops(res, "PP-OCRv5 kirpim", {"sure_s": round(time.time() - t0, 1)}); json.dump(r, open(H / "B_ppocr.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: r[k] for k in ("door", "window", "sure_s")}, flush=True)
if mode == "A":
    g = sheet()[400:10500, 400:5300]; OX, OY = 400, 400; Hs, Ws = g.shape; reads = []; t0 = time.time()
    for rot in (0, 90, 180, 270):
        im = g if rot == 0 else cv2.rotate(g, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot]); h, w = im.shape
        for y0 in range(0, max(1, h - 200), 1400):
            for x0 in range(0, max(1, w - 200), 1400):
                for p, t in read(im[y0:y0 + 1600, x0:x0 + 1600]):
                    q = p + [x0, y0]; x, y = q[:, 0], q[:, 1]
                    if rot == 90: q = np.stack([y, Hs - 1 - x], 1)
                    elif rot == 180: q = np.stack([Ws - 1 - x, Hs - 1 - y], 1)
                    elif rot == 270: q = np.stack([Ws - 1 - y, x], 1)
                    reads.append({"box": (q + [OX, OY]).tolist(), "text": t, "rot": rot})
        print(rot, len(reads), round(time.time() - t0), flush=True)
    json.dump(reads, open(H / "A_ppocr_okumalar.json", "w", encoding="utf-8"), ensure_ascii=False)
    r = score_spot(reads, "PP-OCRv5 4 yon"); r["sure_s"] = round(time.time() - t0); json.dump(r, open(H / "A_ppocr.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print({k: r[k] for k in ("door", "window", "okuma_sayisi", "sure_s")})
