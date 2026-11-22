# Detailed results (22 Nov 2026)

## Opening network variants (5 held-out floors, 144 openings)

From `openings/aciklik_karsilastir.py`. The end correction is measured on 3 training floors only.

| Variant | Found | Wrong boxes (rule-valid) | Found but invalid | Valid and found | Jamb ≤ 3 cm | Jamb ≤ 5 cm |
|---|---|---|---|---|---|---|
| a3, old box extraction | 136 | 45 (20) | 9 | 127 | – | 95 |
| a3, `boxes3` with class split and L split | **141** | 45 (21) | 12 | **129** | 60 | 98 |
| a5, retrained | *running* | | | | | |

Median jamb error per floor (a3 + `boxes3`): Malkoç 2.8, Kışla 85/1 3.2, 84/1 5.2, 85/Z 2.9, 84/Z 4.2 cm.

## End-to-end chain (`openings/aciklik_zincir.py`)

| Floor | Drawn right | Wrong drawn | Height right, read | Height right, inferred | Height wrong, read | Height wrong, inferred | Height "?" |
|---|---|---|---|---|---|---|---|
| Malkoç K2 | 38 | 1 | 25 | 6 | 3 | 1 | 3 |
| Kışla 85/1 | 38 | 2 | 35 | 1 | 0 | 0 | 2 |
| Kışla 84/1 | 25 | 7 | 18 | 4 | 0 | 2 | 1 |
| Kışla 85/Z | 6 | 2 | 6 | 0 | 0 | 0 | 0 |
| Kışla 84/Z | 23 | 8 | 16 | 3 | 1 | 0 | 3 |
| **Total** | **130** | **20** | **100** | **14** | **4** | **3** | **9** |

How heights are chosen:
- **Read:** a vote of the tag reads within 0.6 m whose first number fits the box width (±15 cm). A door height
  below 170 is treated as a misread.
- **Inferred:** the floor majority of the same type and width class (±10 cm). A single odd, non-round read
  such as 201 or 212 is replaced by the majority. Every inference is flagged.

## Tag reading candidates

Common harness: `tags/ocr_bench/test2.py`. The Malkoç K2 hand label has exact tags; Kışla 85/1 uses the DGN
width ±12 cm and height ±2.

**Crop readers on Malkoç K2:**

| Reader | Doors /25 | Windows /15 | Time |
|---|---|---|---|
| RapidOCR | 6 | 0 | 23 s |
| PP-OCRv5 server (CPU) | 4 | 0 | 130 s |
| PaddleOCR-VL-1.6 | 22 | 5 | 36 s |
| GLM-OCR | 22 | 0 | 69 s |
| MinerU2.5 | 20 | 0 | 108 s |

Whole-sheet candidates also tried: RapidOCR in 4 rotations, docTR, EasyOCR, Florence-2-large,
DeepSeek-OCR-2, CRAFT + eDOCr2, and a trained tag-finder net.

**Ensemble and word spotting:**

| Floor | Doors | Windows | Random baseline (doors / windows) |
|---|---|---|---|
| Malkoç K2, word spotting location | 23/25 | 15/15 | 4 / 1 |
| Malkoç K2, numbers fully right | 24/25 | not legible in the scan | |
| Malkoç K2, with the OCR ensemble | 25/25 | | |
| Kışla 85/1 | 25/29 | 9/11 | 7 / 2 |

## Rooms (boundary score, held-out Kışla family)

| Floor | Line net + watershed | Room-area net |
|---|---|---|
| 85/1 | 0.864 | 0.842 |
| 84/1 | 0.778 | 0.601 |
| 85/Z | 0.773 | 0.728 |
| 84/Z | 0.747 | 0.582 |

Earlier piece-classifier U-Net, trained out of family: 0.83–0.84 on floor 019 and 0.70 on floor 017.

## Training data extracted from the office's DGNs

- 65 floors: 1332 rooms, 1467 door openings, 736 windows, 255 balconies.
- Label audit: 856 rooms are clean (≥ 97 % of the outline within 2.5 cm of ink).
- Opening tags: 777 harvested and verified against the DGN width/height on 37 training floors.
