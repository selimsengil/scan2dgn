# What did not work (and what that taught)

Every row was measured. The failures shaped the current design as much as the successes did.

## General-purpose models drawing geometry

| Tried | Result | Lesson |
|---|---|---|
| Qwen 27B (local) drawing room polygons, or boxes | ~0.73 IoU at best for both designs; only column and slant detection was reliable | A VLM can read a plan, but it cannot place a wall to the centimetre. |
| Qwen walking along walls step by step | 9/44 rooms in 4.7 h, worse than one-shot | Agent loops multiply the error and the time. |
| DeepSeek (local) per room | 50 s prefill and 3.4 tok/s | Unusable at ~15 projects/day. |
| Hosted frontier models on one floor, with an encoded "piece package" | best ~0.69 | Beaten by a small U-Net trained on the office's own drawings (0.79–0.84). |

**Decision:** learn the geometry from the office's finished DGNs with small task-specific networks. Use large
models only where reading is needed.

## Rooms

- **Per-piece gradient boosting** on encoded sheet pieces scored 0.44. The pieces are slivers, and their own shape
  says nothing. A U-Net on the grey scan scored 0.79.
- **Rooms cut by a straight line** through open floor turned out to be a graph bug, not a classifier error: the
  hair-thin bodies of drawn lines were dropped as pieces, so neighbours on either side had no link. Linking
  pieces closer than 1.5 px raised one floor from 0.60 to 0.70.
- **Room-area targets were replaced by line targets.** On the held-out family, the hardest floor went from 0.60
  to 0.78.
- **Not all labels are true.** Upper repeated floors in old drawings are often drawn loosely: one floor matched
  the scan 24 %, its ground floor 100 %. Rooms are audited against the ink before training. Unclean outlines are
  ignored rather than learned.

## Openings

- **Feeding OCR tag hints into the opening net made it worse**, because hints in production are noisy (264 hints
  for 40 openings on one floor).

  | Variant | Drawn right / 144 |
  |---|---|
  | No tags | 127 |
  | Realistic-noise training, no tags | 114 |
  | Realistic-noise training, with production tags | 99 |

  The net now only draws. Tags are used for the numbers alone.
- **The post-processing first broke boxes that were already valid.** Rule: a valid box is never touched, and a
  correction that does not reduce the rule errors is discarded.
- **Widening the face-snap search from 30 to 45 px hurt.** It snapped to the wrong face.
- **Systematic end overshoot:** each end was ~2.3–3.2 cm long. The correction is calibrated per network on
  *training* floors only, never on the test floors.
- **Box extraction logic bugs, fixed without retraining:**
  - a door and its neighbouring window merged into one piece;
  - L-shaped pieces where two doors meet at a corner;
  - a grow step that added overshoot.

  Location went from 136/144 to 141/144, and valid drawn boxes from 127 to 129.

## Tags

- **No single OCR reads everything.** Each candidate misses different tags, and VLM readers invent digits on
  illegible windows.
- **Whole-sheet VLM OCR** (PaddleOCR-VL, DeepSeek-OCR-2) was slow and missed small rotated tags.
- **A metric that measured nothing:** "a read within 0.6 m of an opening" scored the same for random placement.
  Every location metric is now reported next to a random baseline.
- **Word spotting first took 33 minutes per floor.** Two causes: generic templates (strokes that match
  everywhere) and an O(n²) peak dedupe. Fixed by tightening seeds to their ink, dropping templates with more than
  80 copies, and using local maxima capped at 300 peaks.

## Process

- Concurrent GPU jobs crashed with "illegal memory access", so the rule is one GPU job at a time.
- Long runs are queued as scripts that wait for the previous artefact, so a PC restart loses one step, not the
  chain.
