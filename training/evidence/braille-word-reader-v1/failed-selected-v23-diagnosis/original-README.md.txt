# Current candidate versus baseline: cached held405 diagnosis

Approximate box matching; not AP or independent physical validation. No model execution, threshold search, or external607 data.

| Model | Correct / 405 | Accepted | Rejected | Accepted precision |
|---|---:|---:|---:|---:|
| baseline | 357 | 362 | 43 | 98.62% |
| candidate | 352 | 355 | 50 | 99.15% |

Lost correct blocks: 11; gained: 6. Lost outcomes: `{"accepted_wrong": 1, "clipped": 3, "low_confidence": 7}`.

Both models use their existing selected thresholds, recorded in JSON; no thresholds are searched or changed. Matching is greedily one-to-one at IoU >=0.25, without reading labels. Unmatched predictions are not automatically false positives.

| Length | Blocks | Baseline correct | Candidate correct | Baseline unmatched GT | Candidate unmatched GT |
|---|---:|---:|---:|---:|---:|
| 1 | 87 | 74 | 71 | 9 | 9 |
| 2 | 63 | 57 | 57 | 10 | 10 |
| 3 | 51 | 45 | 46 | 14 | 13 |
| 4 | 33 | 31 | 32 | 5 | 4 |
| 5 | 30 | 26 | 26 | 14 | 11 |
| 6 | 69 | 62 | 61 | 36 | 36 |
| 7 | 24 | 18 | 18 | 15 | 14 |
| 8 | 33 | 29 | 28 | 8 | 8 |
| 9 | 15 | 15 | 13 | 0 | 0 |
| 10 | 0 | 0 | 0 | 0 | 0 |
| 11 | 0 | 0 | 0 | 0 | 0 |
| 12 | 0 | 0 | 0 | 0 | 0 |

| Pattern | GT cells | Baseline matched / correct mask | Candidate matched / correct mask |
|---|---:|---:|---:|
| A (1) | 144 | 129 / 126 | 131 / 128 |
| B (3) | 18 | 16 / 16 | 16 / 16 |
| C (9) | 6 | 6 / 6 | 6 / 5 |
| D (25) | 51 | 45 / 45 | 45 / 44 |
| E (17) | 129 | 119 / 119 | 119 / 119 |
| G (27) | 15 | 12 / 11 | 13 / 13 |
| H (19) | 3 | 3 / 3 | 3 / 3 |
| I (10) | 138 | 130 / 130 | 131 / 131 |
| J (26) | 48 | 43 / 43 | 43 / 43 |
| K (5) | 102 | 95 / 95 | 95 / 95 |
| L (7) | 42 | 40 / 40 | 40 / 40 |
| M (13) | 81 | 75 / 75 | 75 / 75 |
| N (29) | 105 | 97 / 97 | 97 / 97 |
| O (21) | 186 | 174 / 174 | 174 / 174 |
| P (15) | 57 | 53 / 53 | 54 / 54 |
| Q (31) | 36 | 36 / 36 | 36 / 36 |
| R (23) | 81 | 77 / 76 | 78 / 78 |
| S (14) | 93 | 86 / 86 | 86 / 86 |
| T (30) | 54 | 53 / 52 | 53 / 53 |
| U (37) | 84 | 80 / 80 | 80 / 78 |
| W (58) | 108 | 104 / 102 | 104 / 104 |
| X (45) | 9 | 7 / 7 | 7 / 7 |
| Y (61) | 3 | 3 / 3 | 3 / 3 |
| Z (53) | 36 | 35 / 35 | 35 / 35 |

Full row-level diagnosis: [comparison.json](comparison.json), SHA-256 `fcfd028833d1dd93408b28e4fd91073c4b32aeac7494bd4c8e932928f076979f`.

Fixture SHA-256 `e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625`. 30 identical quality rejections remain failures. Lengths10–12 have no examples in this held set; natural135 + stress270 share source crops.

Source/model/cache hashes, selected thresholds, matching rules, paired geometry statistics, all 405 decisions per model, and per-cell masks/boxes are recorded in JSON. Production/APK/physical gates are unchanged.
