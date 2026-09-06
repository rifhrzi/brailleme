# Current candidate versus baseline: cached held405 diagnosis

Approximate box matching; not AP or independent physical validation. No model execution, threshold search, or external607 data.

| Model | Correct / 405 | Accepted | Rejected | Accepted precision |
|---|---:|---:|---:|---:|
| baseline | 357 | 362 | 43 | 98.62% |
| candidate | 357 | 361 | 44 | 98.89% |

Lost correct blocks: 1; gained: 1. Lost outcomes: `{"low_confidence": 1}`.

Both models use their existing selected thresholds, recorded in JSON; no thresholds are searched or changed. Matching is greedily one-to-one at IoU >=0.25, without reading labels. Unmatched predictions are not automatically false positives.

| Length | Blocks | Baseline correct | Candidate correct | Baseline unmatched GT | Candidate unmatched GT |
|---|---:|---:|---:|---:|---:|
| 1 | 87 | 74 | 74 | 9 | 9 |
| 2 | 63 | 57 | 57 | 10 | 10 |
| 3 | 51 | 45 | 45 | 14 | 14 |
| 4 | 33 | 31 | 32 | 5 | 4 |
| 5 | 30 | 26 | 26 | 14 | 14 |
| 6 | 69 | 62 | 61 | 36 | 36 |
| 7 | 24 | 18 | 18 | 15 | 15 |
| 8 | 33 | 29 | 29 | 8 | 8 |
| 9 | 15 | 15 | 15 | 0 | 0 |
| 10 | 0 | 0 | 0 | 0 | 0 |
| 11 | 0 | 0 | 0 | 0 | 0 |
| 12 | 0 | 0 | 0 | 0 | 0 |

| Pattern | GT cells | Baseline matched / correct mask | Candidate matched / correct mask |
|---|---:|---:|---:|
| A (1) | 144 | 129 / 126 | 130 / 127 |
| B (3) | 18 | 16 / 16 | 16 / 16 |
| C (9) | 6 | 6 / 6 | 6 / 6 |
| D (25) | 51 | 45 / 45 | 45 / 44 |
| E (17) | 129 | 119 / 119 | 119 / 119 |
| G (27) | 15 | 12 / 11 | 12 / 10 |
| H (19) | 3 | 3 / 3 | 3 / 3 |
| I (10) | 138 | 130 / 130 | 130 / 130 |
| J (26) | 48 | 43 / 43 | 43 / 43 |
| K (5) | 102 | 95 / 95 | 95 / 95 |
| L (7) | 42 | 40 / 40 | 40 / 40 |
| M (13) | 81 | 75 / 75 | 75 / 75 |
| N (29) | 105 | 97 / 97 | 97 / 97 |
| O (21) | 186 | 174 / 174 | 174 / 174 |
| P (15) | 57 | 53 / 53 | 53 / 53 |
| Q (31) | 36 | 36 / 36 | 36 / 36 |
| R (23) | 81 | 77 / 76 | 77 / 76 |
| S (14) | 93 | 86 / 86 | 86 / 86 |
| T (30) | 54 | 53 / 52 | 53 / 52 |
| U (37) | 84 | 80 / 80 | 80 / 80 |
| W (58) | 108 | 104 / 102 | 104 / 103 |
| X (45) | 9 | 7 / 7 | 7 / 7 |
| Y (61) | 3 | 3 / 3 | 3 / 3 |
| Z (53) | 36 | 35 / 35 | 35 / 35 |

Full row-level diagnosis (gzip; the following hash is for the decompressed JSON): [comparison.json.gz](comparison.json.gz), SHA-256 `2447c27fc97981b448e0b7dc8dc3e570981a11e187b352088fcfb93119e723d8`.

Fixture SHA-256 `e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625`. 30 identical quality rejections remain failures. Lengths10–12 have no examples in this held set; natural135 + stress270 share source crops.

Source/model/cache hashes, selected thresholds, matching rules, paired geometry statistics, all 405 decisions per model, and per-cell masks/boxes are recorded in JSON. Production/APK/physical gates are unchanged.
