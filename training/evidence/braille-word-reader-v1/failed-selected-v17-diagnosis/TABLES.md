# Run22 mirror versus v15: cached held405 diagnosis

Approximate box matching; not AP or independent physical validation. No model execution, threshold search, or external607 data.

| Model | Correct / 405 | Accepted | Rejected | Accepted precision |
|---|---:|---:|---:|---:|
| control | 357 | 362 | 43 | 98.62% |
| mirror | 314 | 325 | 80 | 96.62% |

Lost correct blocks: 45; gained: 2. Lost outcomes: `{"accepted_wrong": 6, "clipped": 16, "low_confidence": 10, "unsupported_pattern": 13}`.

Both models use their existing selected thresholds; the mirror proposal/accept thresholds are lower. Matching is greedily one-to-one at IoU >=0.25, without reading labels. Unmatched predictions are not automatically false positives.

| Length | Blocks | Control correct | Mirror correct | Control unmatched GT | Mirror unmatched GT |
|---|---:|---:|---:|---:|---:|
| 1 | 87 | 74 | 66 | 9 | 9 |
| 2 | 63 | 57 | 46 | 10 | 10 |
| 3 | 51 | 45 | 39 | 14 | 13 |
| 4 | 33 | 31 | 31 | 5 | 5 |
| 5 | 30 | 26 | 24 | 14 | 13 |
| 6 | 69 | 62 | 51 | 36 | 36 |
| 7 | 24 | 18 | 18 | 15 | 14 |
| 8 | 33 | 29 | 27 | 8 | 8 |
| 9 | 15 | 15 | 12 | 0 | 0 |
| 10 | 0 | 0 | 0 | 0 | 0 |
| 11 | 0 | 0 | 0 | 0 | 0 |
| 12 | 0 | 0 | 0 | 0 | 0 |

| Pattern | GT cells | Control matched / correct mask | Mirror matched / correct mask |
|---|---:|---:|---:|
| A (1) | 144 | 129 / 126 | 129 / 116 |
| B (3) | 18 | 16 / 16 | 16 / 14 |
| C (9) | 6 | 6 / 6 | 6 / 6 |
| D (25) | 51 | 45 / 45 | 45 / 44 |
| E (17) | 129 | 119 / 119 | 119 / 119 |
| G (27) | 15 | 12 / 11 | 13 / 13 |
| H (19) | 3 | 3 / 3 | 3 / 3 |
| I (10) | 138 | 130 / 130 | 130 / 130 |
| J (26) | 48 | 43 / 43 | 43 / 43 |
| K (5) | 102 | 95 / 95 | 95 / 83 |
| L (7) | 42 | 40 / 40 | 40 / 35 |
| M (13) | 81 | 75 / 75 | 75 / 75 |
| N (29) | 105 | 97 / 97 | 97 / 89 |
| O (21) | 186 | 174 / 174 | 174 / 172 |
| P (15) | 57 | 53 / 53 | 54 / 52 |
| Q (31) | 36 | 36 / 36 | 36 / 36 |
| R (23) | 81 | 77 / 76 | 78 / 77 |
| S (14) | 93 | 86 / 86 | 86 / 86 |
| T (30) | 54 | 53 / 52 | 53 / 52 |
| U (37) | 84 | 80 / 80 | 80 / 78 |
| W (58) | 108 | 104 / 102 | 104 / 104 |
| X (45) | 9 | 7 / 7 | 7 / 7 |
| Y (61) | 3 | 3 / 3 | 3 / 3 |
| Z (53) | 36 | 35 / 35 | 35 / 35 |

Full row-level diagnosis (gzip; the following hash is for the decompressed JSON): [comparison.json.gz](comparison.json.gz), SHA-256 `072eb832e019ff416ac2d8fd7d3018d51f06b532a55fcd45a4b2e98db8a76363`.

Fixture SHA-256 `e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625`. Thirty identical quality rejections remain failures. Lengths10–12 have no examples in this held set; natural135 + stress270 share source crops.

Source/model/cache hashes, selected thresholds, matching rules, paired geometry statistics, all 405 decisions per model, and per-cell masks/boxes are recorded in JSON. Production/APK/physical gates are unchanged.
