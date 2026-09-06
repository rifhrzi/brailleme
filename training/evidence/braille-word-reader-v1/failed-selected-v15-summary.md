# Ringkasan kandidat selected-v15

**Gerbang angka offline: GAGAL.** Kata utuh benar 325/607 (53.54%); hasil diterima benar 325/353 (92.07%).

Regresi pengembangan publik yang telah diamati; bukan holdout fisik independen. Seluruh penolakan tetap masuk penyebut keberhasilan kata utuh. Presisi hanya memakai hasil yang diterima; N/A berarti tidak ada hasil diterima.

| Panjang | Total | Diterima | Benar utuh | Kata utuh % | Presisi diterima % | Ditolak |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 152 | 96 | 94 | 61.84% | 97.92% | 56 |
| 2 | 109 | 74 | 63 | 57.80% | 85.14% | 35 |
| 3 | 73 | 50 | 43 | 58.90% | 86.00% | 23 |
| 4 | 66 | 40 | 37 | 56.06% | 92.50% | 26 |
| 5 | 70 | 32 | 31 | 44.29% | 96.88% | 38 |
| 6 | 43 | 21 | 19 | 44.19% | 90.48% | 22 |
| 7 | 35 | 14 | 14 | 40.00% | 100.00% | 21 |
| 8 | 24 | 9 | 8 | 33.33% | 88.89% | 15 |
| 9 | 15 | 6 | 6 | 40.00% | 100.00% | 9 |
| 10 | 13 | 6 | 6 | 46.15% | 100.00% | 7 |
| 11 | 5 | 5 | 4 | 80.00% | 80.00% | 0 |
| 12 | 2 | 0 | 0 | 0.00% | N/A | 2 |
| **Total** | **607** | **353** | **325** | **53.54%** | **92.07%** | **254** |

| Pemeriksaan | Hasil | Syarat | Status |
| --- | --- | --- | --- |
| Kata utuh | 53.54% | >=75% | Gagal |
| Presisi hasil diterima | 92.07% | >=95% | Gagal |
| Negatif publik diterima | 0/44 | 0/44 | Lulus |
| Kasus penolakan diterima | 2/14 | 0/14 | Gagal |
| Paritas ekspor | 405 input | Tensor dan keputusan sesuai | Lulus |

Gerbang angka tidak menyatakan fitur selesai atau APK siap diserahkan. Model/config, native runtime pada APK yang sama, signing, smoke test, dan pengujian perangkat tetap memerlukan bukti masing-masing.

| Alasan penolakan 607 kata | Jumlah |
| --- | --- |
| clipped | 162 |
| gap | 8 |
| low_confidence | 58 |
| no_grid | 11 |
| overlapping_cells | 2 |
| unsupported_pattern | 13 |

44 gambar negatif publik: 44 ditolak, 0 diterima. Gambar ini juga dipakai training/regresi; tidak dianggap holdout independen.
Alasan negatif: no_grid=44.

66 blok eksperimen lama: 20 benar, 28 diterima, 38 ditolak. Blok ini tidak otomatis merupakan kata utuh dan tidak digabung ke penyebut 607 kata.
Alasan legacy: clipped=23, gap=6, low_confidence=7, overlapping_cells=1, unsupported_pattern=1.

Komposisi berulang: 11/12 benar, 11 diterima. Kasus penolakan: 12/14 ditolak dengan benar, 2 masih diterima. Kedua kelompok diagnostik terpisah dari benchmark kata natural dan tidak ditambahkan ke skor 607.

| Diagnostik yang gagal | Target | Hasil | Alasan |
| --- | --- | --- | --- |
| style-03-mamamamamama.png | MAMAMAMAMAMA | Ditolak | clipped |
| selection_cut-01.png | Tolak | KAJDO | - |
| selection_cut-02.png | Tolak | BLYCS | - |

Validation: 357/405 benar (88.15%), 357/362 hasil diterima benar (98.62%). Jumlah validation diperiksa terhadap manifest dan agregat; generator ini tidak menjalankan ulang tensor atau mengklaim audit 405 baris keluaran validation.

Status artifact pada ekspor: `unverified_candidate`. `fixture_human_boundary_reviewed`: `false`. Flag tersebut disalin apa adanya dari bukti beku; generator tidak mengesahkan pemeriksaan batas manusia.

APK internal: belum disertakan; tidak ada klaim APK utama selesai.
Perangkat fisik: pending; bukti pengujian belum disertakan.

| Ambang kandidat | Nilai |
| --- | --- |
| proposalScore | 0.25 |
| acceptScore | 0.25 |
| dotConfidence | 0.7 |
| nmsIou | 0.25 |
| minCellWidth | 0.008 |
| minCellHeight | 0.06 |
| edgeMargin | 0.008 |
| maxRowDeviation | 0.55 |
| maxGapRatio | 1.2 |
| maxOverlapRatio | 0.5 |

| Identitas | SHA-256 |
| --- | --- |
| Model TFLite aktual | `406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230` |
| Checkpoint aktual | `cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7` |
| Fixture external (byte JSON setelah dekompresi jika gzip) | `e69d7c0d9582aef7d51a2c15706f6bf62bc3cdfe35d22ad28ad007301d56b41b` |
| Fixture validation (byte JSON setelah dekompresi jika gzip) | `e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625` |
| Fixture repetition (byte JSON setelah dekompresi jika gzip) | `bdf89716ab644baa8f7e0a47c90f058e5da5ccfb594c9403070760f86819c41c` |
| Fixture rejection (byte JSON setelah dekompresi jika gzip) | `4d92222c0fe73691efaeeba278252741b739cd04491121abb5ab579e25170f89` |
| [public-benchmark.json](<failed-selected-v15/public-benchmark.json>) | `e564506480706d74826a9af404b943bacac77b15a5a0eeb8b9da5b7d073b0b19` |
| [export-report.json](<failed-selected-v15/export-report.json>) | `b709a6cc4bf2b45a2e37c7633ee3925235af2083fbf8f501195143b06b1bfd47` |
| [checkpoint-selection.json](<failed-selected-v15/checkpoint-selection.json>) | `ba40ae7f9243f488e729e16a85f8f5cdb0a8079e55bac6c0eb4e689d33d9d5bd` |
| [selected-validation-fullprecision.json](<failed-selected-v15/selected-validation-fullprecision.json>) | `d136021888194895e3bd19cd3800ace90cb3cc83776a87799fd7fda8d9623b39` |
| [repetition-diagnostics.json](<failed-selected-v15/repetition-diagnostics.json>) | `f1353fe48d20d9a37e5c233666a581289a0177b18db4fa56c225d3a5bedf421a` |
| [rejection-diagnostics.json](<failed-selected-v15/rejection-diagnostics.json>) | `c9130e004cf12cc8e5e51ccee6081c3d6d2807fe965c4ae0ba7dd26c88b21901` |
| [Generator](<../../scripts/word_report.py>) | `b01b70eda691c407f4945ca765f593db1957533d96cd8b88d184346126abd1dc` |

Ringkasan dihitung dari baris final yang telah diterima/ditolak, lalu dicocokkan dengan agregat, fixture, model/checkpoint aktual, selection, export, dan diagnostik. Tidak ada inferensi, perubahan threshold, atau penulisan ulang laporan sumber.
