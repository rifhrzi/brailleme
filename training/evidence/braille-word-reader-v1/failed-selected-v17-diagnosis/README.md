# Failed selected-v17: diagnosis mirror pada validasi405

Run22 epoch8 mirror gagal mengungguli kontrol v15: **314/405 versus357/405** blok benar. Seleksi v17 tetap memilih kontrol. Ini arsip diagnosis dari keluaran tersimpan, bukan model yang diaktifkan, APK, atau pengujian perangkat fisik.

Baca [temuan](FINDINGS.md) dan [tabel per panjang/pola](TABLES.md). Sebanyak27 kesalahan pola pantulan A/B/K/L berkaitan dengan kotak bergeser ke kiri; penjelasan ini aproksimasi dan tidak mengisolasi penyebab optimisasi/normalisasi. Matching memakai pasangan satu-ke-satu IoU >=0,25 setelah NMS, tanpa menggunakan label untuk mencocokkan. Ini bukan AP.

Seluruh405 blok berasal dari135 crop natural dan270 varian stress yang berkorelasi. Tiga puluh gambar ditolak kualitas pada kedua kandidat dan tetap dihitung gagal. Panjang10–12 dan hurufF/V tidak terwakili. Tidak ada607 eksternal, inferensi baru, pencarian ambang, atau perubahan model dalam diagnosis ini.

Ambang tetap milik masing-masing kandidat:

| Parameter | Kontrol v15 | Mirror run22 epoch8 |
|---|---:|---:|
| proposalScore | 0.25 | 0.15 |
| acceptScore | 0.25 | 0.20 |
| dotConfidence | 0.70 | 0.70 |
| nmsIou | 0.25 | 0.25 |

Seluruh ambang geometri serta identitas checkpoint ada dalam [tabel seleksi asli](checkpoint-selection.json), [laporan baseline](baseline-selected-validation-fullprecision.json), dan [laporan cache kandidat](candidate-cache-report.json). Decoder persis yang digunakan tersedia di [evaluator-source/word_decode.py](evaluator-source/word_decode.py). Dataset tetap public development validation; kriteria APK75%/95%, negatif0/44, diagnostik penolakan0/14, dan uji fisik tidak berubah.

## Membaca JSON terkompresi

[comparison.json.gz](comparison.json.gz) berisi JSON asli seluruh405 keputusan per model, enam probabilitas titik, kotak, matching, statistik, dan hash sumber. Kompresi gzip memakai level9, `mtime=0`, tanpa nama file di header. Hash JSON setelah dibuka adalah `072eb832e019ff416ac2d8fd7d3018d51f06b532a55fcd45a4b2e98db8a76363` (5,131,945 byte); hash file gzip adalah `628ba2393bbdb7005a6e7381a8b04601942f1fc696a6fd6e123e670641a95a3f` (714,381 byte).

Contoh membaca langsung tanpa ekstraksi atau inferensi:

```python
import gzip, json
from pathlib import Path
archive = Path("training/evidence/braille-word-reader-v1/failed-selected-v17-diagnosis")
with gzip.open(archive / "comparison.json.gz", "rt", encoding="utf-8") as handle:
    report = json.load(handle)
print(report["summary"]["control"]["words"])
print(report["summary"]["mirror"]["words"])
```

[pattern-followup.json](pattern-followup.json) merinci27 kasus pantulan dan16 kasus clipped yang kehilangan hasil benar. [Hash index](archive-index.json) mencatat hash/ukuran terkompresi dan asli serta asal setiap salinan; [hash index itu sendiri](archive-index.sha256) tersedia terpisah.

Dokumen asli dipertahankan byte demi byte sebagai [FINDINGS.md.txt](original-documents/FINDINGS.md.txt) dan [README.md.txt](original-documents/README.md.txt). FINDINGS/TABLES pada akar arsip menyesuaikan tautan untuk gzip; tidak ada perubahan pada laporan JSON asli. Ekstensi `.txt` pada salinan asli mencegah tautan lokasi kerja lama diperlakukan sebagai tautan arsip.

Skrip asli [compare_control.py](scripts/compare_control.py) dan [pattern_followup.py](scripts/pattern_followup.py), serta CLI [compare_held_cache.py](scripts/compare_held_cache.py), disimpan persis. Skrip asli mengacu pada tata letak workspace ketika dijalankan; replay membutuhkan cache tensor dan fixture asal yang tidak disalin ke arsip ini. CLI menerima lokasi eksplisit. [Bukti replay CLI](generic-replay-proof.json) mencatat kesetaraan810 keputusan dan3258 baris sel GT dengan diagnosis asli. Archive tidak mengandung checkpoint, tensor, atau APK.
