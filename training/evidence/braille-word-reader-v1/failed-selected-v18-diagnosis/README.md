# Failed selected-v18: diagnosis vertical stretch

Run23 epoch9 dan kontrol v15 sama-sama benar pada **357 dari 405 blok**. Kontrol tetap dipilih karena skor kelompok natural/stress yang dibobot sama lebih tinggi. Arsip ini terpisah dari bukti seleksi v18 lainnya dan tidak mengubah model aktif.

Baca [temuan](FINDINGS.md) dan [tabel per panjang/pola](TABLES.md). Satu blok natural `ZWUKOW` menjadi penolakan confidence; satu varian stress `WRAQ` membaik setelah sel A yang terlewat ditemukan. Analisis menggunakan cache tersimpan dan matching satu-ke-satu berbasis IoU >=0,25, tanpa label untuk menentukan pasangan. Ini diagnosis aproksimasi, bukan AP.

Set ini berisi 135 crop natural dan 270 varian stress yang berkorelasi, bukan 405 sumber independen. Tiga puluh penolakan kualitas dihitung gagal pada kedua model. Panjang 10–12 dan huruf F/V tidak terwakili. Tidak ada inferensi baru, external607, pencarian ambang, APK, atau bukti perangkat fisik.

Ambang kontrol dan kandidat persis sama:

| Parameter | Kontrol v15 | Stretch run23 epoch9 |
|---|---:|---:|
| proposalScore | 0.25 | 0.25 |
| acceptScore | 0.25 | 0.25 |
| dotConfidence | 0.7 | 0.7 |
| nmsIou | 0.25 | 0.25 |
| minCellWidth | 0.008 | 0.008 |
| minCellHeight | 0.06 | 0.06 |
| edgeMargin | 0.008 | 0.008 |
| maxRowDeviation | 0.55 | 0.55 |
| maxGapRatio | 1.2 | 1.2 |
| maxOverlapRatio | 0.5 | 0.5 |

Identitas checkpoint dan hasil seleksi asli tersimpan dalam [checkpoint-selection.json](checkpoint-selection.json), [laporan kontrol](baseline-selected-validation-fullprecision.json), [laporan v18](selected-v18-validation-fullprecision.json), dan [metadata cache kandidat](candidate-cache-report.json). [Decoder yang diarsipkan](evaluator-source/word_decode.py) tetap sama untuk kedua kandidat. Kriteria APK 75%/95%, negatif 0/44, diagnostik penolakan 0/14, dan uji fisik tidak berubah.

## Membaca laporan lengkap

[comparison.json.gz](comparison.json.gz) berisi seluruh 405 keputusan per model, kotak, probabilitas titik, matching, dan hash sumber/fixture/cache/checkpoint. Gzip deterministik memakai level 9, `mtime=0`, tanpa nama file dalam header. Ukuran asli 5,123,211 byte; SHA-256 setelah dekompresi `2447c27fc97981b448e0b7dc8dc3e570981a11e187b352088fcfb93119e723d8`. Gzip berukuran 709,942 byte; SHA-256 `e2d0de7b620b083ace5f051ee1a4272b712b7d28232b65844091e7c967c36bcc`.

```python
import gzip, json
from pathlib import Path
archive = Path("training/evidence/braille-word-reader-v1/failed-selected-v18-diagnosis")
with gzip.open(archive / "comparison.json.gz", "rt", encoding="utf-8") as handle:
    report = json.load(handle)
print(report["summary"]["baseline"]["words"])
print(report["summary"]["candidate"]["words"])
```

[archive-index.json](archive-index.json) memuat ukuran/hash setiap file dan asal salinannya, termasuk hash mentah dan gzip. [Hash index](archive-index.sha256) disimpan terpisah. [Log eksekusi](execution.log) berasal dari replay cache yang sudah selesai; pengarsipan tidak menjalankan ulang diagnosis.

Byte dokumen asli dipertahankan sebagai [README.md.txt](original-documents/README.md.txt) dan [FINDINGS.md.txt](original-documents/FINDINGS.md.txt). Dokumen presentasi di akar arsip hanya memperbarui tautan dan keterangan gzip. Semua JSON asli tetap identik setelah dekompresi.

[Skrip diagnosis asli](scripts/compare_held_cache.py) disimpan persis; untuk replay ia memerlukan cache tensor, checkpoint, dan fixture workspace yang tidak disalin ke arsip ini. Tidak ada tensor, model, atau APK dalam arsip.
