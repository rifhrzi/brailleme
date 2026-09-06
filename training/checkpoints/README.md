# Checkpoint yang dipertahankan

Pembersihan 6 September 2026 mempertahankan model terbaik yang sudah diukur,
bobot untuk training lanjutan, konfigurasi, sumber training asli, dan bukti
evaluasi. Identitas setiap berkas tercatat dalam [retained-models.json](retained-models.json).

| Kegunaan | Checkpoint | Model aplikasi |
| --- | --- | --- |
| Pembaca satu kata, kandidat terbaik run20 epoch 09 (v15/v26) | [best-word.keras](braille-word-reader-v1/best-word.keras) | [word-reader.tflite](../../assets/models/braille-word-reader-v1/word-reader.tflite) |
| Classifier satu sel v2, untuk pengembangan jalur classifier | [best-candidate.keras](braille-cell-classifier-v2/best-candidate.keras) | [braille_model.tflite](../../assets/models/braille-cell-classifier-v2/braille_model.tflite) |

Kedua model mempunyai tugas dan benchmark berbeda. Classifier satu sel tidak
dipakai untuk menentukan pemenang benchmark kata.

## Identitas dan hasil pembaca kata

- Checkpoint: `cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7`.
- TFLite: `406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230`.
- Bobot run20 epoch 09, selected-v15, dan selected-v26 telah dibandingkan dan
  mempunyai hash yang sama; hanya satu salinan Keras disimpan di sini.
- Decoder aplikasi v2: 323/607 kata utuh benar (53,21%), 323/351 hasil diterima
  benar (92,02%). Decoder historis v15 memberi 325/607 (53,54%) pada bobot sama.
- Regresi publik yang sudah diamati, bukan holdout fisik independen. Target
  75% / 95% belum lulus; lihat [bukti model aktif](../evidence/braille-word-reader-v1/experimental-internal-best/README.md).

[Direktori run20](braille-word-reader-v1/run20/) menyimpan konfigurasi, riwayat,
protokol sampling, dan snapshot sumber tanpa mengubah byte aslinya. Path run
lama di dalam laporan adalah provenance historis; checkpoint kandidat lain
di path tersebut direncanakan dibersihkan, tetapi penghapusannya ditolak oleh
pemeriksaan otomatis dan file lokal masih ada. Catatan `RUNNING-HANDOFF.json` adalah
snapshot masa lalu dan tidak menandakan proses training masih berjalan.

## Melanjutkan training

Lingkungan lokal yang dipertahankan: Python 3.12.14 di
`C:\bm-envs\word-scan-torch-only\Scripts\python.exe`, Keras 3.15.1, Torch
2.7.1+cu128. [Catatan lingkungan](braille-word-reader-v1/environment/README.md)
dan [versi paket](braille-word-reader-v1/environment/requirements-installed.txt)
menjelaskan dependensi lokal, termasuk penggunaan paket Torch dari lingkungan
`word-scan`. Kedua lingkungan tersebut berada di luar repo dan tetap disimpan.

Jalankan dari akar repo, dengan direktori output baru:

```powershell
$env:KERAS_BACKEND = 'torch'
$trainingPython = 'C:\bm-envs\word-scan-torch-only\Scripts\python.exe'
& $trainingPython training/scripts/word_train.py `
  --data .braille-training-work/word-scan-v1-20260905/data `
  --train-manifest ../combined-training-v5/combined-training-manifest.json `
  --base-train-manifest ../complete-source-training-v3/combined-training-manifest.json `
  --supplemental-source-index ../dsbi-training-contexts-v2/source-index.json `
  --dsbi-point-metadata ../dsbi-point-metadata-v1/points.json `
  --dsbi-visible-partial-supervision `
  --resume training/checkpoints/braille-word-reader-v1/best-word.keras `
  --output .braille-training-work/word-scan-v1-20260905/continuation-from-best `
  --epochs 10 --batch-size 24 --learning-rate 0.00002 `
  --negative-repeat 27 --supplemental-negative-repeat 2 `
  --illumination-augmentation --balance-source-families `
  --retain-epochs --geometry-loss relative_size_huber
```

Perintah tersebut memulai eksperimen baru dari bobot terbaik, dengan optimizer
dan urutan sampling baru. Ini bukan pemulihan persis ke langkah training yang
terputus. Normalisasi dan arsitektur diwarisi dari checkpoint. Tentukan protokol
eksperimen baru sebelum menjalankan training; checkpoint dan ambang dipilih
pada validation, kemudian diuji lagi dengan decoder aplikasi yang berlaku.
Pembersihan ini tidak menjalankan training tambahan atau mengukur ulang akurasi.

## Data dan riwayat eksperimen

Manifest training, sumber split, dan fixture evaluasi tersimpan sebagai gzip
berhash di [data-manifests](../evidence/braille-word-reader-v1/data-manifests/README.md).
Semua delapan hash data dalam konfigurasi run20 tersedia di arsip tersebut.
Pixel cache yang dibutuhkan run20 tetap ada lokal, termasuk dua konteks
`mask16-complete-context`; direktori lokal lengkap dicatat dalam manifest retensi.

Foto dataset dan lingkungan Python tidak diunggah ke GitHub. Pada clone baru,
ambil dataset sesuai revisi tercatat, pulihkan manifest mengikuti indeks, lalu
bangun kembali pixel cache dan cocokkan hash sebelum training. Dataset asli di
`C:\bm-datasets` tetap ada pada mesin ini.

Laporan evaluasi dan diagnosis di `training/evidence/braille-word-reader-v1`
tetap disimpan untuk membandingkan percobaan. Catatan tambahan dari folder
kerja lama dipadatkan dalam [experiment-notes-20260906.zip](../evidence/braille-word-reader-v1/experiment-notes-20260906.zip).
Arsip itu berisi konfigurasi, sumber, hasil JSON, dan log historis; checkpoint
yang kalah, paket dependency, dan cache tensor tidak disertakan. Ekstrak ke
direktori terpisah bila membutuhkan catatan lama.
