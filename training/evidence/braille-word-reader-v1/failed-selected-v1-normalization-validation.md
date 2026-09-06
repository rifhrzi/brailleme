# Sensitivitas normalisasi pada validasi alami

Kandidat gagal `selected-v1` diuji pada **135 blok validasi alami** yang dibekukan. Model, ambang decoder, dan koordinat pilihan tetap sama. Tidak ada gambar external test, pelatihan ulang, penyesuaian ambang, atau promosi model.

| Input | Kata utuh benar | Hasil diterima | Ketepatan hasil diterima |
| --- | ---: | ---: | ---: |
| RGB asli / 255 | 118/135 — 87,41% | 120 | 98,33% |
| Standardisasi kanal, grayscale, target std 0,05 | 117/135 — 86,67% | 118 | 99,15% |
| Standardisasi kanal, grayscale, target std 0,10 | 97/135 — 71,85% | 102 | 95,10% |
| Standardisasi kanal, grayscale, target std 0,15 | 73/135 — 54,07% | 86 | 84,88% |

Setiap kanal dikurangi mean dan dibagi sample standard deviation dengan batas minimum `1e-6`. Kanal hasil standardisasi dirata-ratakan, dikalikan target std, ditambah `0,5`, dipotong ke `[0,1]`, lalu diulang menjadi tiga kanal. Tidak ada pembulatan kembali ke uint8. Operasi ini terinspirasi dari [normalisasi AngelinaReader](https://raw.githubusercontent.com/IlyaOvodov/AngelinaReader/master/data_utils/data.py), tetapi disesuaikan dengan kontrak input model ini; bukan reproduksi persis preprocessing/model AngelinaReader.

Pemeriksaan kualitas asli dilaporkan dalam dua posisi: pada RGB semula dan setelah transformasi. Keduanya menghasilkan **nol penolakan kualitas pada semua varian**. Karena itu perbedaan pada tabel berasal dari keluaran pengenalan. Baseline menghasilkan kembali 118 kata benar yang sama dengan laporan validasi checkpoint terpilih.

Hasil ini menunjukkan bahwa normalisasi kontras yang lebih kuat mengganggu model yang dilatih memakai RGB asli. Probe target std 0,05 mendekati baseline, tetapi tidak membuktikan perbaikan. Eksperimen ini tidak menentukan hasil model yang dilatih ulang dengan normalisasi, akurasi external test, atau kamera fisik.

Inferensi memakai satu interpreter TFLite CPU dengan dua thread, GPU dinonaktifkan, prioritas Windows below normal, dan penghentian otomatis bila memori tersedia kurang dari 1 GiB. Pengujian selesai dalam 13,75 detik setelah inisialisasi; RSS maksimum yang diamati sekitar 326 MiB dan memori sistem tersedia minimum 1,06 GiB. Proses telah keluar.

Rincian per panjang kata, semua hasil individual, asumsi preprocessing, dan hash tersedia dalam [laporan JSON](failed-selected-v1-normalization-validation.json). Reproduksi dari repository:

```powershell
C:\bm-envs\angelina-benchmark\Scripts\python.exe scripts/word-normalization-sensitivity.py --run .braille-training-work/word-scan-v1-20260905/selected-v1 --data .braille-training-work/word-scan-v1-20260905/data --output training/evidence/braille-word-reader-v1/failed-selected-v1-normalization-validation.json
```
