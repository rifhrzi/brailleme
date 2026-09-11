# Hasil eksperimen kapital, angka, dan tanda baca

**Target 80% belum tercapai. Kandidat dipertahankan untuk riset dan tidak
dipromosikan ke aplikasi.** Model baru membaik pada komposisi kapital, angka,
dan tanda baca, tetapi masih gagal syarat akurasi/ketepatan dan menurunkan
hasil huruf lama. Model aktif tetap memakai baseline yang dipertahankan.

## Target dan data

Target: minimal **80% teks utuh** pada setiap kelompok positif dan subkelompok
lebih dari satu karakter. Penolakan contoh positif tetap dihitung salah.
Ketepatan hasil diterima minimal 95% dan seluruh 40 rangkaian invalid harus
ditolak, dilaporkan sebagai syarat terpisah.

TRAIN berisi 6.096 komposisi, validation 540, dan TEST 540 komposisi dari 18
halaman DSBI TEST yang sebelumnya tidak digunakan untuk gradien/pemilihan
checkpoint. Ini komposisi potongan foto publik, bukan 540 foto kata Indonesia
alami atau hasil kamera ponsel. Sebanyak 96% contoh positif memuat lebih dari
satu karakter. Revisi, hash manifest, dan batas cakupan terdapat pada
[README eksperimen](../../README-ACCURACY80-ID.md).

## Metode yang dijalankan

Checkpoint lama dilanjutkan melalui tiga tahap: pembaca keluaran dengan backbone
dibekukan; fine-tuning dengan penekanan indikator/tanda baca; lalu penguatan loss
titik dan sampling contoh TRAIN sulit. Setengah batch tetap memakai data lama.
Donor kapital memakai empat transformasi dengan permutasi bit yang benar.
Threshold sama dan dibekukan di seluruh perbandingan.

Tahap pertama menyelesaikan 10 epoch dan memilih epoch 10. Tahap kedua
menyelesaikan 23 epoch dan memilih epoch 19. Tahap ketiga menyelesaikan 48 epoch
dan memilih epoch 45: **81 epoch lengkap dipertahankan**. SHA-256 detektor
terpilih: `090260a2a29f802cdcd07140344feaaf04d98bc48cd2be7a5d75e0482d6ee38c`.
Validation detektor terpilih mencapai rata-rata lima kelompok 65,4%, dengan
kelompok terendah 61%; ini bukan hasil TEST dan belum memenuhi target 80%.

Pilot pembaca sel tambahan selesai 17 epoch dalam 29,64 detik, memakai detektor
beku. TRAIN memakai potongan sel dari kotak anotasi; evaluasi memakai potongan
dari prediksi detektor. Pilot tidak memenuhi kriteria
perbaikan yang ditetapkan, sehingga tidak dipakai. Pilihan pilot dengan
ketepatan tinggi justru menurunkan rata-rata exact validation menjadi 57% dan
nilai terendah menjadi 43,16%. Hasil ini dicatat dalam
[pemilihan pilot](refiner-pilot-selection.json).

[Keputusan metode final](final-method-selection.json) membekukan detektor tunggal
epoch 45 dan parser `six-dot-id-v4-experimental` sebelum TEST atau regresi akhir
dibuka. Tidak ada tuning lanjutan setelah reservasi TEST.

## Hasil akhir pada TEST yang sama

Angka berikut berasal dari **inferensi file TFLite aktual** pada 540 komposisi
TEST yang sama. Baseline dan kandidat memakai parser v4, filter kualitas dan
threshold identik yang sudah dibekukan. Sumber:
[perbandingan artefak TEST](test-artifact-comparison-summary.json).

| Kelompok | Baseline: benar/total | Kandidat: benar/total | Kandidat multi-karakter | Ketepatan hasil diterima |
| --- | --- | --- | --- | --- |
| Huruf kecil | 9/100 (9%) | 71/100 (71%) | 68/95 (71,58%) | 71/79 (89,87%) |
| Kapital awal | 1/100 (1%) | 73/100 (73%) | 69/95 (72,63%) | 73/83 (87,95%) |
| Seluruh kata kapital | 1/100 (1%) | 69/100 (69%) | 64/95 (67,37%) | 69/82 (84,15%) |
| Angka | 2/100 (2%) | 78/100 (78%) | 73/95 (76,84%) | 78/84 (92,86%) |
| Tanda baca | 6/100 (6%) | 59/100 (59%) | 59/100 (59%) | 59/72 (81,94%) |

- Rata-rata lima kelompok positif: **350/500 (70%)** kandidat,
  **19/500 (3,8%)** baseline. Target
  80% pada setiap kelompok dan multi-karakter **gagal**.
- Ketepatan hasil diterima kandidat pada seluruh contoh positif:
  **350/400 (87,5%)**. Syarat minimal 95% pada setiap kelompok **gagal**.
- Rangkaian invalid yang keliru diterima: **1/40** kandidat, **9/40** baseline;
  syarat nol penerimaan salah **gagal**.

### Rincian tanda baca

Setiap angka menilai **seluruh string yang mengandung tanda**, bukan hanya
keberhasilan mengenali satu sel tanda baca.

| Tanda | Teks utuh benar/total | Benar/hasil diterima |
| --- | ---: | ---: |
| `.` | 4/12 (33,33%) | 4/7 |
| `,` | 7/11 (63,64%) | 7/8 |
| `?` | 6/11 (54,55%) | 6/9 |
| `!` | 6/11 (54,55%) | 6/7 |
| `:` | 6/11 (54,55%) | 6/7 |
| `;` | 8/11 (72,73%) | 8/9 |
| `-` | 5/11 (45,45%) | 5/6 |
| `(` dan `)` berpasangan | 17/22 (77,27%) | 17/19 |

Kedua tanda kurung muncul pada 22 string yang sama; jumlah tersebut tidak
digandakan saat menghitung total corpus.

Kombinasi kapital/angka dengan tanda baca belum memiliki kelompok evaluasi
tersendiri. Desimal, matematika, kalimat, dan dokumen penuh di luar cakupan.

## Regresi huruf lama dan keputusan promosi

Perbandingan berikut menggunakan decoder v4 dan threshold yang sama. Corpus
607 dan 405 sudah diamati selama pengembangan, sehingga merupakan pemeriksaan
regresi, bukan holdout baru. Inferensi kandidat memakai checkpoint native Torch;
baseline memakai output tersimpan yang hash-nya terikat model/data lama.
Sumber: [regresi publik](public-regression-summary.json).

| Evaluasi | Baseline | Kandidat |
| --- | ---: | ---: |
| Teks benar pada 607 crop publik | 323/607 (53,21%) | 309/607 (50,91%) |
| Ketepatan hasil diterima pada 607 | 323/353 (91,50%) | 309/360 (85,83%) |
| Teks benar pada validation lama 405 | 355/405 (87,65%) | 318/405 (78,52%) |
| Negatif non-Braille yang keliru diterima | 0/44 | 0/44 |

Pada 607 crop, 76 contoh yang sebelumnya benar menjadi salah/ditolak, sedangkan
62 membaik: berkurang 14 teks benar. Ketepatan baseline 91,50% dihitung ulang
dengan v4 pada perbandingan ini; gunakan angka itu untuk membandingkan kandidat.
Negatif 44 merupakan data lama yang digunakan untuk training/regresi.

**Promosi dibatalkan** karena target baru belum lulus dan kedua corpus huruf
lama mengalami penurunan. Tidak ada pemilihan checkpoint pengganti atau
penyesuaian threshold berdasarkan TEST/regresi akhir. Bobot aplikasi tidak
diganti; checkpoint baru tetap disimpan untuk pengembangan berikutnya.

## Konversi, pengujian, dan artefak

[Paritas export](export-parity.json) lulus pada seluruh 540 input validation:
Torch/TensorFlow/TFLite menghasilkan keputusan semantik berurutan yang sama,
termasuk sesudah filter kualitas. Selisih maksimum Torch/TensorFlow
`8,1986e-5` dan Torch/TFLite `1,4174e-4` memenuhi pemeriksaan gabungan
`atol=1e-4, rtol=1e-4`. Nilai maksimum kedua bukan klaim bahwa seluruh selisih
absolut lebih kecil dari `1e-4`.

Pada TEST, ringkasan keputusan native Torch/TFLite sama **540/540** untuk
identitas sumber, kelompok, expected/predicted text, alasan, dan jumlah sel.
Laporan native TEST menyimpan ringkasan, sehingga paritas seluruh mask/peran/
keluaran sel pada TEST tidak diklaim; paritas semantik lengkap tersebut diuji
pada validation export.

[86 pengujian kode](qualification-tests.json) lulus tanpa kegagalan, error atau
skip. `npm.cmd run model:verify` juga lulus pemeriksaan integritas model aktif;
[log-nya](final-deployed-model-verify.log) tetap mempertahankan kegagalan target
akurasi yang tercatat. Lulus tes kode/integritas bukan lulus akurasi atau rilis.

[Audit akhir](final-audit.json) memeriksa hitungan hasil, hash dan kronologi
tanpa inferensi ulang: 1.751 path terikat bukti dan 224 referensi file dalam
indeks retensi lama lolos. Tidak ditemukan tuning setelah TEST; folder build
terdaftar tetap 1 dari maksimum 2. [Pemeriksaan corpus akhir](final-corpus-integrity.json)
mengonfirmasi hash seluruh 7.176 gambar tetap sama.

| Bukti final | Status |
| --- | --- |
| Run/epoch final | `run-dots`, epoch 45; refiner tidak dipakai |
| Checkpoint tersimpan | [best-word.keras](../../checkpoints/braille-word-reader-id-v4/best-word.keras), 4.120.098 byte |
| SHA-256 checkpoint | `090260a2a29f802cdcd07140344feaaf04d98bc48cd2be7a5d75e0482d6ee38c` |
| TFLite tersimpan | [word-reader.tflite](../../checkpoints/braille-word-reader-id-v4/word-reader.tflite), 3.239.392 byte |
| SHA-256 TFLite | `e82a540e7b2e1c8d5b3bd5c3004e1cc5409367293c27b24f13f2a7c05f015325` |
| Bukti evaluasi final | [reservasi TEST](test-evaluation-reservation.json), [native TEST](test-evaluation-summary.json), [indeks hasil](final-test-evaluation.json) |
| Keputusan | [Kualifikasi final](../../checkpoints/braille-word-reader-id-v4/qualification.json): **tidak dipromosikan; target belum tercapai dan regresi huruf lama** |
| Retensi tambahan | [retained-models-20260910.json](../../checkpoints/retained-models-20260910.json), melengkapi indeks lama |

Bukti kerja berada di `C:\bm-symbols-20260908\accuracy80-20260910`.
`corpus-audit.json` dan `corpus-verification.json` sudah mencatat pemisahan sumber,
transformasi, kotak sel, dan hash 7.176 PNG. Seluruh 81 epoch detektor dan 17
epoch pilot, snapshot kode, dataset serta lingkungan training dipertahankan.
Training ini tidak membuat folder build/staging Android baru atau APK baru.

[EVIDENCE-COPIES.json](EVIDENCE-COPIES.json) mencatat salinan ringkas laporan
beserta SHA-256 sumber asli. Baris keputusan lengkap, oracle NPZ dan gambar
tetap berada dalam direktori percobaan yang dilindungi.

Hasil ini terbatas pada komposisi publik. Tepi potongan, pencahayaan antar sel
dan distribusi sumber berbeda dari kata pada kertas asli. Belum ada pembuktian
akurasi kamera pada kertas Braille fisik atau validasi native perangkat untuk
kandidat ini. Statusnya artefak riset, belum siap produksi.
