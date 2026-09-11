# Hasil eksperimen model, 8 September 2026

**Epoch 10 pada laporan ini tidak dikirim dalam APK.** Ia terpilih oleh
validation training, lalu promosinya dibatalkan karena regresi eksternal yang
dikonfirmasi menggunakan decoder yang sama: 323/607 menjadi 306/607. APK
mempertahankan bobot baseline dengan decoder baru. Angka pada dokumen ini
adalah hasil kandidat penelitian; untuk hasil model APK, gunakan
[laporan deployment baseline](../indonesian-symbols-baseline-20260908/EXPERIMENT-ID.md).
Seluruh JSON dan checkpoint kandidat tetap dipertahankan tanpa perubahan.

**Status: eksperimen internal; target akurasi belum terpenuhi.** Aturan kapital
dan angka sudah diterapkan serta menghasilkan keputusan yang sama pada Python
dan TypeScript. Pengenalan indikator kapital oleh model masih lemah dan sejumlah
gambar tidak valid masih diterima. Hasil ini bukan bukti akurasi kamera pada
kertas Braille fisik.

## Training dan pemilihan

Satu putaran selesai tepat **10 epoch**, batch **24**, learning rate awal
**0.00002**, tanpa OOM atau percobaan tambahan. Arsitektur, tensor enam titik,
serta threshold baseline dipertahankan. Semua epoch disimpan. Kandidat terpilih
adalah **epoch 10**, dengan checkpoint SHA-256:

`0ae8497660efa3378d81b959cf585f83c6af3c07514f45fb2b1fb1a90382a6c2`

Pemilihan terlebih dahulu melarang penurunan akurasi teks utuh atau ketepatan
hasil diterima pada dua kelompok validation lama yang dibekukan: **natural135**
dan **stress270**. Karena tidak ada kandidat yang memenuhi semua target,
pembanding berikutnya adalah ketepatan kelompok terlemah. Epoch 10 memenuhi
aturan pemilihan ini, tetapi **tidak memenuhi target kesiapan**.

| Validation huruf lama | Baseline: benar/total | Epoch 10: benar/total | Ketepatan hasil diterima: baseline → epoch 10 |
|---|---:|---:|---:|
| Natural | 129/135 | 130/135 | 100% → 100% |
| Stress | 226/270 | 229/270 | 98.26% → 98.28% |

Detail seluruh kandidat dan urutan pemilihan tersedia pada
[checkpoint-selection.json](checkpoint-selection.json). Hasil 405 input ini
adalah validation pemilihan model, bukan evaluasi independen setelah pemilihan.

## Data simbol dan hasil aktual TFLite

Data sumber AngelinaDataset dipulihkan pada revisi
`e13038d483bb58e17046d640e078dc4768276e9b`; DSBI TRAIN pada revisi
`6937eb8973efe5ff337e5c018485cb626303f5c2`. DSBI TEST tidak digunakan. Hash
sumber DSBI cocok dengan manifest asli yang dipertahankan.

Data tambahan terdiri dari **3.096 komposisi training** dan **516 komposisi
validation**, dibuat dari potongan foto publik. Pool simbol berasal dari 296
halaman training dan 14 halaman validation; pembagian halaman mendahului crop
dan augmentasi. **516 komposisi bukan 516 foto independen.** Data eksternal
607 contoh tidak masuk gradien atau pemilihan checkpoint/threshold.

Pola titik 6 dan 3456 dari sumber Rusia/matematika hanya digunakan sebagai
bentuk enam titik; makna kapital/angka Indonesia ditetapkan pada komposisi
sintetis baru. Pool validation asli tidak memiliki titik 6. Pola yang tidak
tersedia dilengkapi dengan refleksi/rotasi potongan validation dan transformasi
bit yang dicatat eksplisit. Tidak ada foto kata kapital Indonesia alami atau
sampel kertas fisik baru dalam evaluasi ini.

| Kelompok komposisi | Teks benar/total | Akurasi teks utuh | Benar/hasil diterima | Ketepatan hasil diterima |
|---|---:|---:|---:|---:|
| Huruf kecil | 111/140 | 79.29% | 111/113 | 98.23% |
| Kapital tunggal | 10/140 | 7.14% | 10/107 | 9.35% |
| Seluruh kata kapital | 11/140 | 7.86% | 11/81 | 13.58% |
| Rangkaian angka | 54/64 | 84.38% | 54/55 | 98.18% |

Dari 32 rangkaian yang wajib ditolak, **9 masih diterima**. Huruf kecil dan
angka mencapai target pada komposisi ini; kapital dan penolakan belum. Model
dapat melewatkan indikator sehingga teks yang lolos aturan tetap salah.
Lihat [symbol-readiness.json](symbol-readiness.json),
[corpus-audit.json](corpus-audit.json), dan
[synthetic-transform-audit.json](synthetic-transform-audit.json).

## Regresi publik dan kasus diagnostik

Evaluasi ulang pada 607 potongan kata publik menghasilkan **306/607 teks benar
(50.41%)** dan **306/332 hasil diterima benar (92.17%)**. Hasil historis model
sebelumnya adalah 323/607 (53.21%) dan 323/351 (92.02%). Ada penurunan 17 teks
utuh benar, sekitar 2.80 poin persentase. Temuan eksternal ini dilaporkan sebagai
keterbatasan; tidak digunakan untuk retuning atau memilih epoch lain.

Corpus 607 sudah pernah diamati selama pengembangan sebelumnya, sehingga
**bukan holdout independen**. Geometri/crop kata dipertahankan; expected text
huruf tanpa indikator diturunkan menjadi lowercase sesuai decoder baru.

- Negatif publik: **44/44 ditolak**, juga merupakan data regresi/training.
- Blok publik lama: **23/66 benar**, dibanding historis 20/66; bukan benchmark
  kata utuh.
- Pengulangan sel: **12/12 benar**.
- Diagnostik penolakan baru: **13/14 benar**; satu gambar `selection_cut` masih
  diterima. Semua 3 unsupported, 3 multirow, 3 over-limit, dan 2 blank ditolak.
- Dua crop foto publik berlabel angka `66` dan `67`: **2/2 benar**. Kedua crop
  sudah diketahui dari pengembangan sebelumnya, bukan holdout baru. Keduanya
  dipindahkan dari asumsi negatif lama karena angka sekarang didukung.

## Konversi, kesetaraan, dan preservasi

TFLite baru berukuran **3.259.716 byte**, menggunakan operator bawaan saja,
dengan SHA-256:

`659a235c0ee32f6ee59b7388dc278abcbba4711b8c742380619f3b7a1682f715`

Konversi dijalankan ulang dari checkpoint terpilih. Pada 405 input, keputusan
raw dan setelah pemeriksaan kualitas identik untuk Torch, TensorFlow, serta
TFLite. Selisih numerik maksimum masing-masing pasangan sekitar 2.52724e-5,
5.11408e-5, dan 5.07236e-5, semuanya di bawah batas 5e-4.

Replay aktual TypeScript terhadap tensor/piksel TFLite cocok **405/405** dan
**516/516** untuk teks, alasan penolakan, dan kualitas. Seluruh hasil decoder
yang diterima juga cocok pada mask, peran, keluaran karakter, dan kotak sel.
Kesetaraan implementasi ini tidak mengubah hasil akurasi yang gagal di atas.

Startup fixture `rawenstwa` dibangun ulang dari RGB sumber asli, dengan crop,
input float, kualitas, dan keputusan TypeScript yang diperiksa untuk model
final. Bukti ini merupakan pengujian host; pengujian Android/APK dilaporkan
terpisah. Suite Python **182 tes lulus**.

Sebelum binding aset aplikasi baru, hash **1.974 file baseline** dan **3.612 PNG
komposisi** diperiksa tetap sama. Hash PNG dicatat ketika training sedang
berjalan dan diperiksa ulang setelah selesai; bukan diklaim sebagai pembekuan
seluruh piksel sebelum training. Checkpoint/evidence lama tetap dipertahankan.

Checkpoint baru, konfigurasi, lingkungan, dan snapshot sumber disimpan di
`training/checkpoints/braille-word-reader-id-v3/`. Semua epoch mentah, dataset,
lingkungan Python, dan oracle besar tetap berada di direktori luar repo:
`C:\bm-symbols-20260908`, `C:\bm-datasets`, serta `C:\bm-envs`.

Langkah berikutnya adalah memperoleh foto indikator kapital/angka Indonesia
yang berlabel dan memisahkan halaman fisik holdout sebelum pengembangan
lanjutan. Putaran training tambahan tidak dijalankan dalam update ini.
