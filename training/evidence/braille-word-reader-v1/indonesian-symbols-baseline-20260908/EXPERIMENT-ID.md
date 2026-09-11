# Model yang dipertahankan untuk APK eksperimen

**APK memakai checkpoint baseline dengan decoder baru `six-dot-id-v3`. Epoch
10 hasil training tidak dikirim.** Promosi kandidat dibatalkan karena menurunkan
hasil huruf dasar pada 607 crop publik dengan decoder dan threshold yang sama.
Kapital dan angka didukung oleh aturan penerjemahan, tetapi akurasi modelnya
belum memenuhi target. Update ini tetap eksperimen internal.

## Keputusan setelah satu putaran training

Satu putaran selesai tepat **10 epoch**, arsitektur sama, learning rate awal
**0.00002**, batch **24**, tanpa OOM atau training tambahan. Semua epoch
dipertahankan. Validation natural135/stress270 memilih epoch 10, yang kemudian
gagal pemeriksaan promosi pada regresi huruf publik.

| Evaluasi dengan decoder/threshold sama | Baseline | Kandidat epoch 10 |
|---|---:|---:|
| Teks utuh benar dari 607 | 323 (53.21%) | 306 (50.41%) |
| Hasil yang diterima | 351 | 332 |
| Ketepatan hasil diterima | 92.02% | 92.17% |

Sebanyak **42 contoh yang sebelumnya benar menjadi salah/ditolak**, sedangkan
25 contoh membaik; selisihnya **17 teks benar berkurang**. Ada 281 contoh benar
pada keduanya dan 259 salah/ditolak pada keduanya. Karena pengguna meminta
regresi huruf dasar tidak diturunkan, **baseline dipertahankan**. Tidak ada
pemilihan epoch alternatif atau penyesuaian threshold berdasarkan evaluasi ini.

[checkpoint-selection.json](checkpoint-selection.json) tetap mencatat sejarah
pemilihan training epoch 10 secara utuh. Keputusan APK yang berbeda dicatat
pada [deployment-selection.json](deployment-selection.json), dengan rincian
607 baris pada [external-baseline-comparison.json](external-baseline-comparison.json).
Checkpoint dan semua bukti epoch 10 tetap tersimpan untuk penelitian pada
[folder kandidat](../indonesian-symbols-20260908/EXPERIMENT-ID.md).

Corpus 607 sudah diamati pada pengembangan sebelumnya, sehingga **bukan
holdout independen**. Hasilnya digunakan sebagai veto promosi; tidak masuk
gradien, pemilihan checkpoint training, atau pemilihan threshold. Crop,
geometri, pola titik, dan expected text kedua model sama.

## Hasil model yang benar-benar digunakan APK

Data simbol berikut adalah **komposisi sintetis dari potongan foto publik**,
bukan foto kata Indonesia alami atau pengujian kamera kertas fisik.

| Kelompok | Teks benar/total | Akurasi teks utuh | Benar/hasil diterima | Ketepatan hasil diterima |
|---|---:|---:|---:|---:|
| Huruf kecil | 103/140 | 73.57% | 103/107 | 96.26% |
| Kapital tunggal | 2/140 | 1.43% | 2/99 | 2.02% |
| Seluruh kata kapital | 0/140 | 0% | 0/88 | 0% |
| Rangkaian angka | 40/64 | 62.50% | 40/43 | 93.02% |

Dari 32 rangkaian yang wajib ditolak, **7 masih diterima**. Tidak ada kelompok
baru yang memenuhi sekaligus target 75% teks utuh dan 95% ketepatan hasil
diterima. Melewatkan sebuah indikator dapat menghasilkan teks yang sah menurut
aturan tetapi tetap salah. Lihat [symbol-readiness.json](symbol-readiness.json).

Hasil lain yang diuji ulang pada file TFLite yang sama:

- Validation pemilihan lama: **355/405 teks benar**, 355/359 hasil diterima
  benar; natural129/135 dan stress226/270 sama dengan baseline sebelum training.
- Regresi publik: **323/607 benar**, **44/44 negatif ditolak**, 20/66 blok lama
  benar. Negatif 44 juga digunakan untuk training/regresi; blok 66 bukan
  benchmark kata utuh.
- Pengulangan sel: **11/12 benar**.
- Diagnostik penolakan: **12/14 benar**; dua `selection_cut` masih diterima.
- Dua foto publik yang sudah diketahui berlabel angka `66` dan `67`: **2/2
  benar**, dilaporkan terpisah dari komposisi sintetis dan bukan holdout baru.

## Data, konversi, dan kesetaraan

AngelinaDataset dipulihkan pada revisi
`e13038d483bb58e17046d640e078dc4768276e9b`; DSBI TRAIN pada revisi
`6937eb8973efe5ff337e5c018485cb626303f5c2`, dengan hash file asli yang cocok.
DSBI TEST tidak digunakan. Komposisi tambahan berjumlah 3.096 training dan
516 validation. Pool simbol berasal dari 296 halaman training dan 14 halaman
validation, dipisahkan sebelum crop/augmentasi. **516 komposisi bukan 516 foto
independen.** Titik 6 pada sumber Rusia/matematika tidak diberi makna kapital
Indonesia secara otomatis. Pola validation yang tidak tersedia dibuat dengan
refleksi/rotasi potongan validation dan pemetaan bit yang dicatat secara jelas.

Checkpoint APK tetap:

`training/checkpoints/braille-word-reader-v1/best-word.keras`

SHA-256 `cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7`.

Konversi benar-benar dijalankan ulang dan menghasilkan TFLite **3.259.716 byte**
yang identik dengan bobot baseline lama:

`406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230`

Bukti lama tidak dipakai ulang sebagai bukti decoder baru. Pada 405 input,
keputusan raw serta setelah pemeriksaan kualitas sama untuk Torch, TensorFlow,
dan TFLite. Selisih numerik maksimum pasangan masing-masing sekitar 2.72989e-5,
5.42998e-5, dan 4.55379e-5, di bawah batas 5e-4.

Replay aktual Python/TypeScript lulus **405/405** dan **516/516**, mencakup teks,
alasan penolakan, kualitas, serta mask/peran/keluaran/kotak sel untuk hasil
decoder yang diterima. Ini membuktikan kesetaraan implementasi, bukan akurasi
Braille fisik. Startup `rawenstwa` dibuat ulang dari RGB sumber asli dan
diperiksa kembali pada jalur crop, input float, kualitas, serta decoder TS.
Pengujian Android/APK dilaporkan terpisah.

Suite Python lengkap lulus **186 tes** pada lingkungan Torch dengan CPU,
termasuk empat tes keputusan veto promosi baru. Perintah, exit code, dan hash
log ada pada [python-tests-deployment.json](python-tests-deployment.json).

Seluruh **1.974 file baseline** dan **3.612 PNG komposisi** diperiksa tidak
berubah sebelum binding aset aplikasi. Checkpoint epoch 10, semua epoch mentah,
serta bukti kandidat dipertahankan. Hash PNG pertama dicatat saat training
berjalan lalu diperiksa ulang; bukan diklaim sebagai pembekuan piksel sebelum
training. Dataset, lingkungan, dan oracle besar tetap di `C:\bm-datasets`,
`C:\bm-envs`, serta `C:\bm-symbols-20260908`.

Sampel indikator Indonesia berlabel serta holdout halaman fisik diperlukan
sebelum klaim akurasi penggunaan nyata. Tidak ada training tambahan dalam
update ini dan tidak ada klaim siap produksi.
