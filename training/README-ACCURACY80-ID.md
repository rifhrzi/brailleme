# Training kapital, angka, dan tanda baca - 10 September 2026

Percobaan ini menargetkan **80% teks utuh pada setiap kelompok**, termasuk
subkelompok dengan lebih dari satu karakter. Hasil yang ditolak tetap masuk
penyebut akurasi. Ketepatan hasil diterima minimal 95% dan penolakan seluruh
kasus wajib ditolak tetap diperiksa terpisah. Status akhir tercatat pada
laporan percobaan; keberadaan checkpoint tidak berarti target telah lulus.

**Training dan evaluasi selesai; target 80% belum tercapai. Kandidat tidak
dipromosikan karena target baru gagal dan hasil huruf lama menurun.** File
TFLite aktual pada TEST komposisi publik mencapai kapital awal **73%**, seluruh
kata kapital **69%**, angka **78%**, tanda baca **59%**, dan huruf kecil **71%**.
Rata-rata lima kelompok 350/500 (70%); ketepatan hasil positif yang diterima
350/400 (87,5%). Satu dari 40 rangkaian invalid masih diterima.
Lihat [laporan hasil](evidence/braille-word-reader-id-v4-20260910/RESULTS-ID.md)
untuk penyebut, baseline berpasangan, ketepatan hasil diterima dan regresi.
Bobot model aplikasi tetap dipertahankan.

## Cakupan dan sumber

Huruf kecil, kapital awal, seluruh kata kapital, angka beberapa digit termasuk
nol awal, dan `. , ? ! : ; - ( )`. Parser eksperimen menggunakan pola Indonesia
dalam [World Braille Usage, edisi ketiga, halaman 70](https://www.perkins.org/wp-content/uploads/2021/07/world-braille-usage-third-edition.pdf).
Tanda kurung harus berpasangan dan `?` mengikuti token karena pola selnya
ambigu tanpa konteks. Batasnya 12 karakter tercetak dan 24 sel, termasuk tanda.
Ini belum mencakup kontraksi, kalimat, desimal, matematika, atau dokumen penuh.
Corpus tanda baca saat ini memakai token huruf kecil. Kombinasi kapital/angka
dengan tanda baca, walaupun sebagian didukung parser, belum memiliki kelompok
evaluasi tersendiri.

Data tersimpan di `C:\bm-symbols-20260908\accuracy80-20260910`. Dataset asal
dan lingkungan Python yang sudah ada digunakan kembali. Dataset uji tambahan
berada di `C:\bm-datasets\DSBI-test-6937eb8`, bukan folder build Android.

| Bagian | Jumlah | Peran |
| --- | ---: | --- |
| Training baru | 6.096 | 1.200 tiap kelompok positif dan 96 rangkaian invalid |
| Validation | 540 | Pemilihan checkpoint; halaman sumber validation lama |
| TEST | 540 | Uji akhir dari 18 halaman DSBI TEST yang baru diakses |

Data baru berupa **komposisi potongan foto publik**; bukan foto alami satu
kata Indonesia, dan bukan pengujian kamera pada kertas fisik. Sebanyak 96%
contoh positif memiliki lebih dari satu karakter. Jumlah komposisi bukan
jumlah halaman atau pengambilan gambar independen. TEST terpisah berdasarkan
halaman sumber; keluarga buku dapat sama dengan keluarga pada TRAIN.

Revisi AngelinaDataset tetap `e13038d483bb58e17046d640e078dc4768276e9b`;
DSBI tetap `6937eb8973efe5ff337e5c018485cb626303f5c2`. Pemilihan 18 halaman
TEST dilakukan dengan seed `20260910`, bergiliran antar keluarga dari daftar
TEST penerbit sebelum inferensi. Byte sumber diperiksa terhadap Git blob pada
revisi tersebut, kemudian dicatat SHA-256-nya. Gambar DSBI berasal dari pemindai
datar. Kotak sel DSBI mempertahankan seluruh posisi 2x3 dan margin aslinya.

Semua manifest, hash 7.176 PNG, sumber donor, transformasi, versi dataset, dan
pemisahan halaman dicatat dalam `corpus-audit.json` serta
`corpus-verification.json`. Data TEST tidak boleh digunakan untuk gradien,
pemilihan epoch, threshold, atau pemilihan metode.

Hash manifest yang dibekukan:

| Manifest | SHA-256 |
| --- | --- |
| `train-symbols.json` | `7030f2525206baa943854f5c0bbc48dd82820ec7c0ea275e5f1c3a2c21c30dcf` |
| `validation-symbols.json` | `874c8ff878420cb511caba05e3c751a623a840def7a2c0014c802bd82790a701` |
| `test-symbols.json` | `3aace84bbcfd5337ec41fcaa8e3c5c98d2fb773fbefdea4702ff7400f8309310` |

## Metode

Percobaan lama memiliki ketidakcocokan variasi penanda kapital: donor training
alami, sedangkan donor validation berasal dari rotasi 180 derajat huruf A.
Percobaan baru memberikan transformasi identitas, horizontal, vertikal, dan
180 derajat dengan permutasi enam bit yang tepat pada TRAIN maupun evaluasi.
Sampling memilih transformasi dan halaman donor secara seimbang.

Training melanjutkan checkpoint MobileNetV2 dan pembaca enam titik yang
dipertahankan. Separuh setiap batch memakai data lama, separuh data baru dibagi
rata antar kelompok semantik, termasuk rangkaian invalid. Semua tahap memakai
batch 24 dan 160 langkah per epoch. Tahap kelanjutan ditentukan dari validation
dan diagnostik TRAIN; TEST tetap tertutup untuk pemilihan metode.

| Tahap | Bobot awal | Pengaturan | Status tahap |
| --- | --- | --- | --- |
| `run-heads` | Baseline lama | Backbone dibekukan; LR `0.0001`; batas 24 epoch | 10 epoch lengkap, epoch 10 dipilih; epoch 11 yang terputus tidak dipakai |
| `run-finetune` | Pilihan `run-heads` | Backbone dilatih; LR `0.0001`; bobot indikator/tanda baca 3; batas 36 epoch | 23 epoch lengkap, epoch 19 dipilih melalui validation |
| `run-dots` | Pilihan `run-finetune` | LR `0.00005`; bobot indikator/tanda baca 1,5; skala loss titik 3; sampling contoh TRAIN sulit | 48 epoch lengkap, epoch 45 dipilih melalui validation |

Total tiga tahap tersebut adalah **81 epoch lengkap**. Detektor terpilih adalah
`run-dots/best-word.keras`, sama dengan epoch 45, SHA-256
`090260a2a29f802cdcd07140344feaaf04d98bc48cd2be7a5d75e0482d6ee38c`.
Validation detektor ini memiliki rata-rata lima kelompok 65,4% dan kelompok
terendah 61%; angka tersebut **bukan hasil TEST** dan belum memenuhi target 80%.

Sebelum membuka TEST, ditambahkan satu pilot pembaca sel dengan batas maksimum
10 menit GPU. Detektor dibekukan; TRAIN pembaca tambahan memakai kotak sel
anotasi. Evaluasinya memakai potongan dari prediksi detektor, sehingga hasil
evaluasi tidak mengandalkan posisi sel anotasi.
Pilot selesai **17 epoch dalam 29,64 detik** dan tidak memenuhi kriteria
perbaikan yang ditetapkan. Pilihan pilot dengan ketepatan tinggi menghasilkan
rata-rata exact validation 57%, di bawah detektor tunggal 65,4%, sehingga
**refiner tidak dipakai**. Artefaknya tetap disimpan di `run-cell-refiner-pilot`;
cache TRAIN `cell-refiner-train-v1.npz` beserta JSON provenance dipertahankan.
Keputusan akhir detektor tunggal dibekukan sebelum TEST dan regresi dibuka.
Tidak ada kelanjutan training atau pemilihan ulang setelah evaluasi akhir.

Tahap ketiga berasal dari temuan bahwa penanda kapital sudah lebih mudah
dikenali, tetapi kesalahan enam bit dan keyakinan rendah masih menumpuk pada
satu kata. Pembobotan kesulitan diperoleh hanya dari 6.096 gambar TRAIN memakai
checkpoint awal tahap tersebut: 3.923 contoh berbobot 1, 640 berbobot 2, dan
1.533 berbobot 3. Bobot sampling dinormalisasi di dalam setiap kelompok; label,
gambar, dan threshold tidak diganti. Diagnostik dengan posisi sel anotasi
berfungsi untuk analisis training, bukan angka akurasi pemindaian.

Threshold aktif dibekukan: proposal/accept `0.25`, keyakinan titik `0.7`, beserta
seluruh batas geometri dalam konfigurasi run. Setiap epoch lengkap dan keputusan
pemilihan disimpan. Model aplikasi tidak otomatis diganti.

Windows Application Control menolak DLL optree yang diperlukan lingkungan
Keras lama. Jalur eksperimen memakai operasi PyTorch langsung yang berjalan
pada GPU, tanpa mengubah kebijakan keamanan. `word_torch_port.py` membaca
arsitektur serta bobot checkpoint yang sama dan menyimpan kembali arsip
`.keras`. Pemeriksaan port pada delapan fixture pengembangan yang sudah ada
mencatat selisih maksimum `5.2453e-6`; bukti ini bukan pengganti konversi dan
paritas checkpoint final pada seluruh validation. Optimizer Adam memakai
operasi tensor langsung dengan clip norm 10. Setiap tahap memulai state Adam
baru. Checkpoint menyimpan bobot, bukan state optimizer untuk melanjutkan secara
bit-identik di tengah putaran.

## Perintah

Jalankan dari akar repository. Perintah training berikut mendokumentasikan
konfigurasi tahap yang digunakan. Semua tahap dan evaluasi berikut sudah selesai.
Jangan menjalankannya ulang dengan corpus/run
yang sudah ada; file `source/`, `training-config.json`, `history.json`, serta
`selection.json` tiap run menjadi catatan reproduksinya.

```powershell
$trainingPython = 'C:\bm-envs\word-symbols-torch\Scripts\python.exe'
$experiment = 'C:\bm-symbols-20260908\accuracy80-20260910'

& $trainingPython training/scripts/word_symbol_v4_train.py `
  --experiment $experiment --freeze-backbone --run-name run-heads `
  --epochs 24 --steps 160 --batch-size 24 --learning-rate 0.0001 --patience 7

& $trainingPython training/scripts/word_symbol_v4_train.py `
  --experiment $experiment --checkpoint "$experiment/run-heads/best-word.keras" `
  --run-name run-finetune --epochs 36 --steps 160 --batch-size 24 `
  --learning-rate 0.0001 --patience 10 --minority-weight 3

& $trainingPython training/scripts/word_symbol_v4_mine.py `
  --experiment $experiment --checkpoint "$experiment/run-finetune/best-word.keras" `
  --output "$experiment/train-hard-example-weights.json"

& $trainingPython training/scripts/word_symbol_v4_train.py `
  --experiment $experiment --checkpoint "$experiment/run-finetune/best-word.keras" `
  --run-name run-dots --epochs 48 --steps 160 --batch-size 24 `
  --learning-rate 0.00005 --patience 12 --minority-weight 1.5 `
  --dot-loss-scale 3 --hard-example-weights "$experiment/train-hard-example-weights.json"
```

Training hanya mengevaluasi validation. Sesudah semua keputusan training
selesai, evaluasi TEST satu kali menggunakan snapshot sumber run terpilih:

```powershell
$chosenRun = "$experiment/run-dots"
& $trainingPython "$chosenRun/source/word_symbol_v4_train.py" `
  --experiment $experiment --evaluate-selected $chosenRun
```

Reservasi TEST mengikat satu checkpoint dan melarang tuning lebih lanjut pada
percobaan ini. Konversi TFLite memakai oracle validation PyTorch dan lingkungan
TensorFlow terpisah. Direktori export harus baru; konversi tidak membuka TEST.

```powershell
$exportPython = 'C:\bm-envs\word-symbols-tensorflow\Scripts\python.exe'
$exportDir = "$experiment/run-dots/export" # direktori bukti, bukan staging build

& $trainingPython training/scripts/word_symbol_v4_export.py --reference `
  --checkpoint "$chosenRun/best-word.keras" --validation "$experiment/validation-symbols.json" `
  --thresholds "$chosenRun/training-config.json" --output $exportDir

& $exportPython training/scripts/word_symbol_v4_export.py --convert `
  --checkpoint "$chosenRun/best-word.keras" --validation "$experiment/validation-symbols.json" `
  --output $exportDir

# Hanya setelah native TEST final selesai dan export-parity.json lulus:
& $exportPython training/scripts/word_symbol_v4_test_artifacts.py `
  --experiment $experiment --run $chosenRun --export $exportDir
```

Perbandingan TFLite memeriksa checkpoint terpilih yang sama dan baseline aktif
dengan parser/threshold yang sama. Ini adalah konfirmasi artefak setelah
pemilihan, bukan kesempatan memilih checkpoint baru dari TEST. Ketidaksamaan
keputusan dicatat sebagai kegagalan paritas. Bukti konversi, regresi lama, dan
validasi native harus sesuai file final sebelum promosi.

Evaluasi aktual menunjukkan paritas export 540/540 validation dan paritas
ringkasan native Torch/TFLite 540/540 TEST. Kandidat tetap tidak dipromosikan:
selain target TEST gagal, teks benar pada 607 crop publik turun 323 menjadi
309 dan pada 405 contoh lama turun 355 menjadi 318. Baseline dibandingkan ulang
dengan decoder v4 yang sama. **86 tes kode lulus**; pemeriksaan integritas
`model:verify` lulus tanpa mengubah kegagalan target akurasi lama.

## Batas penggunaan dan retensi

Sebanyak 98,98% komposisi TRAIN positif mencampur potongan dari halaman berbeda.
Tepi potongan dan pencahayaan yang berbeda dapat menjadi petunjuk lokasi sel
yang tidak muncul pada kertas asli. Komposisi ini juga belum mewakili seluruh
variasi perspektif, kertas melengkung, tetangga baris, dan kata yang terpotong
bingkai. Rincian berada di `PREPARATION-NOTES.md` dalam direktori percobaan.
Pengukuran offline tidak menggantikan validasi kamera fisik.

Tidak ada folder build Android baru yang diperlukan untuk training ini.
Seluruh indeks `training/checkpoints/retained-*.json`, checkpoint lama,
dataset, lingkungan training, APK, dan bukti lama tetap dilindungi.
Checkpoint riset dan file TFLite final disimpan pada
`training/checkpoints/braille-word-reader-id-v4`. Seluruh 81 epoch detektor dan
17 epoch pilot tetap dipertahankan. Gunakan
[indeks retensi tambahan](checkpoints/retained-models-20260910.json) bersama
seluruh indeks historis. [Inventaris detektor](evidence/braille-word-reader-id-v4-20260910/DETECTOR-RETENTION-INVENTORY.json)
menyediakan hash 81 checkpoint beserta laporan epochnya. Arsip bukti lama tidak
diubah. Tidak ada model aplikasi yang diganti, build/staging Android tambahan,
atau APK baru dalam eksperimen ini.
