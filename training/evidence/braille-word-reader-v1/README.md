# Bukti model pembaca satu kata

**Status terbaru, 6 September 2026:** pengguna meminta memakai bobot dengan akurasi
tertinggi yang tersedia. Bobot v15/v26 dihubungkan sebagai kandidat eksperimen
internal dengan decoder v2: 323/607 kata utuh (53,21%), 323/351 hasil diterima
benar (92,02%). Target awal tetap belum lulus. Lihat
[catatan kandidat yang dipakai](experimental-internal-best/README.md).
Riwayat di bawah tidak menyatakan training masih berjalan.

Model ini membaca seluruh sel di konteks foto sekaligus. MobileNetV2 `alpha=0.5`
dengan bobot awal ImageNet menggabungkan fitur stride 16, 8, dan 4. Input publik
adalah RGB `float32` `[1,192,480,3]` dalam rentang 0–1; normalisasi MobileNet
ke −1–1 berada di dalam model. Output `[1,48,120,11]` berisi keberadaan sel,
offset pusat x/y, lebar/tinggi relatif terhadap konteks, lalu probabilitas titik
1–6. Bit paling rendah adalah titik 1. Model memiliki 215.739 parameter.

Area pilihan pengguna mempunyai rasio 1–8; konteks model tetap 2,5:1. Decoder
mengabaikan kotak yang sepenuhnya berada di luar pilihan, menolak kotak yang
terpotong atau meragukan di dalam pilihan, lalu menyusun pola A–Z dari kiri ke
kanan. Huruf berulang tidak dihapus. Kamus tidak digunakan untuk melengkapi sel.

Kandidat pertama belum memenuhi syarat: **168/607 kata benar utuh (27,68%)**
dan **168/213 hasil diterima benar (78,87%)**, walaupun semua 44 gambar negatif
ditolak. Laporan yang gagal disimpan utuh di [failed-selected-v1](failed-selected-v1/).
Model ini tidak boleh dipromosikan menjadi model aktif. Pengembangan berikutnya
tetap memisahkan pemilihan checkpoint/ambang pada validation dari regresi
eksternal yang kini sudah pernah diamati.

Kandidat normalisasi berikutnya (`selected-v6`) juga gagal: 168/607 kata
benar utuh, 168/200 hasil diterima benar (84%), dan satu dari 44 negatif
keliru diterima. Konversi 405 input telah lulus, tetapi kesetaraan numerik
tidak memperbaiki kegagalan pengenalan tersebut. Model ini juga tetap
terpisah dari model aktif.

Kandidat `selected-v7` memperbaiki hasil menjadi **222/607 kata benar utuh
(36,57%)** dan **222/265 hasil diterima benar (83,77%)**, dengan semua 44
negatif ditolak. Hasil ini masih gagal kedua syarat pengenalan. Laporan
dibekukan di [failed-selected-v7](failed-selected-v7/); model tidak diaktifkan.
Konversi seluruh 405 input lulus dengan galat maksimum Torch–TensorFlow
0,00002343 dan TensorFlow–TFLite 0,00004939, serta teks/penolakan identik.

## Data dan batas klaim

- AngelinaDataset dikunci pada commit
  `e13038d483bb58e17046d640e078dc4768276e9b`. Seluruh pola yang ada dalam anotasi
  dapat menjadi target enam titik, termasuk pola di luar A–Z.
- Sumber `books/ola` dan `handwritten/ang_redmi` dipisahkan sebagai validation.
  `uploaded/test.txt` hanya digunakan sesudah checkpoint dan ambang dibekukan.
  Pemeriksaan hash melarang duplikat gambar melintasi train/validation/test.
- Training awal memakai 6.327 patch dengan 56 pola enam titik. Dua konteks
  tambahan pada `train-coverage-v2.json` menghasilkan 6.329 patch dan seluruh
  57 pola sumber. Audit berdasarkan identitas gambar dan baris CSV menunjukkan
  42.986 kemunculan sel lengkap hanya mewakili **30.908 dari 69.317 sel sumber
  unik (44,59%)**; kemunculan berulang bukan sel sumber baru.
- Data `complete-source-training-v3` mempertahankan seluruh 6.329 record tadi
  dan menambahkan 15.024 konteks piksel asli. Kini ada **21.353 patch unik**,
  106.459 kemunculan lengkap, dan **69.317/69.317 sel sumber unik** terwakili.
  Semua 240.738 anotasi dalam konteks cocok kembali ke baris CSV sumber;
  pemeriksaan 64 gambar yang tersebar di delapan keluarga sumber dan tiga
  skala cocok persis dengan crop/resize piksel asli. Tidak ada data validation
  atau eksternal yang ditambahkan ke training. [Audit cakupan](training-source-coverage.json)
  membedakan kemunculan, sel unik, dan batas klaim.
- Cakupan tersebut berlaku sebelum augmentasi. Sel yang terpotong setelah
  transformasi tetap diabaikan dalam loss keberadaan sel; ketersediaan semua
  anotasi tidak berarti setiap sel selalu menghasilkan pembaruan bobot pada
  setiap epoch. Dataset awal juga memiliki 65 patch dengan tepat 12 sel utuh
  dan 756 dengan sedikitnya 12 sel.
- Fixture kata memakai batas ruang pada anotasi, pemeriksaan tetangga dari
  seluruh halaman, batas tepi foto, dan pengecualian sambungan kata setelah
  tanda hubung akhir baris. Pola yang tidak didukung tetap termasuk kelompok
  sebelum seluruh kelompok dikecualikan. Tidak ada pemilihan berdasarkan hasil
  model atau kejernihan foto.
- Manifest akhir memuat 135 kata validation dan 607 kata eksternal. Fixture
  eksternal dikunci dengan SHA-256
  `e69d7c0d9582aef7d51a2c15706f6bf62bc3cdfe35d22ad28ad007301d56b41b`.
  [Audit batas](boundary-review.json) menyatakan cakupan pemeriksaan otomatis
  dan sampel visual secara terpisah, termasuk kandidat manifest yang ditolak.
- 44 gambar bukan Braille adalah regresi publik yang juga digunakan dalam
  training. Mereka bukan holdout negatif independen. 66 blok eksperimen lama
  dilaporkan terpisah karena sebagian merupakan potongan, bukan kata utuh.
- Validasi tambahan memakai 135 gambar natural dan 270 variasi kamera dari
  sumber validation yang sama. Seluruh gambar, kotak, dan area pilihan
  ditransformasikan bersama. Manifest dibekukan sebelum inferensi dengan hash
  `e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625`.
  Pemilihan mensyaratkan ketepatan hasil diterima ≥95% pada **masing-masing**
  kelompok natural dan variasi kamera, lalu memaksimalkan rata-rata keberhasilan
  kata utuh kedua kelompok. Penolakan kualitas gambar tetap dihitung gagal.

## Reproduksi

Jalankan dari akar repositori. Python training membutuhkan Keras 3 dan Torch
dengan CUDA; ekspor dijalankan dalam proses terpisah dengan TensorFlow.
Direktori kerja dan model percobaan tetap berada di `.braille-training-work/`.

Contoh kelanjutan dari data dan checkpoint percobaan yang telah diaudit:

```powershell
$env:KERAS_BACKEND='torch'
python training/scripts/word_train.py --data .braille-training-work/word-scan-v1-20260905/data --train-manifest ../complete-source-training-v3/combined-training-manifest.json --output .braille-training-work/word-scan-v1-20260905/new-run --resume .braille-training-work/word-scan-v1-20260905/selected-v7/best-word.keras --epochs 4 --batch-size 24 --learning-rate 0.00005 --illumination-augmentation --negative-repeat 27 --retain-epochs --geometry-loss relative_size_huber
$candidates = @('.braille-training-work/word-scan-v1-20260905/selected-v7/best-word.keras') + @(Get-ChildItem .braille-training-work/word-scan-v1-20260905/new-run/retained-epochs -Filter '*.keras' | Sort-Object Name | ForEach-Object FullName)
python training/scripts/word_select.py --data .braille-training-work/word-scan-v1-20260905/data --output .braille-training-work/word-scan-v1-20260905/new-selection --validation-manifest validation-stress-selection.json --grouped-selection --candidates $candidates
$env:KERAS_BACKEND='tensorflow'
python training/scripts/word_export.py --data .braille-training-work/word-scan-v1-20260905/data --run .braille-training-work/word-scan-v1-20260905/new-selection
python training/scripts/word_benchmark.py --data .braille-training-work/word-scan-v1-20260905/data --run .braille-training-work/word-scan-v1-20260905/new-selection
python -m unittest discover -s training/tests -p 'test_word_*.py' -v
```

Gunakan executable dari lingkungan TensorFlow pada dua perintah terakhir yang
memerlukannya; mengganti `KERAS_BACKEND` tidak memasang dependensi. Data awal
dapat dibuat melalui `word_data.py --per-page 28` pada direktori **baru**.
Jangan menimpa fixture yang sudah dibekukan. Konteks tambahan dibuat oleh
`word_cover_training.py` dari daftar sel belum tercakup yang diaudit, lalu
diverifikasi kembali sebelum dipakai. Perintah contoh menghasilkan kandidat
baru; reproduksi suatu laporan harus memakai snapshot sumber dan hash data
yang tercatat pada run tersebut.

`training-config.json` mencatat backend, GPU, seed, checkpoint asal, dan hash
skrip/data/bobot ImageNet. Run 1 dan berikutnya menyimpan salinan skrip sumber;
warmup awal hanya mencatat hash. Sumber warmup hanya dianggap tersedia bila
byte yang tersimpan kemudian cocok tepat dengan hash aslinya. Memulai
training baru menghasilkan kandidat baru yang wajib dievaluasi ulang. Hasilnya
tidak dianggap identik hanya karena seed sama.

Opsi `--camera-augmentation` memperluas rentang skala horizontal, kontras, dan
blur pada data training. `--retain-epochs` menyimpan setiap checkpoint agar
perbandingan pada validation dapat direproduksi. Zoom keluar tidak dapat
mengubah sel yang semula terpotong menjadi target utuh.

Manifest tambahan `train-coverage-v2.json` menambahkan dua konteks asli dari
sumber training untuk mask 16. Seluruh 57 pola sumber kini terwakili sebagai
sel lengkap; berkas training awal dan fixture evaluasi tetap dipertahankan.
Opsi `--half-turn-augmentation` memutar piksel asli 180 derajat sambil
menukar posisi titik 1/6, 2/5, dan 3/4. Identitas anotasi sisi depan tetap
sama. Kelengkapan sel diperiksa setelah setiap rasterisasi affine/rotasi:
kotak yang masuk kembali ke bingkai tidak dapat memulihkan titik yang sudah
terpotong dari piksel. [Audit visibilitas](augmentation-visibility-audit.json)
menyimpan regresi dan perbandingan 262 patch yang tetap.

Walaupun pemetaan setengah putaran valid secara geometris, percobaan run10
menurunkan pembacaan titik: checkpoint terbaik hanya 61/135 kata natural
benar pada pemeriksaan float32 penuh. Tidak ada evaluasi eksternal untuk
kandidat tersebut karena batas atas skor dua kelompok sudah kalah dari
baseline. Run11 mematikan putaran, mempertahankan profil pencahayaan dan
perbaikan visibilitas, lalu menjadi `selected-v7`. Run12 memakai profil yang
sama pada cakupan sumber v3; BatchNorm dan arsitektur tidak diubah. Hasil
run12 tidak mengalahkan baseline pada 405 validation: skor rata-rata dua
kelompok terbaik 83,15%, dibanding 83,33% milik `selected-v7`. Seleksi v8
mempertahankan checkpoint v7 yang sama; ekspor dan regresi eksternal tidak
diulang untuk model identik tersebut.

Cabang berikutnya dibuat melalui `word_batchnorm.py`: semua bobot awal
identik dengan v7, sementara 40 lapisan BatchNorm boleh memperbarui parameter
affine dan statistik bergerak pada training, dengan momentum 0,99. Backbone
mewarisi mode training model luar; prediksi dan ekspor tetap memakai mode
inferensi. Pemeriksaan 16 input TRAIN menunjukkan keluaran awal dan keluaran
setelah simpan/muat identik persis; pembaruan statistik diagnostik dibatalkan
sebelum checkpoint awal disimpan. Perubahan ini merupakan percobaan terkontrol,
bukan bukti bahwa statistik BatchNorm menyebabkan kegagalan kamera.
Run13 memakai cabang ini selama enam epoch pada data v3, dengan learning rate
0,00005, batch 24, profil pencahayaan yang sama, dan semua checkpoint disimpan.
Audit berkas epoch pertama membuktikan bahwa parameter affine, statistik
bergerak, konvolusi, dan head berubah; tidak ada bobot NaN atau Infinity.

Varian opsional `--image-normalization channel_gray_std05` menambah normalisasi
di dalam model: hitung rata-rata dan simpangan baku populasi per kanal per
gambar, batasi simpangan baku minimum 0,001, rata-ratakan kanal yang telah
distandardisasi, petakan ke `0,5 + 0,05 × gray`, batasi 0–1, lalu ulangi menjadi
tiga kanal. Input aplikasi tetap RGB dibagi 255. Varian ini memerlukan bukti
validasi dan konversinya sendiri sebelum dapat dipromosikan.

Implementasi normalisasi mengurangi satu piksel referensi per kanal sebelum
menghitung rata-rata lebar lalu tinggi. Varians juga memakai dua reduksi
berurutan. Rumus matematis tetap sama; urutan ini menghindari galat akumulasi
float32 ketika puluhan ribu piksel hampir putih dijumlahkan sekaligus.
Oracle selalu dibekukan ulang setelah sumber implementasi berubah.

Opsi `--geometry-loss relative_size_huber` memakai selisih log lebar/tinggi
agar galat ukuran relatif mempunyai bobot konsisten pada berbagai skala.
Anotasi sel dan ambang pemotongan tidak diperkecil untuk menyesuaikan hasil.
Lingkungan Torch tanpa TensorFlow dapat dipakai untuk training dan seleksi;
ekspor harus dijalankan dalam proses terpisah yang menyediakan TensorFlow.

Ekspor membekukan variabel TensorFlow sebelum konversi TFLite. Oracle Torch
mematikan TF32; pemeriksaan seluruh input validation mewajibkan galat maksimum
5×10⁻⁴ serta hasil teks dan penolakan yang identik. Bukti ini terpisah dari
paritas crop/resize native dan pengujian kamera fisik.

## Kriteria penerimaan

Kata harus benar utuh pada sedikitnya 75% seluruh fixture; penolakan dihitung
gagal. Sedikitnya 95% hasil yang diterima harus benar, dengan nol hasil kata
pada 44 regresi negatif. Laporkan hasil menurut panjang kata. Lulus angka
publik, konversi model, atau emulator tidak membuktikan akurasi kamera pada
kertas timbul dan tidak merupakan persetujuan rilis produksi.

Arsitektur mengikuti [kontrak MobileNetV2 Keras](https://keras.io/api/applications/mobilenet/)
dan implementasi backend portabel memakai [Keras 3](https://keras.io/keras_3/).
