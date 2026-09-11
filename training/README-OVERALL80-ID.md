# Lanjutan akurasi teks utuh: overall v5

Eksperimen ini melanjutkan checkpoint ID-v4 tanpa menyalin dataset latihan atau
membuat folder build Android. Hasil ID-v4 dan seluruh indeks retensi sebelumnya
tetap dipertahankan. Direktori kerja training:
`C:\bm-symbols-20260908\overall80-20260911`.

## Hasil akhir 11 September 2026

Model TFLite terpilih mencapai **438/500 teks utuh tepat (87,6%)** pada TEST
baru yang dibekukan setelah pemilihan metode dari validation. Target 80%
overall tercapai. Seluruh 40 contoh invalid ditolak; presisi pada 474 teks
positif yang diterima adalah 92,41%.

| Kelompok | Tepat / total | Akurasi |
| --- | ---: | ---: |
| Huruf kecil | 84/100 | 84% |
| Kapital awal | 87/100 | 87% |
| Seluruh kata kapital | 82/100 | 82% |
| Angka | 94/100 | 94% |
| Tanda baca | 91/100 | 91% |

Audit 200 potongan konteks asli menghasilkan 185/200 (92,5%), dilaporkan
terpisah dari penyebut utama. Torch dan TFLite menghasilkan keputusan teks,
urutan sel, dan jalur pembacaan yang identik pada seluruh 740 contoh tersebut.

Ada batasan yang masih perlu diperbaiki: pada benchmark historis 405 contoh,
hasil turun dari 355/405 (87,65%) menjadi 343/405 (84,69%). Pada benchmark
publik historis 607 contoh, hasil naik dari 323/607 (53,21%) menjadi 350/607
(57,66%), tetapi presisi hasil yang diterima turun dari 91,50% menjadi 88,61%
dengan decoder pembanding yang sama. Empat puluh empat contoh negatif
historis tetap seluruhnya ditolak. Teks sepanjang 12 karakter pada TEST utama
baru mencapai 10/24 (41,67%); target overall tidak berarti semua panjang teks
sama kuatnya.

Model tersimpan di [checkpoint ID-v5](checkpoints/braille-word-reader-id-v5/),
dengan [laporan pengujian](evidence/braille-word-reader-id-v5-20260911/REPORT.md)
dan [indeks retensi](checkpoints/retained-models-20260911.json). Integrasi APK
memakai direktori aset baru `assets/models/braille-word-reader-id-v5`,
dengan model lama tetap dipertahankan. Bukti 87,6% berasal dari benchmark scan
publik, belum dari kamera ponsel fisik. Folder staging Android tetap satu.

Implementasi aplikasi mempertahankan metode pembacaan yang sama. Paritas
TypeScript pada seluruh 540 contoh validation lulus untuk teks, alasan penolakan,
mask fisik, peran simbol, dan rute, dengan hash piksel keenam transformasi identik.
[Bukti integrasi aplikasi](evidence/braille-word-reader-id-v5-native-20260911/runtime-parity.json)
disimpan terpisah dari arsip hasil training. Pengujian sumber aplikasi mencakup
874 tes yang lulus, termasuk tanda baca dalam tampilan hasil, draft, Bacakan,
dan pengambilan deteksi terbaru saat tombol Pindai ditekan.

## Target dan batas pengukuran

Target utama adalah **minimal 400 teks tepat dari 500 contoh positif (80%)**.
Teks harus cocok seluruhnya, termasuk kapital, angka, dan tanda baca
`. , ? ! : ; - ( )`. Contoh positif yang ditolak tetap dihitung salah.
Ketepatan hanya pada hasil yang diterima, penolakan rangkaian invalid, dan
hasil setiap kelompok dilaporkan terpisah; semuanya tidak mengganti penyebut
500 atau mengubah target menjadi 80% setiap kelompok.

TEST utama berisi 100 contoh setiap kelompok: huruf kecil, kapital awal,
seluruh kata kapital, angka, dan tanda baca. Sebanyak 480/500 memuat lebih
dari satu karakter. Empat puluh contoh invalid diperiksa secara terpisah.
Komposisi dan jadwal panjang teks tetap menggunakan pembentuk contoh v4.
Kelompok tanda baca memakai token huruf kecil. Gabungan kapital, angka, dan
tanda baca dalam satu teks belum memiliki kelompok pengujian khusus.

Dua puluh empat halaman DSBI TEST baru dipilih sebelum anotasi atau inferensi,
terpisah dari halaman TRAIN dan 18 halaman TEST v4 yang telah diamati.
Kata-kata merupakan komposisi potongan scan publik; hasil ini tidak membuktikan
akurasi kamera ponsel atau kata Indonesia alami. Audit 200 potongan konteks
asli dilaporkan terpisah, karena distribusinya dominan teks pendek dan tidak
memiliki contoh kapital alami. Audit itu tidak menambah penyebut TEST utama.

## Metode

Fine-tuning memakai batch 24: 11 komposisi positif seimbang, satu komposisi
invalid, enam contoh replay dengan semua sel lengkap, lima potongan konteks
lama, dan satu contoh negatif. Setengah contoh di luar kelompok konteks
menggunakan gambar asli; sisanya memakai perubahan geometri ringan yang
mempertahankan setiap sel lengkap, disertai perubahan cahaya ringan.

Checkpoint lama dipakai sebagai awal training, dengan statistik BatchNorm
dibekukan. Training 24 epoch, masing-masing 160 langkah, selesai sekitar
17,5 menit pada RTX 5060 Laptop. Learning rate naik selama tiga epoch pertama
kemudian turun dengan jadwal cosine. Seluruh checkpoint disimpan dan dinilai
pada validation menggunakan kebijakan pembacaan yang sama. Epoch 14 terpilih;
checkpoint rerata tiga epoch juga diuji pada validation dan tetap dipertahankan.

Lima kebijakan pembacaan diperiksa pada validation yang sama: pembacaan asli,
dua variasi pembacaan ulang dari orientasi berbeda, pembacaan ulang dengan
zoom, serta gabungan keduanya. Metode terpilih `cascade_zoom_fallback`
mencapai 407/500 (81,4%) pada validation sebelum TEST dibuka. Bila pembacaan
asli ditolak, tiga orientasi tambahan diperiksa dengan penomoran titik yang
dikembalikan ke posisi asli. Bila masih ditolak, zoom 1,25 dan 1,5 diperiksa.
Penggabungan hanya mengubah probabilitas titik pada sel yang cocok dan tetap
lengkap; koordinat sel, filter kualitas, dan ambang confidence dipertahankan.
Pemilihan hasil saat inferensi tidak membaca label jawaban.

Pembacaan tambahan hanya berjalan saat diperlukan. Pada 540 contoh TEST
utama, rata-rata diperlukan 1,865 lintasan model per gambar, dengan 14,28
detik total inferensi batch dan pemilihan hasil di CPU pengujian. Ini bukan
pengukuran waktu kamera Android. Model tetap 792.219 parameter, dengan input
`[N,192,480,3]` dan output `[N,48,120,11]`; TFLite berukuran 3.239.392 byte.

Ekspor memeriksa 540 gambar validation dalam enam tampilan (3.240 inferensi).
Seluruh 2.700 keputusan dari lima kebijakan pembacaan sama dengan Torch.
Pemeriksaan numerik awal `atol=1e-4, rtol=1e-4` gagal karena selisih maksimum
0,00025159. Laporan gagal itu tetap disimpan. Sebelum TEST dibuka, pemeriksaan
terpisah menerima toleransi absolut 0,0003 (`rtol=0`) berdasarkan seluruh
205.286.400 nilai yang telah diukur dan kesamaan seluruh keputusan. Ini bukti
kesetaraan empiris pada validation, bukan jaminan universal seluruh input.

## Reproduksi dan bukti

- `word_overall_v5_protocol.py` dan `word_overall_v5_prepare.py`: protokol,
  reservasi sumber, komposisi TEST, dan audit konteks asli.
- `word_overall_v5_train.py`: training dari input TRAIN/validation yang terikat
  path dan SHA-256 protokol. Menolak training setelah metode akhir dibekukan.
- `word_overall_v5_tta.py`: pembacaan NumPy yang sama untuk tensor Torch dan
  TFLite, serta pemilihan kebijakan dari validation.
- `word_overall_v5_zoom.py` dan `word_overall_v5_runtime.py`: pembacaan ulang
  dengan zoom dan pemilihan implementasi inferensi yang dibekukan.
- `word_overall_v5_export.py`: konversi builtin TFLite dan pemeriksaan numerik
  serta keputusan teks seluruh 540 contoh validation dalam enam tampilan.
- `word_overall_v5_evaluate.py`: pembekuan metode dan evaluasi TEST akhir.

Hasil akhir harus merujuk checkpoint, kebijakan pembacaan, decoder, threshold,
manifest, serta hash gambar yang sama. Reservasi evaluasi TEST ditulis sebelum
inferensi. Hasil validation tidak boleh disebut sebagai akurasi TEST akhir.
Kelulusan target riset ini tidak otomatis mengubah status rilis aplikasi.
