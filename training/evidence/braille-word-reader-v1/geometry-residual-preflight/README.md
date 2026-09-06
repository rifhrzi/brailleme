# Pemeriksaan cabang geometri sebelum training

[Laporan utama](preflight.json) mengikat konfigurasi run21 dan seluruh laporan melalui SHA-256. [Indeks arsip](archive-index.json) mencakup sumber asli, protokol, dan pemeriksaan independen. Bukti ini tidak menetapkan model lulus pengenalan atau siap dibundel.

Cabang mengambil fitur pada enam posisi perkiraan titik dari kotak awal. Fitur tersebut mengoreksi keberadaan sel, kotak, dan enam probabilitas titik sekaligus. Input 480 × 192 dan output grid stride 4 dengan 11 kanal tetap sama; tambahan 2.451 parameter menghasilkan total 831.758 parameter.

- Bobot kepala lama digunakan sekali sebagai logit linear. Tidak ada rekonstruksi inverse sigmoid.
- Proyeksi koreksi terakhir dimulai dari nol. Semua keluaran awal dan 280 array bobot lama sama persis dengan parent v15; 52 BatchNorm tetap beku.
- Enam posisi perkiraan dihitung dari 9.983 sel unik DSBI TRAIN. Posisi ini bukan koordinat titik Angelina yang diketahui pasti.
- Sampling memakai koordinat pusat piksel, stride 4, origin 3, interpolasi bilinear, dan clamp batas. Audit independen juga memeriksa pemisahan batch.
- Konversi diuji pada enam input TRAIN/analitik, untuk cabang nol **dan** salinan diagnostik nonnol. Keduanya memakai operator TFLite bawaan dan lulus toleransi numerik.
- Satu backward CUDA batch 24 memeriksa ukuran memori dan gradien; tidak menyimpan hasil optimasi.

Run21 hanya boleh dimulai dari checkpoint nol `d6d43af7f662584d1f84c1ac7411d1572f898224746973405ed080c3ea1171d5`. Salinan diagnostik nonnol serta kedua probe TFLite tidak boleh dilatih lebih lanjut atau diaktifkan dalam aplikasi.

Sumber/crop/validation/ambang pemilihan tetap mengikuti protokol yang dibekukan. Checkpoint terlatih memerlukan evaluasi 405 validation, regresi 607 kata, 44 negatif, 66 blok lama, komposisi berulang, kasus penolakan, dan eksekusi interpreter Android. Bukti APK utama dan kertas fisik tetap terpisah.
