# Perubahan pendekatan: uji model siap pakai tanpa training

**6 September 2026: kandidat ditolak. Fitur pengenalan dan APK kata belum selesai.**

Atas permintaan pengguna untuk mempercepat pekerjaan, training `run31` dihentikan
pada epoch 9. Delapan checkpoint lengkap tetap disimpan, tanpa seleksi atau
aktivasi. Prototipe arsitektur tambahan juga tidak dilanjutkan.

Satu model yang sudah dilatih, [DotNeuralNet YOLOv8m Braille](https://github.com/snoop2head/DotNeuralNet),
diuji tanpa training. Input pertama terlalu besar dibandingkan sel pada halaman
TRAIN. Satu koreksi skala memakai tinggi panduan 24 piksel, ditentukan dari
metadata TRAIN, kemudian ambang dipilih pada 405 fixture validation yang tetap.
Ambang tersebut dibekukan sebelum pemeriksaan regresi.

| Pemeriksaan | Kata benar | Ketepatan hasil diterima | Waktu setelah model dimuat |
|---|---:|---:|---:|
| Validation, input awal | 5/405 (1,23%) | 5/39 (12,82%) | 15,90 detik |
| Validation, koreksi skala | 224/405 (55,31%) | 224/231 (96,97%) | 13,97 detik |
| Regresi kata | 146/607 (24,05%) | 146/158 (92,41%) | 24,39 detik untuk seluruh 743 input regresi |

Seluruh 44 gambar negatif ditolak. Regresi huruf berulang benar 3/12; kasus yang
wajib ditolak lulus 13/14; blok eksperimen lama benar 11/66. Tidak ada pencarian
ambang pada 607 kata atau kelompok regresi lainnya. Waktu di atas berasal dari
GPU komputer, mencakup pemrosesan dan pelaporan setelah model dimuat; unduhan,
pemasangan dependensi, dan pemuatan model tidak termasuk. Ini bukan waktu Android.

Hasil tersebut gagal memenuhi 75% kata utuh dan 95% ketepatan hasil diterima,
serta masih menerima satu masukan yang wajib ditolak. Kandidat tidak
diintegrasikan atau dibangun menjadi APK. Mengganti classifier saja juga tidak
cukup: dengan deteksi dan pemeriksaan kualitas yang sama, 172/607 kata tetap
tertahan; classifier sempurna sekalipun dibatasi 435/607 (71,66%). Batas ini
tidak berlaku bagi detektor yang diubah atau digabung dengan detektor lain.

[Laporan ringkas](report.json) menyimpan angka menurut panjang kata, ambang,
identitas model, dan hash skrip/hasil privat. Berkas eksekusi dan deteksi tersimpan
di `.braille-training-work/word-scan-v1-20260905/dotneuralnet-pretrained-probe-v1/`.
Tidak ada perubahan pada model aktif aplikasi. Tidak ada proses training
lanjutan yang dijalankan setelah evaluasi ini.

Data publik mungkin bertumpang tindih dengan training model siap pakai.
Hasil ini hanya menunjukkan kelayakan pengembangan dan regresi, bukan pengujian
independen pada kertas timbul milik pengguna. Lisensi distribusi bobot dan
kesetaraan native juga belum diselesaikan untuk kandidat yang gagal ini.
