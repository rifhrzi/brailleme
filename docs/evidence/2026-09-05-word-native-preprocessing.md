# Bukti preprocessing foto kata — 5 September 2026

Jalur kata sekarang memakai **crop native → salin pixel → resize antialias TypeScript → tensor RGB float32**. Resize dilakukan pada runtime latar. Jalur classifier 28 × 28 lama tidak diubah.

| Pemeriksaan | Hasil |
| --- | --- |
| Rotasi 0/90/180/270°, mirror, dan crop Android sesuai koordinat TypeScript | 7/7 kasus; seluruh pixel cocok |
| Raw crop Android → resampler TypeScript → tensor, dibandingkan Pillow 12.3.0 | 7/7 kasus; seluruh float32 identik, galat maksimum 0 |
| Referensi unit terpisah dari ukuran 1 × 1 sampai 1296 × 520 | 7/7; SHA-256 byte RGB dan tensor float32 cocok |
| `Bitmap.createScaledBitmap(..., true)` dibandingkan Pillow bilinear | Tidak identik; pada crop panjang galat kanal maksimum 115/255 |
| Metadata Nitro Image 0.15.1 untuk `Bitmap.Config.ARGB_8888` | Menyebut BGRA; byte native terbukti RGBA |

Fixture warna RGB `[11,22,33]` menghasilkan byte native `[11,22,33,255]`. Membaca byte ini sebagai BGRA menukar merah dan biru. Jalur foto kata mengoreksi metadata tersebut khusus untuk Android dengan versi Nitro yang dipin; konversi BGRA generik tetap mengikuti kontraknya. Pemeriksaan dependensi harus diulang saat Nitro diperbarui.

Resampler baru menghitung bilinear antialias secara deterministik, dengan koefisien 22-bit dan pembulatan RGB 8-bit setelah setiap pass. Ini mempertahankan preprocessing Pillow yang dipakai training. Temuan bahwa resize native berbeda tetap tercatat; kesesuaian diperoleh dengan mengganti tahap tersebut pada jalur kata, bukan dengan menganggap kedua algoritma sama.

Koordinat crop integer dalam harness berasal dari fungsi TypeScript aplikasi. Native Android menjalankan operasi crop tersebut dan hasil pixelnya dibandingkan dengan referensi. Perhitungan alternatif langsung di ruang pixel dapat berbeda satu pixel pada batas yang hampir bulat karena floating point; pipeline dan harness menggunakan koordinat integer yang sama dari TypeScript.

Lingkungan native: emulator Android 16/API 36, `sdk_gphone64_x86_64`, little endian. Operasi `Bitmap` berjalan melalui DEX mandiri; kemudian implementasi TypeScript sebenarnya dijalankan pada byte crop native di Node. **Laporan ini sendiri tidak menguji Nitro/JS bridge, performa Hermes, foto kamera fisik, akurasi model, atau aksesibilitas.** Bukti Hermes dikumpulkan terpisah.

Laporan angka, hash sumber, dan hash tensor: [2026-09-05-word-native-preprocessing.json](2026-09-05-word-native-preprocessing.json). Input/output lengkap tersimpan di `C:\bm-word-native-parity-20260905\output`. Reproduksi: [harness native](../../scripts/word-native-parity/README.md).
