# Kandidat terbaik untuk APK percobaan internal

Pada 6 September 2026 pengguna meminta: **"pakai model yang akurasinya paling tinggi"**.
Ini menggantikan syarat menunggu model lulus sebelum tersedia untuk uji internal.
Target awal 75% kata utuh / 95% ketepatan hasil diterima tetap dicatat dan belum
lulus; tidak ada klaim bahwa APK ini memenuhi persyaratan produksi.

Bobot `run20/epoch09`, dikenal sebagai v15/v26, adalah bobot terlatih terbaik
dari perbandingan publik yang telah dijalankan. Bobot yang sama mencapai
325/607 (53,54%) dengan decoder historis. Aplikasi menggunakan decoder v2
yang mempertahankan huruf bersebelahan sebelum NMS berdasarkan posisi;
angka yang berlaku untuk aplikasi adalah **323/607 (53,21%)**, dengan
**323/351 (92,02%)** hasil diterima benar. Penolakan dihitung sebagai kegagalan
kata utuh. Model classifier dengan kotak acuan ideal bukan pembanding scanner.

- SHA-256 TFLite: `406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230`.
- Ukuran: 3.259.716 byte; input RGB float32 `[1,192,480,3]`, output `[1,48,120,11]`.
- Ambang dari validation: proposal 0,25; acceptance 0,25; keyakinan titik 0,70.
- Regresi negatif: 0/44 hasil diterima. Huruf berulang: 11/12 benar.
- Penolakan wajib: 12/14; dua kasus kata terpotong masih keliru diterima.
- Paritas desktop 405 input lulus; ini tidak menggantikan pengujian perangkat fisik.

Manifest memakai status `experimental_candidate`, izin eksplisit uji lokal,
dan `internalAcceptance.passed: false` beserta angka aktual. Build produksi
dan cloud tetap ditolak. Verifikasi aset eksperimen memeriksa integritas dan
kesesuaian laporan tanpa mengubah kegagalan akurasi menjadi kelulusan.

Model dan gambar pemeriksaan startup dibundel agar aplikasi dapat berjalan
offline. Foto startup `RAWENSTWA` berasal dari crop RGB asli, dan dipilih ulang
dengan crop, resize, serta decoder TypeScript saat ini. Keberhasilan startup
memeriksa pemuatan pipeline, bukan ketepatan semua kata.

Laporan terkait disalin tanpa mengubah angka atau hash asal ke direktori ini.
Hasil publik merupakan bukti pengembangan yang telah diamati, bukan holdout
fisik independen. Gunakan [panduan uji mandiri](../../../../docs/testing/WORD-SCAN-SELF-TEST-ID.md)
untuk mencatat keberhasilan, kesalahan, dan penolakan pada kertas timbul.
