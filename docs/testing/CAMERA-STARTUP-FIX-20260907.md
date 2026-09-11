# Perbaikan kamera APK internal — 7 September 2026

APK perbaikan lulus uji kamera emulator Android API 36. Perangkat fisik pada
screenshot pengguna tidak tersedia; penyebab spesifik pada HP tersebut belum
dikonfirmasi melalui log perangkat.

## Temuan dan perubahan

APK lama tanggal 6 September berhasil direproduksi menampilkan "Kamera belum
dapat digunakan" ketika kamera pertama dibuka. Log memperlihatkan sesi mencapai
OPENING, lalu aplikasi melepas koneksi kamera. Pindai ulang berikutnya berhasil.
Handler lama membuang detail error sehingga pemicu asli tidak tercatat.

Kode VisionCamera 5.2.2 mengirim perintah `torchMode` saat controller tersedia,
tanpa menunggu preview. Kegagalan perintah itu masuk ke `onError` yang sama dengan
kegagalan sesi. Aplikasi sebelumnya mengirim `off` pada kamera yang melaporkan
dukungan senter, lalu menutup seluruh kamera jika perintah gagal.

Perbaikan di `src/features/live/word-photo-camera.tsx`:

- Menunggu sesi dan preview berjalan sebelum mengubah senter. Tidak mengirim
  perintah `off` saat pembukaan awal; kegagalan senter menonaktifkan kontrol
  senter dan mempertahankan preview serta pengambilan foto.
- Memberi batas 15 detik untuk kamera yang terdeteksi tetapi tidak mulai
  menampilkan preview, sehingga pengguna mendapat kesempatan mencoba ulang.
- Mencatat error kontrol/sesi sebagai `BRAILLE_WORD_CAMERA_ERROR` tanpa objek
  foto, piksel, atau hasil pembacaan.
- Mengaitkan error saat shutter dengan ID pengambilan aktif, agar layar tidak
  mengabaikannya. Mengabaikan error penutupan kamera yang terlambat setelah foto
  tersedia atau komponen ditinggalkan.
- Menghilangkan bingkai yang menimpa tulisan error. Layar membedakan kegagalan
  membuka kamera dari kegagalan memproses foto.

Jalur kegagalan senter juga dijelaskan dalam
[laporan upstream VisionCamera](https://github.com/mrousavy/react-native-vision-camera/issues/4069).
Ini mendukung perbaikan kontrol kamera; bukan bukti pemicu pada HP pengguna.

## Pemeriksaan

- Model, konfigurasi rilis, lint, typecheck, dan 34 suite / **491 tes** lulus.
  Coverage statement 90,56%. Delapan tes tambahan meliputi startup, senter gagal,
  timeout, rescan, error terlambat, dan error saat shutter.
- Build release lokal dengan `APP_ENV=preview`, JDK 17, ARM64 dan x86_64 lulus.
- Tanda tangan v2 dan zip alignment lulus. Sertifikat debug sama dengan APK lama.
- Android minimum API 29. Tidak memiliki izin INTERNET atau penyimpanan umum.
- Hash APK terpasang identik dengan berkas penyerahan. Hash model kata terbundel
  cocok dengan source: `406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230`.
- **3/3 pembukaan baru** setelah force-stop mencapai preview dan tombol foto aktif.
- Pengambilan foto menghasilkan gambar 1856×1392, diputar menjadi 1392×1856,
  diproses, lalu pola warna non-Braille emulator ditolak. Foto tetap terlihat.
- Pindai ulang dan kembali dari latar belakang mengaktifkan preview kembali.
- Senter dapat dinyalakan/dimatikan dengan preview tetap aktif.
- Log crash kosong. Bukti CameraService menunjukkan klien `com.brailleme.app`.

Percobaan harness yang berjalan sebelum pemasangan selesai dan percobaan resume
tanpa menunggu transisi Home tidak dihitung. Pemeriksaan valid memakai berkas
`verified-start-*`, `fixed-rescan`, dan `fixed-resume-ready`.

## Berkas dan batas verifikasi

[APK perbaikan](../../output/android/BrailleMe-0.2.0-camera-fix-2026-09-07.apk)
(133.135.553 byte), SHA-256:
`7a1a2e551d9527845dec32c888ddb90b08068589d7fb109493937658df71f265`.

[Laporan integritas APK](../../output/camera-fix-20260907/artifact.json),
[hasil runtime](../../output/camera-fix-20260907/runtime-checks.json),
[preview berhasil](../../output/camera-fix-20260907/verified-start-1-camera.png),
[foto selesai diproses](../../output/camera-fix-20260907/fixed-capture.png), dan
[error APK lama](../../output/camera-fix-20260907/baseline-error.png).
Berkas output diabaikan Git; staging lokal: `C:\bm-camera-20260907`.

Ini APK eksperimen internal dengan signing debug, bukan rilis produksi.
Keberhasilan kamera emulator tidak membuktikan perbaikan pada seluruh HP atau
akurasi pembacaan kertas Braille fisik. Pemilik HP pada screenshot masih perlu
menguji APK ini, terutama pembukaan pertama, foto, pindai ulang, serta kembali
setelah aplikasi masuk latar belakang.
