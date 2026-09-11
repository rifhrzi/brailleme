# Smoke test APK kata melalui kamera emulator

Protokol update 0.3.0: jalankan pada APK eksperimen internal yang hash dan pemeriksaan integritasnya tercatat. Dokumen ini adalah prosedur; bukti pelaksanaan dilaporkan terpisah. Arsip di bawah berasal dari 5 September dan tidak membuktikan tracker baru. Pemeriksaan bantuan CLI emulator lokal pada 5 September 2026 mengonfirmasi dukungan `-camera-back imagefile:<filename>`. Menjalankan bantuan ini tidak menyalakan emulator.

## Bukti diagnostik kamera yang sudah tersedia

[Harness native v3 terpisah](../../training/evidence/braille-word-reader-v1/native-camera-capture-diagnostic-v3/README.md) telah menjalankan VisionCamera, modul geometri lokal, dan helper TypeScript yang sama dengan sumber aplikasi. APK `com.brailleme.geometrydiagnostic` hanya memuat UI diagnostik dan tidak memuat model. Pernyataan ini berlaku untuk arsip diagnostik 5 September, bukan status update terbaru.

| Pengambilan aktual pada emulator | Hasil |
| --- | --- |
| Target foto 1920×1440; pratinjau/foto aktual 1280×960; empat rotasi dan viewport 1:1/2:1 | 6/6 kasus lulus empat penanda, galat maksimum 0,804 piksel pada toleransi tetap 4 piksel; 24/24 keanggotaan panduan sesuai |
| Target foto tambahan 1920×1080; foto aktual 1280×720, pratinjau 1280×960 | 2/2 kasus gagal, galat 119,907–120,601 piksel |
| JPEG mentah setelah orientasi EXIF dibandingkan bitmap Nitro | 8/8 identik untuk seluruh kanal RGB, galat maksimum 0 |

Foto 16:9 sebenarnya sama dengan baris 240–959 dari JPEG 4:3, sementara matriks sensor menyatakan crop mulai baris 120. Selisih sudah ada sebelum konversi Nitro; sumber publik emulator konsisten dengan crop satu sisi, tetapi kaitan revisi sumber ke binary terpasang belum dibuktikan. Kegagalan tetap dicatat tanpa koreksi heuristik. Dua warna tidak terlihat dalam percobaan awal yang juga disimpan; percobaan awal tersebut tidak dihitung sebagai keberhasilan empat penanda.

Foto PNG/JPEG, screenshot, metadata, sumber, dependency, log, dan hash tersedia dalam [indeks arsip](../../training/evidence/braille-word-reader-v1/native-camera-capture-diagnostic-v3/archive-index.json). [Verifier dan replay piksel offline](../../scripts/word-native-capture/README.md) memeriksa bukti tanpa build/emulator. Cakupan hanya kamera belakang tanpa mirror pada emulator x86_64. Ini tidak membuktikan pemetaan perangkat fisik, keberhasilan pengenalan, TTS, aksesibilitas, atau smoke test APK utama. Pengukuran penanda pada APK utama yang akhirnya diserahkan tetap wajib diulang.

## Prasyarat

- `model:verify:word`, config, lint, typecheck, tes, dan ekspor Android lulus pada sumber yang akan dibundel.
- APK update 0.3.0 berasal dari `C:\bm-symbols-apk-20260908`, memakai dependency sesuai lockfile, memuat model dan fixture sesuai hash manifest, serta lolos `apksigner verify`, `zipalign -c -P 16 -v 4`, dan pemeriksaan dua ABI: `arm64-v8a` serta `x86_64`. Jalankan [verifier APK](../../scripts/word-android-smoke/README.md#integritas-apk-sebelum-dipasang) dan simpan seluruh log. Periksa identitas ELF setiap pustaka geometry, VisionCamera, Nitro Image/Modules/TFLite, Worklets, dan TensorFlow Lite JNI. Pemeriksaan alignment ZIP tidak membuktikan alignment segmen ELF. Catat status signing internal secara eksplisit.
- Bekukan hash APK, model, manifest, bundle, fixture startup, source/crop gambar kamera, expected text, dan parameter penempatan gambar. Simpan hasil baru terpisah dari harness resizer lama.
- Hentikan Metro. Emulator menggunakan AVD uji tersendiri; instalasi pengguna pada perangkat fisik tidak disentuh.

## Input kamera

Siapkan satu gambar Braille dari fixture publik yang memiliki anotasi satu kata, serta satu gambar negatif/halaman kosong. Batas huruf dan expected text mengikuti anotasi. Letakkan kata pada kanvas netral dengan ruang cukup untuk bingkai kamera; simpan crop, skala, rotasi, dan hash kanvas. Gunakan penanda orientasi di luar bingkai jika perlu.

Backend `imagefile` menghasilkan frame kamera. Penyesuaian awal posisi/rotasi/skala dinilai dari pratinjau sebelum pengambilan; jangan mengubah label atau menghapus sel berdasarkan hasil model. Catat seluruh pengambilan, termasuk penolakan. Pengujian ini memeriksa integrasi kamera software, bukan pengaruh kertas timbul, cahaya, autofocus, atau kamera perangkat fisik.

Kanvas yang dibuat helper hanya memberikan penempatan nominal. Bidang pandang pratinjau dan foto dapat berbeda; jangan menggunakan rumus `cover` dari helper sebagai oracle pemetaan kamera. Empat penanda warna di luar konteks model harus terlihat pada pratinjau dan foto beku untuk membandingkan posisi secara independen. Build preview mencatat `BRAILLE_WORD_CAPTURE_GEOMETRY` berisi identitas pengambilan, ukuran pratinjau, empat sudut panduan/sensor, transformasi/crop/rotasi/mirror foto, ukuran tegak, dan ROI akhir. Cocokkan catatan pengambilan yang sama dengan penanda serta kotak panduan pada screenshot. Log tidak memuat piksel, teks hasil, atau lokasi file; build produksi tidak mengeluarkannya. Jika penanda atau metadata tidak cukup untuk membuktikan pemetaan, laporkan batas bukti tersebut.

Contoh menyalakan emulator setelah pekerjaan training mengizinkan penggunaan resource:

```powershell
$wordEmulator = Join-Path $env:LOCALAPPDATA 'Android\Sdk\emulator\emulator.exe'
$wordCameraImage = 'C:\bm-symbols-apk-20260908\evidence\camera-word.png'
$wordEmulatorArguments = @('-avd', 'BrailleMe_API_36', '-port', '5554', '-read-only', '-no-window', '-no-audio', '-no-snapshot', '-no-boot-anim', '-memory', '2560', '-cores', '2', '-gpu', 'swiftshader', '-feature', '-Vulkan', '-camera-back', "imagefile:$wordCameraImage", '-camera-front', 'none')
Start-Process -FilePath $wordEmulator -ArgumentList $wordEmulatorArguments -WindowStyle Hidden -RedirectStandardOutput C:\bm-symbols-apk-20260908\evidence\emulator-stdout.log -RedirectStandardError C:\bm-symbols-apk-20260908\evidence\emulator-stderr.log
```

Gunakan path gambar tanpa spasi untuk contoh di atas. Interaksi dilakukan melalui ADB; gunakan serial `emulator-5554` secara eksplisit pada setiap perintah. Emulator terpasang menaikkan permintaan awal 1536 MiB menjadi 2560 MiB saat diagnostik; contoh kini meminta 2560 MiB secara eksplisit. Catat RAM yang diminta dan yang benar-benar dipakai, lalu jadwalkan emulator setelah build/training melepaskan resource. Verifikasi `sys.boot_completed` sebelum instalasi; jangan memakai sleep panjang sebagai bukti kesiapan. Karena AVD dibuka `-read-only`, instal ulang APK dengan hash yang sama pada setiap sesi baru.

## Pemeriksaan pada APK utama

1. Instal APK yang sudah di-hash dan jalankan activity hasil `aapt dump badging`, tanpa Metro. Lewati onboarding melalui UI dan uji pemberian izin kamera. Simpan screenshot serta UI hierarchy saat tombol **Pindai** siap.
2. Keadaan siap hanya tercapai setelah loader menjalankan model native dan fixture PNG yang dibundel melalui preprocessing sebenarnya. Kaitkan observasi ini dengan hash model/fixture di APK dan alur `loadAndValidateWordModel`; jangan mengklaim log prediksi startup jika aplikasi tidak mengeluarkan log tersebut.
3. Tunggu bingkai otomatis mengikuti satu kelompok utuh beserta indikator dan status siap, lalu tekan **Pindai** sekali. Jangan mengubah label berdasarkan prediksi. Rekam status pemrosesan, foto beku, kotak setiap sel fisik, urutan indikator/huruf/digit, dan hasil yang tampil. Bandingkan hasil dengan anotasi, termasuk sel berulang bila fixture tersebut tersedia. Pemeriksaan tap ganda memastikan tidak ada dua pengambilan dalam satu sesi.
4. Tekan **Bacakan**. Uji **Pindai ulang**: hasil sebelumnya hilang, pembacaan lama dihentikan, dan Bacakan tidak aktif sebelum hasil baru berhasil. Bukti audio offline memerlukan TTS/voice offline yang benar-benar tersedia; keberhasilan event UI saja tidak membuktikan suara terdengar.
5. Saat pemrosesan foto, kirim aplikasi ke latar lalu buka lagi. Pastikan sesi baru kembali siap dan hasil sesi lama tidak muncul. Periksa logcat untuk crash, ANR, recycled Bitmap, atau akses interpreter yang sudah dilepas.
6. Matikan emulator uji, lalu jalankan kembali dengan gambar negatif sebagai `imagefile`. Startup fixture tetap harus lulus. Pada kertas kosong, status harus kembali Mencari Braille dengan Pindai nonaktif dan tanpa hasil/Bacakan. Jika tekstur negatif justru menghasilkan kandidat posisi, rekam kejadian tersebut; ketika Pindai dapat ditekan, hasil foto harus ditolak. Gunakan instalasi APK serta hash yang sama.
7. Aktifkan mode pesawat melalui UI Android atau perintah connectivity yang tersedia pada emulator tersebut. Lakukan cold start, pengambilan, dan percobaan TTS offline. Catat keberhasilan masing-masing secara terpisah.
8. Simpan screenshot/UI hierarchy/logcat, hasil pemeriksaan signing dan aset, kondisi jaringan, versi dependency native, serta daftar pengambilan dengan expected/actual/reason. Hentikan hanya emulator uji yang dimulai untuk pekerjaan ini.

Pengambilan kamera software dan fixture startup native melengkapi evidence resizer DEX/Hermes. Keduanya tidak menggantikan holdout fisik, pengujian TalkBack/font 200% pada perangkat, atau persyaratan produksi lainnya.

## Tambahan wajib untuk kapital, angka, dan bingkai otomatis

- Uji `abc`, `Abc`, `ABC`, semua digit, `2026`, `001`, indikator kosong, indikator ganda di tengah, campuran yang tidak didukung, serta batas 12 karakter/24 sel. Sumber tiap foto dan pola titik ditetapkan sebelum prediksi.
- Bingkai luar dan kotak tiap sel bergerak pada preview, tetapi tidak memicu shutter otomatis. Indikator memiliki kotak sendiri dan tidak menyumbang panjang teks. Hasil schema 4 mempertahankan case dan nol awal.
- Uji pergeseran kelompok melewati tengah/batas crop, beberapa kata/baris, indikator di ujung, target hilang, dan huruf berulang. Pindai harus nonaktif ketika pengamatan berumur >500 ms; bingkai hilang pada 800 ms.
- Rekam frame → sensor → preview → foto dengan penanda yang terlihat, termasuk rotasi dan mirror. Uji shutter ketika tracker sibuk, rotasi saat shutter, dan hasil yang terlambat setelah background atau pindai ulang. Kotak preview yang halus tidak boleh dipakai sebagai koordinat crop foto.
- Periksa pembukaan pertama, fokus sentuh, senter, kamera tanpa flash, penolakan kontrol senter, dan kembali dari background secara terpisah.
- Jalankan pemindai selama 10 menit. Catat FPS preview, latensi tracker, latensi tombol hingga hasil, dan PSS memori. Target preview ≥24 FPS dan p95 tombol hingga hasil ≤2 detik. `BRAILLE_WORD_TRACKING_METRICS` berisi penghitung callback/admission/drop dan histogram latensi; ini bukan pengukuran seluruh frame sensor. `BRAILLE_WORD_PHOTO_METRICS` mencatat durasi pengambilan aktual tanpa teks hasil.
- Uji font 200% dan TalkBack: status hanya diumumkan saat berubah, kotak tidak menerima fokus, target sentuh minimal 48 dp. Keberhasilan uji unit tidak menggantikan pengujian Android/TalkBack.
- Protokol gambar negatif menilai tidak adanya hasil salah. Jika tidak ada kelompok yang terdeteksi, Pindai boleh tetap nonaktif; jangan memaksa tombol aktif untuk membuat klaim pengujian foto.

Pisahkan status aturan, input sintetis/foto publik, Android emulator, dan kertas fisik. Kegagalan indikator model dicatat sebagai kegagalan keseluruhan. Tidak ada perubahan database atau publikasi Play Store pada update ini.
