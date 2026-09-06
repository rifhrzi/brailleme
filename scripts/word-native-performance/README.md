# Harness Hermes terpisah

Ini sumber entry yang benar-benar diuji dalam `training/evidence/braille-word-reader-v1/native-worklet-performance.json`. Template disalin tanpa perubahan dari staging yang diuji. Ekstensi `.template` menjaga entry diagnostik tetap di luar build aplikasi. APK diagnostik tidak boleh dibagikan sebagai APK BrailleMe baru.

Harness menjalankan decode PNG Nitro, crop 1296 × 520, pembacaan raw pixel, dan `resizeWordRawPixels` di runtime Hermes latar. Setiap byte RGB dibandingkan dengan Pillow, sementara heartbeat di JS mengukur apakah pemrosesan memblokir seluruh interval. Empat pengulangan ini tidak menguji kamera, inferensi model, aksesibilitas, atau performa perangkat fisik. Nilai `maxAbsError` membandingkan float32 dengan `byte / 255` presisi double; zero `mismatchByteCount` berarti setiap kanal kembali ke byte referensi yang sama.

`prepare.py` menyalin **kode saat dijalankan** ke staging baru. Laporan 5 September merekam hash seluruh file resizer `adfb83f1a3ce1fea6b924cf0d97ed1db7f4f68d8f5fe85c538c4bc02cb3b7e5a`. Setelah pengujian itu, `recognizeWordImage` mendapat guard crop seluruh gambar agar Bitmap sumber tidak ikut didaur ulang. Fungsi resampler dan entry yang diukur tidak berubah. Hash lama dalam laporan sengaja dipertahankan; pengujian baru harus menghasilkan laporan baru dengan hash sumbernya sendiri.

Snapshot sumber yang diuji diarsipkan dalam `training/evidence/braille-word-reader-v1/native-worklet-resizer-tested.ts.txt`. Verifier memeriksa hash snapshot, konstanta tensor, serta kesamaan ketiga fungsi konversi/resampling dengan kode saat ini. Template APK awal memuat native React Native 0.86.2 dan Expo 57.0.16, sementara bundle diagnostik dibuat dengan dependency workspace React Native 0.86.3 dan Expo 57.0.20; Nitro Image 0.15.1 dan Worklets 0.10.1 sama. Perbedaan ini dicatat dalam laporan dan memerlukan smoke test APK utama dengan seluruh dependency yang sesuai lockfile.

Jalankan harness DEX di [word-native-parity](../word-native-parity/README.md) dahulu untuk menghasilkan `source.png`. Python membutuhkan Pillow. Contoh persiapan dari direktori repository:

```powershell
C:\bm-envs\angelina-benchmark\Scripts\python.exe scripts/word-native-performance/prepare.py --source-png C:\bm-word-native-parity-20260905\output\source.png --stage C:\bm-word-native-performance-recheck
New-Item -ItemType Junction -Path C:\bm-word-native-performance-recheck\node_modules -Target (Join-Path (Get-Location).Path 'node_modules')
Set-Location C:\bm-word-native-performance-recheck
node node_modules/expo/bin/cli export:embed --platform android --entry-file index.tsx --dev false --minify true --bundle-output output/index.android.bundle --assets-dest output/assets --bytecode --max-workers 2
```

Gunakan template internal yang sudah diverifikasi dan memiliki versi native dependencies yang sama. Pengujian awal memakai `output/android/BrailleMe-0.1.0-internal-preview-universal-2026-09-04.apk`, SHA-256 `15a771ffe541bcfc82e60d729de3a02e08abd1cf656a8f5b5b3a86eec97ec2e9`. `rebundle.py` hanya mengganti bundle JS, menghapus tanda tangan lama, dan mempertahankan data lainnya. Perubahan dependensi native memerlukan template baru.

Dari staging, gunakan path absolut ke repository untuk script berikut:

```powershell
C:\bm-envs\angelina-benchmark\Scripts\python.exe C:\Users\rifqi\OneDrive\Documents\Github\brailleme\scripts\word-native-performance\rebundle.py --template C:\Users\rifqi\OneDrive\Documents\Github\brailleme\output\android\BrailleMe-0.1.0-internal-preview-universal-2026-09-04.apk --bundle output/index.android.bundle --output output/unsigned.apk
$wordBuildTools = Join-Path $env:LOCALAPPDATA 'Android\Sdk\build-tools\36.0.0'
$wordJavaBin = 'C:\Users\rifqi\.jdks\brailleme-temurin-17\jdk-17.0.20+8\bin'
& "$wordBuildTools\zipalign.exe" -p -f 4 output/unsigned.apk output/aligned.apk
& "$wordJavaBin\keytool.exe" -genkeypair -keystore diagnostic.keystore -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 30 -dname 'CN=Word Resizer Diagnostic,O=Local Test,C=ID' -storepass android -keypass android
& "$wordBuildTools\apksigner.bat" sign --ks diagnostic.keystore --ks-key-alias androiddebugkey --ks-pass pass:android --key-pass pass:android --out output/diagnostic-only.apk output/aligned.apk
& "$wordBuildTools\apksigner.bat" verify --verbose --print-certs output/diagnostic-only.apk
& "$wordBuildTools\zipalign.exe" -c 4 output/diagnostic-only.apk
& "$wordBuildTools\aapt.exe" dump badging output/diagnostic-only.apk
Get-FileHash output/diagnostic-only.apk
```

Sertifikat di atas khusus diagnostik lokal. `apksigner.bat` membutuhkan `JAVA_HOME` yang menunjuk JDK 17 jika shell belum dikonfigurasi. Pasang pada emulator terpisah; jangan menimpa instalasi perangkat pengguna dengan paket yang bertanda tangan berbeda. Jalankan activity paket dari hasil `aapt`, tanpa server Metro, dan simpan log `WORD_NATIVE_PERF_RUN` serta `WORD_NATIVE_PERF_RESULT`. Laporan awal dan tangkapan layar berada di `C:\bm-word-native-perf-20260905\output`; masing-masing hash artefak tercatat dalam laporan evidence.
