# Pemeriksaan pixel foto kata

Harness ini menjalankan operasi `Bitmap` Android melalui DEX mandiri, tanpa memasang aplikasi atau menambahkan route debug. Tujuannya memeriksa orientasi, pemotongan, urutan kanal warna, dan perbedaan resize. Kecepatan Hermes, kamera fisik, model, serta aksesibilitas perlu diuji terpisah.

Gunakan emulator yang sudah menyala atau perangkat uji dan sesuaikan lokasi JDK:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/word-native-parity/run.ps1 -JavaDirectory 'C:\Users\rifqi\.jdks\brailleme-temurin-17\jdk-17.0.20+8' -Serial emulator-5554
node scripts/word-native-parity/pixel-reference.cjs C:\bm-word-native-parity-20260905\output
C:\bm-envs\angelina-benchmark\Scripts\python.exe scripts/word-native-parity/compare.py --evidence C:\bm-word-native-parity-20260905\output --geometry C:\bm-word-native-parity-20260905\geometry-reference.json
```

`-CompileOnly` menyiapkan DEX dan koordinat TypeScript tanpa mengakses Android. Script membutuhkan Android SDK platform 36/build-tools 36.0.0, JDK 17, Node dengan dependensi proyek, serta Python dengan NumPy/Pillow. APK aplikasi dan direktori staging lama tidak diubah.

`geometry-reference.cjs` menjalankan fungsi TypeScript aplikasi untuk mendapatkan koordinat integer yang benar-benar diberikan kepada native. Harness kemudian menjalankan rotasi/mirror dan crop Android, serta menyimpan bitmap dan byte hasilnya. `pixel-reference.cjs` menjalankan resampler TypeScript pada byte crop native di Node; `compare.py` membandingkan seluruh kanal dan tensor terhadap Pillow.

Laporan mempertahankan dua kesimpulan yang berbeda: `Bitmap.createScaledBitmap` tidak identik dengan Pillow; resampler antialias TypeScript yang dipakai jalur kata harus sesuai dengan Pillow. Foto kamera Android dari Nitro Image 0.15.1 juga memerlukan koreksi metadata BGRA ke byte RGBA. Konverter BGRA generik tidak diubah. Upgrade Nitro harus mengulang fixture warna ini.

Pembanding memakai RGB bilinear Pillow dengan koefisien 22-bit dan pembulatan setelah masing-masing pass; implementasi rujukan tersedia di [Pillow Resample.c](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c). Fixture sintetis ini menguji kontrak pixel, bukan akurasi pembacaan Braille.

Untuk foto uji startup, `fixture-geometry.cjs` mengambil koordinat crop integer dan selection dari `recognizeWordImage` aktual. `word_fixture.py` memeriksa crop serta seluruh byte RGB sumber terhadap cache validasi sebelum menjalankan model. Calon terpilih juga diperiksa dengan quality check, hash tensor, dan decoder TypeScript memakai output TFLite yang sudah dihitung; tidak ada inferensi kedua. Foto yang dipilih tetap harus lulus startup Android sebenarnya.

Audit tanpa TensorFlow/model dan regresi pembulatan dapat dijalankan dengan:

```powershell
C:\bm-envs\word-scan-torch-only\Scripts\python.exe training/scripts/word_fixture.py --run .braille-training-work/word-scan-v1-20260905/fixture-geometry-audit --data .braille-training-work/word-scan-v1-20260905/data --geometry-only
C:\bm-envs\word-scan-torch-only\Scripts\python.exe -m unittest discover -s training/tests -p test_word_fixture_geometry.py -v
```

Audit 135 foto validasi menemukan satu batas pembulatan nyata: crop `KOTIKI` dalam PNG sumber berubah dari `[42,42,252,127]` menjadi `[42,42,253,127]`. Record tersebut dikecualikan hanya dari pemilihan foto startup; benchmark dan piksel validasi tidak diubah. Sebanyak 134 record lain memiliki crop dan seluruh byte RGB hasil resize yang identik. Bukti tersimpan pada `training/evidence/braille-word-reader-v1/native-fixture-geometry-audit.json`.
