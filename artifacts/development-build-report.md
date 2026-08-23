# Laporan Development Build BrailleMe

Tanggal verifikasi: 9 Agustus 2026  
Target: Android development client, classifier satu sel Braille

## Kesimpulan

Development build berhasil dibuat, ditandatangani untuk debug, dipasang, dan dijalankan pada emulator Android API 36. Jalur native kamera → crop tengah → resize 28 × 28 → tensor RGB float32 → inferensi TFLite CPU → quality gate → stabilisasi temporal berjalan tanpa crash setelah cold reload.

Build ini **belum lulus sebagai pembaca Braille yang akurat atau sempurna**. Pada uji kamera emulator ber-ground-truth, fixture kelas A menghasilkan top-1 V dengan confidence 87,74%. Karena ambang sementara adalah 90%, aplikasi menolak kandidat tersebut dan tidak menampilkannya sebagai hasil stabil. Abstention bekerja sesuai desain, tetapi kesalahan top-1 menunjukkan model dan preprocessing kamera masih memerlukan data fisik, kalibrasi, serta retraining sebelum fitur dapat dipakai sebagai pembaca.

## Artifact APK

| Target | File | Ukuran | SHA-256 | ABI |
| --- | --- | ---: | --- | --- |
| Ponsel Android modern | `brailleme-development-arm64-v8a.apk` | 144.095.620 byte | `50632AE514F531509312080A026B299FEF60ECADFE37B5F0CE28C5F13FF1EBB6` | `arm64-v8a` |
| Emulator Android | `brailleme-development-x86_64.apk` | 146.146.303 byte | `4CE2AED8EB412576403BC5ABD6814E7EE4E36335A4081DF6F4ED660AE7EA5621` | `x86_64` |

Kedua APK:

- berisi `libNitroImage`, `libNitroModules`, `libNitroTflite`, `libVisionCamera`, `libVisionCameraWorklets`, dan `libworklets`;
- lolos verifikasi signature APK v2;
- lolos verifikasi `zipalign`;
- memiliki package `com.brailleme.app`, versi `0.1.0`, `minSdk 29`, dan `targetSdk 36`.

APK ini adalah development client dan masih memerlukan Metro. Izin development/transitif seperti `INTERNET` dan `SYSTEM_ALERT_WINDOW` membuatnya bukan kandidat rilis.

## Kontrak model

- Artifact: `braille_model.tflite`
- SHA-256: `096d2822321732fd8e0e46fd6ca53e8569cc533f81a5593827d439f1a51f18f5`
- Ukuran: 979.488 byte
- Input: float32 NHWC `[1, 28, 28, 3]`
- Output: softmax float32 `[1, 26]`
- Label eksternal: `a` sampai `z`
- Status manifest: `unverified_candidate`

Model tidak memiliki metadata normalisasi, label tertanam, kelas background/unknown, spasi, detector, atau segmenter. Ia hanya dapat menerima crop satu sel yang sudah tepat.

## Hasil pemeriksaan

| Pemeriksaan | Hasil |
| --- | --- |
| Hash, ukuran, dan jumlah label model | Lulus |
| TypeScript strict typecheck | Lulus |
| ESLint | Lulus tanpa warning |
| Jest | 9 suite, 66 test lulus |
| Expo Android export | Lulus; model 979.488 byte ikut terpaket dengan hash identik |
| Dependency compatibility check | Lulus |
| Expo Doctor | 19/20; dua modul native belum ditandai teruji New Architecture oleh React Native Directory |
| Native build x86_64 | Lulus |
| Native build arm64-v8a | Lulus |
| Install dan model-load pada AVD API 36 | Lulus |
| Kamera, crop, preprocessing, dan inferensi native | Berjalan |
| Pause/resume kamera | Berjalan |

## Uji runtime

1. Scene sintetis biasa menghasilkan confidence sekitar 69–71% dan ditolak tanpa hasil stabil. Ini memverifikasi jalur negatif serta mencegah model A–Z memaksakan huruf pada scene yang tidak meyakinkan.
2. Fixture publik `a1.JPG0dim.jpg` dari [Braille Character Dataset](https://www.kaggle.com/datasets/shanks0465/braille-character-dataset), dengan ground truth A berdasarkan nama file dataset, dialirkan melalui fitur resmi emulator [`-camera-back imagefile`](https://developer.android.com/studio/run/emulator-commandline).
3. Pada jalur kamera lengkap, top-1 adalah V dengan confidence 87,74%, margin 81,22%, luminance 0,678, dan kontras 0,124. Kandidat salah tersebut ditolak karena confidence di bawah 90%.

Bukti visual:

- [Scene negatif dan status abstain](brailleme-model-probe-direct.png)
- [Diagnostik scene negatif](brailleme-model-probe-diagnostics.png)
- [Fixture A di ROI kamera](brailleme-model-probe-fixture-a-roi.png)
- [Top-1 V ditolak pada fixture berlabel A](brailleme-model-probe-labelled-fixture-diagnostic.png)

## Eksplorasi dataset, bukan bukti holdout

Forward-pass graf/weight secara independen pada 1.560 gambar dataset publik menghasilkan 1.526/1.560 benar (97,82%) untuk kandidat normalisasi RGB `0..1`. Angka ini tidak boleh dipakai sebagai klaim akurasi produk karena:

- provenance dan train/test split model tidak tersedia;
- dataset tersebut sangat mungkin sama atau dekat dengan distribusi training;
- evaluasi belum membuktikan parity dengan runtime LiteRT pada perangkat;
- model menunjukkan kesalahan ber-confidence tinggi;
- data JPEG datar tidak mewakili Braille timbul, variasi cahaya, fokus, sudut, dan kamera ponsel.

## Gate yang masih wajib

Sebelum menyatakan fitur siap, lakukan paling sedikit:

1. Konfirmasi preprocessing, channel order, dan urutan label dari kode training atau golden sample resmi.
2. Kumpulkan corpus fisik ber-ground-truth: 20 baris development dan 10 baris holdout dengan sedikitnya tiga variasi pengambilan.
3. Bekukan threshold sebelum membuka holdout.
4. Capai stable-token precision ≥90%, coverage ≥60%, p95 ROI-layak-ke-stabil ≤2 detik, preview ≥24 FPS, nol false utterance pada sedikitnya 12 scene negatif, dan soak 10 menit.
5. Tambahkan kelas unknown/background atau calibrated out-of-distribution rejection.
6. Untuk pembacaan kata/baris, tambahkan deteksi ROI, grid, segmentasi per sel, dan spasi berbasis geometri sebelum classifier ini dipanggil.

Sampai gate tersebut terpenuhi, layar **Uji Model Satu Sel** harus tetap dianggap alat eksperimen development dan hasilnya wajib diverifikasi manusia.
