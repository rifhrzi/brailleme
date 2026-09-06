# Persiapan inferensi interpreter Android

Status 6 September 2026: runner x86_64/API 29 telah dikompilasi dan enam input oracle telah dibekukan. **Belum ada eksekusi Android, bukti operator berjalan, atau hasil parity native.** Emulator tidak dijalankan selama training. Folder ini tidak membuat APK, mengubah model aktif, atau melewati gate pengenalan.

Runner menggunakan C API dari `com.google.ai.edge.litert:litert:1.4.0`, dependency persis `react-native-fast-tflite` 3.0.1. Loader utama memanggil `loadTensorflowModel(..., [])`; kode Nitro membuat model dari byte, membuat options default tanpa setter thread/delegate, membuat interpreter, melepas wrapper model/options, lalu mengalokasikan tensor. `word_interpreter.c` mengikuti urutan tersebut dan mempertahankan byte model sampai interpreter dilepas. Copy input, invoke, dan pembacaan data output juga memakai C API yang sama. Kontrak publik tersedia pada [referensi C API](https://developers.google.com/edge/api/tflite/c).

Ini menghindari asumsi bahwa default Java `Interpreter` sama dengan default C API. File pustaka bernama `libtensorflowlite_jni.so`, tetapi runner memanggil ekspor C langsung; Java/JNI bootstrap, bridge Nitro, worklet, loader aplikasi dan lifecycle kamera tidak dijalankan. Backend default internal tetap milik build LiteRT tersebut; tidak ada pemaksaan XNNPACK/thread count agar cocok dengan desktop. Desktop emitter menggunakan TensorFlow TFLite dengan empat thread dan melaporkan konfigurasi sebenarnya.

## Kompilasi ringan tanpa Android

[NDK mendukung pemanggilan compiler langsung](https://developer.android.com/ndk/guides/other_build_systems), sehingga tidak perlu Gradle, DEX, atau pemuatan React Native. Gunakan stage baru; script menolak overwrite.

```powershell
& 'C:/bm-envs/angelina-benchmark/Scripts/python.exe' scripts/word-native-interpreter/compile.py --stage C:/bm-word-native-interpreter-20260906-final --aar 'C:/Users/rifqi/.gradle/caches/modules-2/files-2.1/com.google.ai.edge.litert/litert/1.4.0/28dce4edc35e39fb81d46b2c9ce5e22646ff52da/litert-1.4.0.aar' --ndk 'C:/Users/rifqi/AppData/Local/Android/Sdk/ndk/27.1.12297006'
```

Perintah di atas sudah dijalankan; ganti nama stage untuk pengulangan. Compile report menyimpan command, versi compiler, hash sumber, headers, AAR, runner, dan ELF. Hasil final:

| Artefak | SHA-256 |
| --- | --- |
| LiteRT 1.4.0 AAR | `54b567572cb2d1295571c785198f87597809cbf22d6cb8acfeac23869041887d` |
| Pustaka x86_64 dari AAR | `b1fdd3d23e58df54f8ce47c704ed4bc0882be37e479fe86a87108d5f8153c422` |
| Runner final | `995cedbcba3d29f5213ad47ba65d02ce4cc5aa0aac6e18db566aedad44ca0503` |
| `compile-report.json` final | `7cec578d0da01ec25a0e310f0877b1118efc41c1d24c7687d3546668af871064` |

Pustaka AAR identik dengan dependency native stage kamera. Packaging APK diagnostik sebelumnya menghasilkan file stripped berbeda delapan byte dan hash `4d201057ca55005c7288436d21d521c915716c76d7d397e5c71afcba6edec5e7`; jangan menyebut kedua binary byte-identik. Pengujian ini mengikat pustaka AAR yang benar-benar dipush. Library hanya membutuhkan `libm`, `libdl`, `liblog`, dan `libc` Android; runner memakai RUNPATH `$ORIGIN` dan jalur library eksplisit saat eksekusi.

## Enam input dan kandidat terlatih

`prepare.py` tanpa `--desktop-report` hanya membekukan enam input asli dari `geometry-head-parent-v1/initialization-oracle.npz`. Hash NPZ harus `9330946dafe06c7705b3f2c85cce26f8bbfeca11763ba7465d62e809180d5669`: tiga crop TRAIN serta gambar hitam, putih dan random deterministik. Setiap file berisi RGB float32 little-endian `[1,192,480,3]`, tepat 1.105.920 byte, tanpa resize atau pembulatan RGB8 baru.

```powershell
& 'C:/bm-envs/angelina-benchmark/Scripts/python.exe' scripts/word-native-interpreter/prepare.py --oracle .braille-training-work/word-scan-v1-20260905/geometry-head-parent-v1/initialization-oracle.npz --initialization-report .braille-training-work/word-scan-v1-20260905/geometry-head-parent-v1/geometry-initialization.json --output C:/bm-word-native-interpreter-20260906-final/oracle-inputs
```

Sesudah seleksi dan ekspor kandidat terlatih baru lulus parity 405, emitter training menulis `native-six-desktop-outputs.npy` (`[6,48,120,11]`, little-endian float32, C-contiguous) serta `native-six-desktop.json`. Laporan mengikat model, checkpoint, oracle, keluaran, threshold, versi interpreter, thread/delegate dan operator yang benar-benar terlihat. Tambahkan `--desktop-report <selected-run>/native-six-desktop.json` dan pilih output bundle baru untuk menyiapkan kandidat itu. Preparer memverifikasi keterikatan checkpoint/model/405 parity/threshold sebelum menyalin model ke folder diagnostik.

Jika kontrol tanpa cabang geometry tetap terpilih, tidak ada bukti GATHER/FLOOR/BROADCAST_TO/TILE dari pengujian kontrol itu. Tunda kandidat geometry sampai tersedia. Conversion probe nonzero lama hanya boleh diuji sebagai diagnostik inisialisasi yang diberi label terpisah, tidak pernah diaktifkan atau dilanjutkan training. Script ini sengaja mensyaratkan laporan kandidat terlatih untuk tahap eksekusi normal.

## Eksekusi sesudah jendela resource disepakati

Jalankan hanya setelah training, seleksi, ekspor dan benchmark melepas resource. Script membutuhkan perangkat/emulator yang sudah menyala; tidak menyalakan emulator. Ganti path bundle dan output berikut dengan direktori kandidat yang disiapkan, lalu gunakan serial eksplisit:

```powershell
& 'C:/bm-envs/angelina-benchmark/Scripts/python.exe' scripts/word-native-interpreter/run.py --compiled C:/bm-word-native-interpreter-20260906-final --bundle C:/bm-word-native-trained-six-bundle --output C:/bm-word-native-trained-six-result --adb 'C:/Users/rifqi/AppData/Local/Android/Sdk/platform-tools/adb.exe' --serial emulator-5554
node scripts/word-native-interpreter/compare.cjs C:/bm-word-native-trained-six-bundle C:/bm-word-native-trained-six-result C:/bm-word-native-trained-six-result/parity.json
```

Transfer memakai folder Android unik dan memeriksa SHA-256 pada perangkat untuk model, library, executable serta setiap input sebelum invoke. Runner memverifikasi jumlah/dimensi/tipe/ukuran tensor; nilai input dan output harus finite serta berada dalam `[0,1]`. Satu interpreter dipakai berurutan. Output tiap input disimpan sebagai 253.440 byte float32, dengan indeks, status, durasi invoke, dan error stage. Completion melaporkan `getrusage` peak RSS dalam KiB; nilainya belum diketahui sebelum eksekusi.

Comparator mensyaratkan keenam indeks/file tepat, output finite, dan seluruh elemen memenuhi `abs(native-desktop) <= 1e-4 + 1e-4*abs(desktop)`. Ia menjalankan decoder TypeScript aplikasi sebenarnya pada kedua tensor menggunakan threshold kandidat yang sama dan membandingkan status, teks, alasan serta urutan mask. Perbandingan hanya memakai selection seluruh input; random float tetap utuh, sehingga tidak ada pemeriksaan kualitas RGB8 atau klaim pipeline kamera. Semua file, transfer dan sumber decoder diikat dengan hash. Kegagalan tensor atau keputusan menghasilkan exit nonzero; threshold tidak dilonggarkan.

`node --test scripts/word-native-interpreter/compare.test.cjs` memeriksa lima kasus sintetis: enam tensor identik, output hilang, output berlebih, galat numerik walaupun keputusan sama, dan perubahan keputusan walaupun tensor masih dalam toleransi. File uji dibuat sementara dan dihapus; hasil ini hanya membuktikan guard comparator, bukan inferensi native.

Enam input ini memeriksa jalur cabang geometry pada interpreter aktual, bukan akurasi kata. Pengujian 405 input native yang lebih luas dapat disiapkan terpisah bila waktunya memungkinkan. Parity model lengkap, benchmark 607 kata, smoke test APK utama tanpa Metro, dan pengujian kertas timbul/aksesibilitas tetap memiliki bukti masing-masing.
