# Helper ADB untuk smoke test kata

## Integritas APK sebelum dipasang

`verify_apk.py` memakai Python 3.11+ dan hanya membaca APK serta tool SDK lokal. Semua path diberikan eksplisit; ADB dan serial tidak diperlukan. Contoh setelah APK baru selesai dibangun:

```powershell
python scripts/word-android-smoke/verify_apk.py --apk 'C:\bm-symbols-apk-20260908\BrailleMe-0.3.0-internal.apk' --repo 'C:\Users\rifqi\OneDrive\Documents\Github\brailleme' --build-tools 'C:\Users\rifqi\AppData\Local\Android\Sdk\build-tools\36.0.0' --java-home 'C:\Users\rifqi\.jdks\brailleme-temurin-17\jdk-17.0.20+8' --output 'C:\bm-symbols-apk-20260908\evidence\apk-integrity-v1'
```

Folder output harus belum ada. Helper menyimpan SHA-256 APK, hash model sumber/terbundel dan manifest workspace, inventaris ABI, hasil `apksigner verify --verbose --print-certs`, `zipalign -c -P 16 -v 4`, serta `aapt dump badging`/`permissions`. Java dijalankan langsung dari JDK 17 yang diberikan; `JAVA_HOME` diset hanya untuk proses tool. Kontrak internal ini mensyaratkan paket `com.brailleme.app`, versi 0.3.0/kode 3, minimum SDK 29, tepat ABI `arm64-v8a` dan `x86_64`, DN sertifikat Android Debug, izin CAMERA, dan ketiadaan INTERNET serta izin penyimpanan/media.

Gambar `nativeSelfTest` juga diperiksa: hash sumber harus cocok dengan manifest, dimensi PNG harus cocok dengan anotasi, dan tepat satu entri PNG di APK harus memiliki hash serta ukuran berkas yang sama. Semua entri PNG diperiksa agar perubahan nama resource oleh Metro tidak menjadi alasan melewati pemeriksaan. Identitas sumber gambar, ukuran, dan hash manifest ikut dibekukan dalam laporan.

Untuk setiap `lib/<ABI>/*.so`, helper membaca ELF64 little-endian, memeriksa `e_machine` 183 untuk ARM64 atau 62 untuk x86_64, lalu mencatat setiap segmen `PT_LOAD`. Header tidak valid atau ABI salah menggagalkan integritas APK. Alignment segmen di bawah 16.384 byte dicatat pada `deployment_limitations` dan `elf_16kb_segment_alignment_passed: false`; ini tidak disamakan dengan hasil `zipalign`. Sesuai lingkup APK internal untuk pengujian perangkat 4 KB, keterbatasan segmen 16 KB tidak otomatis menggagalkan pemeriksaan integritas lainnya. Header yang lolos pun belum membuktikan runtime 16 KB; lihat [panduan pemeriksaan ELF dan ZIP Android](https://developer.android.com/guide/practices/page-sizes#elf-alignment).

Setiap stdout/stderr dan exit code disimpan. Exit code tool yang gagal, ketidakcocokan hash/kontrak, ZIP rusak, atau perubahan APK selama pemeriksaan menghasilkan `status: failed` dan exit 1. `passed_internal_apk_integrity` hanya membuktikan pemeriksaan berkas tersebut; pengujian aplikasi, bridge native, kamera fisik, dan kesiapan rilis tetap terpisah. Helper tidak memasang, menyalin, menandatangani ulang, atau menimpa APK/bukti sebelumnya.

## Interaksi emulator

Helper Python stdlib ini menyiapkan tindakan dan bukti untuk [protokol APK utama](../../docs/testing/WORD-APK-NATIVE-SMOKE-ID.md). Helper tidak menyalakan emulator, memasang APK, memodifikasi aplikasi, atau menetapkan hasil pengujian lulus. Gunakan setelah prasyarat model/build pada protokol terpenuhi.

Semua perintah memerlukan serial eksplisit `emulator-NNNN`. Sebelum berinteraksi, helper memeriksa state ADB dan `ro.kernel.qemu=1`; perangkat fisik ditolak. Gunakan `--adb <path-adb.exe>` jika ADB tidak ditemukan melalui PATH atau direktori Android SDK.

```powershell
$wordSmoke = @('scripts/word-android-smoke/smoke.py', '--serial', 'emulator-5554', '--output', 'C:\bm-symbols-apk-20260908\evidence\main-smoke')
python @wordSmoke dump --label ready
python @wordSmoke wait --description 'Pindai' --enabled --timeout 15
python @wordSmoke tap --description 'Pindai'
python @wordSmoke screenshot --label capture-01
python @wordSmoke dump --label capture-01
python @wordSmoke tap --description 'Bacakan'
python @wordSmoke tap --description 'Pindai ulang'
python @wordSmoke logcat --lines 1000
```

`dump` menyimpan XML hierarki dan menampilkan atribut elemen berteks/deskripsi. Pilih `--text` untuk atribut `text`, atau `--description` untuk `content-desc`; pencocokan selalu persis. Elemen tersembunyi, di luar batas induk, dan tombol nonaktif tidak ditap. Dua target yang cocok pada lokasi berbeda menghasilkan error. Titik tap dihitung dari batas elemen yang terlihat, sehingga tidak bergantung pada koordinat layar tetap.

`wait` mengamati kemunculan elemen; tambahkan `--enabled` untuk menunggu tombol aktif. Polling dibatasi 0,25–30 detik, dengan jeda paling lama 0,5 detik dan timeout per operasi ADB paling lama 8 detik. Error dump sementara dicatat dan dicoba lagi selama batas waktu masih tersedia. Serial/QEMU guard memiliki timeout tersendiri sebelum polling.

Untuk percobaan tap ganda, `tap --description 'Pindai' --count 2` mengirim dua tap berturut-turut ke lokasi target dari satu hierarki. Catat observasi jumlah pengambilan pada APK; dua perintah input bukan bukti bahwa pencegahan pengambilan ganda berhasil.

Perintah tambahan:

- `start --component <package/activity>` menjalankan komponen eksplisit yang sebelumnya diperoleh dari `aapt dump badging`; helper tidak menebak activity.
- `key HOME`, `key BACK`, atau `key APP_SWITCH` membantu pengujian interupsi/latar belakang.
- `screenshot --label ...` memakai `screencap` ke file unik di `/sdcard`, lalu `adb pull`. Tidak ada redirection biner melalui PowerShell.
- `logcat --lines ...` menyimpan maksimum 2.000 baris yang sudah tersedia, tanpa menghapus logcat.

Setiap perintah, kegagalan, elemen yang dipilih, koordinat tap, serta path/hash XML, PNG, dan log dicatat pada `actions.jsonl`. Artefak memakai nama unik. File sementara di emulator dihapus hanya setelah berhasil ditarik; kegagalan pembersihan dicatat. Semua pengambilan, termasuk penolakan, tetap dicatat. Bandingkan hasil kata secara manual dengan anotasi fixture yang dibekukan; jangan mengambil expected text dari keluaran aplikasi. Tangkapan layar atau event UI tidak membuktikan suara terdengar maupun akurasi kamera fisik.

Pemeriksaan helper tanpa ADB/perangkat:

```powershell
python -m unittest discover -s scripts/word-android-smoke -p 'test_*.py' -v
python scripts/word-android-smoke/smoke.py --help
```

Tes tersebut hanya menguji parsing hierarki, guard emulator, koordinat tap, batas polling, pencatatan, dan alur screenshot dengan mock. Itu bukan bukti smoke test APK.

## Kolektor runtime 10 menit

Jalankan setelah APK **preview** terpasang dan layar scanner sudah dibuka pada emulator yang dipilih:

```powershell
python scripts/word-android-smoke/collect_runtime.py --serial emulator-5554 --output output/runtime-20260908 --duration 600
```

Kolektor hanya membaca status ADB/QEMU, PID aplikasi, log agregat `BRAILLE_WORD_TRACKING_METRICS` dan `BRAILLE_WORD_PHOTO_METRICS`, `dumpsys meminfo`, diagnostik layanan kamera, serta timestamp presentasi SurfaceFlinger. Kolektor tidak mengubah kamera, menekan tombol, memasang APK, atau menghapus log. Arahkan fixture dan lakukan skenario pindai secara terpisah selama pengamatan. Semua perintah ADB memakai serial eksplisit; perangkat fisik ditolak.

Artefak `metadata.json`, `samples.jsonl`, `summary.json`, dan diagnostik layanan kamera disimpan dalam folder baru/kosong. Ringkasan memuat perubahan PSS, jumlah frame presentasi baru, waktu foto, dan histogram latensi tracker. Jika pemilihan surface ambigu, FPS dibiarkan kosong. Gunakan `--surface-layer '<nama persis dari dumpsys SurfaceFlinger --list>'` untuk memilih layer yang sudah diperiksa.

Pada Android 16/API 36, keluaran seperti `RequestedLayerState{com.brailleme.app/com.brailleme.app.MainActivity#246 parentId=245}` dinormalisasi menjadi `com.brailleme.app/com.brailleme.app.MainActivity#246` sebelum memanggil `--latency`. Pemilihan otomatis mengutamakan satu nama buffer `package/activity` yang persis; container dengan awalan heksadesimal, `ActivityRecord`, dan `ActivityRecordInputSink` tidak dipilih. Jika tidak ada buffer activity, satu `SurfaceView` dapat dipilih. Dua buffer atau dua kandidat SurfaceView tetap dianggap ambigu. Opsi `--surface-layer` menerima nama bersih maupun pembungkus API 36 yang lengkap. Hasil pengamatan lama tanpa surface tetap disimpan; ulangi pengukuran di folder baru.

**FPS surface adalah indikator komposisi tampilan, bukan bukti langsung FPS kamera.** Frame TextureView bisa digabung dengan surface aplikasi. Penghitung `busyDrops`/`throttled` hanya mencakup callback aplikasi, karena CameraX pada versi terpasang tidak menyediakan laporan semua frame yang dibuang. Histogram bersifat kumulatif per mount tracker dan tidak membuktikan persentil secara tepat. Kolektor selalu menulis `acceptanceResult: not_assessed`; bukti emulator tidak menetapkan akurasi Braille fisik atau kesiapan rilis.
