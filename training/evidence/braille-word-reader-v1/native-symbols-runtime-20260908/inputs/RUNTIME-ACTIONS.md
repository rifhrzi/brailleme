# Aksi runtime 0.3.0 yang dapat diulang

Untuk overlay kamera yang terus diperbarui, gunakan [helper snapshot tanpa idle](ui-dump-no-idle/README.md). Root sudah menjalankan DEX v3 pada emulator: 31 node, root tersedia setelah 117 ms, durasi helper 319 ms; XML pertama tersimpan di `C:\bm-symbols-apk-20260908\evidence\no-idle-v3-first.xml`. Ini memverifikasi pembacaan hierarki, bukan hasil scan atau penerimaan runtime.

Tambahkan kedua argumen berikut ke perintah harness setelah root memasang DEX tersebut:

```powershell
$wordNoIdle = @('--ui-dump-dex', '/data/local/tmp/brailleme-ui-dump-v3.dex', '--ui-dump-sha256', '79e234d57492167293889ef7df22815d53caa2f376b8e1ccaf1da031f4dee30a')
```

Contoh: `& $wordActionPython $wordActionScript @wordActionCommon @wordNoIdle --output <folder-baru> --duration 480 --interval 30 exercise`. Harness tidak mengunggah DEX. Sebelum **setiap** eksekusi, serial/QEMU/AVD dan SHA-256 file remote diperiksa; perubahan hash menghentikan eksekusi. Log dan metadata mencatat mode, hash, perintah, waktu tunggu root, serta identitas XML. Kegagalan helper atau XML yang tidak lengkap tidak digantikan oleh snapshot lama. Mode ini memakai batas proses 15 detik yang tetap dipotong deadline keseluruhan; ADB lain tetap maksimal delapan detik. Tanpa kedua argumen, perilaku `uiautomator dump` bawaan tetap berlaku.

`runtime-actions.py` memakai helper stdlib ADB yang sudah ada. Jalankan hanya setelah root membangun, memverifikasi, dan memasang APK baru. Harness ini tidak menyalakan emulator atau memasang APK. Setiap perintah wajib menunjuk `emulator-5554`; sebelum mutasi, harness memeriksa state ADB, `ro.kernel.qemu=1`, dan hasil `emu avd name` tepat `BrailleMe_API_36`. Versi aplikasi terpasang harus 0.3.0/kode 3. Activity diberikan eksplisit berdasarkan hasil `aapt dump badging`.

Contoh konfigurasi PowerShell berikut mengasumsikan activity hasil pemeriksaan adalah `com.brailleme.app/.MainActivity`:

```powershell
$wordActionPython = 'C:\bm-envs\word-symbols-tensorflow\Scripts\python.exe'
$wordActionScript = 'output/update-symbols-tracking-20260908/runtime-actions.py'
$wordActionCommon = @('--adb', 'C:\Users\rifqi\AppData\Local\Android\Sdk\platform-tools\adb.exe', '--serial', 'emulator-5554', '--component', 'com.brailleme.app/.MainActivity')
& $wordActionPython $wordActionScript @wordActionCommon --output 'C:\bm-symbols-apk-20260908\evidence\actions-prepare-v1' prepare
```

`prepare` membuka aplikasi terpasang dan mengikuti tombol yang benar-benar ditemukan pada XML: **Buka Beranda**, **Buka pemindai kata** atau **Lihat status pemindai**, **Aktifkan kamera**, **Izinkan kamera**, lalu tombol foreground permission Android berdasarkan resource ID. Error pemuatan model berhenti sebagai observasi gagal/incomplete; harness tidak menyembunyikannya dengan retry otomatis. Batas persiapan 120 detik.

Urutan pengujian yang menjaga pengukuran runtime tetap dapat ditafsirkan:

1. Jalankan `prepare` dan periksa screenshot untuk memastikan orientasi serta letak Braille sesuai feed.
2. Jalankan `cold-start` dengan folder bukti baru untuk force-stop, membuka aplikasi kembali, dan mencapai kamera. Ini menghasilkan PID baru.
3. Sesudah kamera siap, jalankan collector 600 detik dari shell/proses lain. Sesuai `runtime-protocol-v1.json`, biarkan preview tanpa aksi shutter selama 115 detik pertama, lalu jalankan `exercise --duration 480 --interval 30` dari proses terpisah. Catat waktu mulai aktual; fase yang direncanakan bukan bukti fase tersebut telah terjadi. Jangan menjalankan cold start atau perubahan font selama collector ini, karena collector berhenti bila PID berubah.
4. Sesudah collector selesai, jalankan `background` untuk HOME, menunggu tiga detik, kembali ke aplikasi, dan mendokumentasikan pemulihan kamera.
5. Jalankan `font-200` secara terpisah. Harness menyimpan font_scale sebelumnya, mengatur dan memverifikasi 2.0, mendokumentasikan UI serta satu siklus, lalu mencoba memulihkan dan memverifikasi nilai awal dalam `finally`.

```powershell
& $wordActionPython $wordActionScript @wordActionCommon --output 'C:\bm-symbols-apk-20260908\evidence\actions-cold-v1' cold-start
& $wordActionPython scripts/word-android-smoke/collect_runtime.py --adb 'C:\Users\rifqi\AppData\Local\Android\Sdk\platform-tools\adb.exe' --serial emulator-5554 --output 'C:\bm-symbols-apk-20260908\evidence\runtime-600-v1' --duration 600
```

Perintah collector di atas tetap berjalan; sesudah 115 detik awal, jalankan perintah berikut di proses terpisah:

```powershell
& $wordActionPython $wordActionScript @wordActionCommon --output 'C:\bm-symbols-apk-20260908\evidence\actions-600-v1' --duration 480 --interval 30 --feed-manifest 'C:\bm-symbols-apk-20260908\evidence\labelled-feeds-v1\manifest.json' --feed-name 'lowercase-abc.png' exercise
```

Gunakan `--feed-name` yang sesuai dengan feed emulator yang benar-benar dipasang saat startup. Argumen ini memverifikasi hash file berlabel, tetapi **tidak mengganti feed kamera** atau membuktikan bahwa emulator sedang menampilkannya. Konfirmasi gambar dan orientasi pada screenshot. Rekam perubahan rotasi/penempatan sebagai input baru; jangan mengganti label menurut hasil prediksi.

```powershell
& $wordActionPython $wordActionScript @wordActionCommon --output 'C:\bm-symbols-apk-20260908\evidence\actions-background-v1' background
& $wordActionPython $wordActionScript @wordActionCommon --output 'C:\bm-symbols-apk-20260908\evidence\actions-font200-v1' font-200
```

`exercise` hanya menekan **Pindai** yang berupa button, terlihat, dan enabled. Tab bawah yang juga bernama Pindai tidak dipakai sebagai shutter. Hasil sukses maupun penolakan diamati lewat UI, kemudian tombol **Pindai ulang** membuka percobaan berikutnya. Jika shutter nonaktif pada input kosong, aksi capture tidak dikirim dan keadaan tersebut dicatat. Jika kontrol berada di luar viewport, satu scroll per siklus memakai batas ScrollView dari XML. Batas durasi 1–600 detik, interval 10–60 detik, timeout masing-masing ADB paling lama delapan detik dan terpotong deadline keseluruhan.

Setiap aksi mempunyai catatan perintah, snapshot XML, screenshot PNG, logcat PID aplikasi bila tersedia, dan pengamatan teks/status/disabled. `pidof` exit 1 dengan stdout/stderr kosong dicatat sebagai PID tidak ada, misalnya sebelum aplikasi pertama kali dibuka atau sesudah force-stop. Snapshot awal dan sesudah force-stop boleh menyimpan `pid: null` serta `logcat: null`. Setelah aplikasi dibuka atau aksi UI dikirim, snapshot memerlukan PID hidup; jika PID hilang, XML/screenshot tetap disimpan tetapi hasil menjadi incomplete/exit 1, sekalipun UI lama masih menampilkan Pindai. Field `app_process_required` mencatat aturan tiap snapshot. Kegagalan ADB, timeout, atau daftar PID ambigu tetap menghentikan aksi. Folder output wajib belum ada. `summary.json` selalu menyatakan `acceptanceResult: not_assessed`; penyelesaian perintah bukan klaim akurasi, native parity, audio, atau validasi perangkat fisik. Aksi atau observasi yang tidak selesai menghasilkan exit 1 dan tetap menyimpan bukti yang sempat diperoleh. Semua prediksi yang mungkin terlihat tetap berada di bukti lokal tugas.

Feed yang diperiksa pada `labelled-feed-review-v1.json`: `abc`, `Abc`, `ABC`, `2026`, `001`, `a`, `aab`, dan kosong. Ketujuh gambar Braille memakai record pertama yang cocok dari manifest validasi, mempertahankan seluruh sumber 480×192 yang diperbesar 2× dengan Pillow bilinear ke 960×384, lalu ditempatkan pada [32,320,992,704] di kanvas 1024×1024. Label, hash sumber/output, parser indikator, dan komposisi piksel lolos pemeriksaan offline. Orientasi runtime belum dinilai.

Pemeriksaan lokal tanpa emulator:

```powershell
& $wordActionPython output/update-symbols-tracking-20260908/test-runtime-actions.py -v
& $wordActionPython $wordActionScript --help
```

Sesudah collector dan action harness selesai, `analyze-runtime.py` membaca bukti secara offline tanpa ADB. Berikan SHA APK yang telah diverifikasi dan folder output baru di luar direktori input. `--actions` dapat diulang untuk menyertakan aksi background/cold start/font yang dijalankan terpisah.

```powershell
& $wordActionPython output/update-symbols-tracking-20260908/analyze-runtime.py --runtime 'C:\bm-symbols-apk-20260908\evidence\runtime-600-v1' --actions 'C:\bm-symbols-apk-20260908\evidence\actions-600-v1' --apk $wordVerifiedApk --apk-sha256 $wordVerifiedApkSha256 --output 'C:\bm-symbols-apk-20260908\evidence\runtime-analysis-v1.json'
& $wordActionPython output/update-symbols-tracking-20260908/test-analyze-runtime.py -v
```

Laporan mencatat p95 foto dengan nearest-rank atas **semua** status success/rejected/error yang benar-benar tercatat, rincian status, PSS pertama/terakhir/maksimum, serta FPS komposisi surface sebagai proxy. Penghitung tracking dipisahkan saat counter atau bucket menurun. Penurunan `elapsedMs` saja tidak dianggap reset karena clock pelaporan dapat mulai ulang saat aktivitas kamera berubah. Dalam setiap segmen, hanya selisih snapshot terakhir dan pertama yang dijumlahkan; prefix awal dan tail tanpa laporan tidak diklaim. Rate merupakan batas bawah berdasarkan selisih waktu `deviceEpochSeconds`, tidak tersedia jika interval nonpositif. Histogram menghasilkan batas bucket p95, bukan nilai latency presisi.

Analyzer mengikat SHA APK aktual, source saat analisis, dan semua file input yang dibaca. Ini tidak membuktikan APK tersebut terpasang atau source saat analisis identik dengan build; pasangkan dengan bukti build/install. Bukti hilang, JSON rusak, durasi tidak lengkap, atau ketidaksesuaian jumlah dilaporkan dan menghasilkan exit 1 sambil mempertahankan statistik yang masih dapat dihitung. Tidak adanya foto dapat berarti tombol tidak ditekan atau input kosong; tidak otomatis dianggap kegagalan aplikasi. Tap Pindai tidak diklaim sebagai capture yang selesai. Laporan selalu `acceptanceResult: not_assessed`, tanpa PASS perangkat fisik atau rilis.
