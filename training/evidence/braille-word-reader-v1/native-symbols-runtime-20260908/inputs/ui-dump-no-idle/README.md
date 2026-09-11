# Snapshot UI tanpa menunggu idle

`BrailleUiDump` adalah helper `app_process` terpisah, bukan perubahan aplikasi. `uiautomator dump` bawaan menunggu idle dan dapat gagal saat bingkai deteksi terus diperbarui. Helper ini menghubungkan `UiAutomation` dengan `FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES`, meminta tree terkompresi beserta ID view, membersihkan cache, lalu membaca root aktif langsung. Tidak ada `waitForIdle`, perubahan animasi, injeksi input, instalasi APK, atau pengaturan izin.

Konstruktor/koneksi hidden API diakses lewat refleksi. DEX sudah dikompilasi terhadap SDK Android 36 dengan min API 29. Tujuh pemeriksaan host memvalidasi nama output dan karakter XML; itu bukan bukti koneksi Android. Build v1 gagal pada emulator karena main Looper belum disiapkan oleh `app_process`; crash helper tersimpan pada `C:\bm-symbols-apk-20260908\evidence\no-idle-first-attempt.log`. Build v2 menambahkan `Looper.prepareMainLooper()` sebelum membuat koneksi. Laporan compile v2: `C:\bm-symbols-apk-20260908\evidence\ui-dump-no-idle-build-v2\report.json`; bukti v1/v2 dipertahankan.

Build v2 berhasil tersambung tetapi root belum tersedia pada pembacaan pertama. Build v3 menunggu root yang hilang dengan polling 100 ms selama maksimal lima detik, tanpa menunggu idle. Root telah menjalankan v3 pada emulator: 31 node, `root-wait-ms=117`, durasi helper 319 ms. XML aktual: `C:\bm-symbols-apk-20260908\evidence\no-idle-v3-first.xml`, SHA-256 `a142667fa8359bd484f422b0e4bf22e79507b8ec2a85f9219f364dca929520e9`. DEX v3 SHA-256 `79e234d57492167293889ef7df22815d53caa2f376b8e1ccaf1da031f4dee30a`. Ini bukti pembacaan UI, bukan scan atau hasil runtime aplikasi.

Root task menjalankan perintah berikut secara eksklusif setelah memeriksa serial `emulator-5554`, `ro.kernel.qemu=1`, dan AVD `BrailleMe_API_36`. Jangan bersamaan dengan proses UiAutomation lain. Perintah berikut adalah resep pemakaian ulang.

```powershell
$wordUiAdb = 'C:\Users\rifqi\AppData\Local\Android\Sdk\platform-tools\adb.exe'
& $wordUiAdb -s emulator-5554 push 'C:\bm-symbols-apk-20260908\evidence\ui-dump-no-idle-build-v3\classes.dex' /data/local/tmp/brailleme-ui-dump-v3.dex
& $wordUiAdb -s emulator-5554 shell env CLASSPATH=/data/local/tmp/brailleme-ui-dump-v3.dex app_process /system/bin BrailleUiDump /data/local/tmp/brailleme-ui-v6-001.xml
& $wordUiAdb -s emulator-5554 pull /data/local/tmp/brailleme-ui-v6-001.xml 'C:\bm-symbols-apk-20260908\evidence\brailleme-ui-v6-001.xml'
```

Gunakan timeout proses luar 15 detik dan nama remote/local baru setiap pengamatan. Helper menolak output yang sudah ada dan hanya mengizinkan `/data/local/tmp/brailleme-ui-NAME.xml`. XML dibuat lengkap di memori sebelum `CREATE_NEW`. Root yang belum ada dipoll maksimal lima detik, lalu traversal dibatasi lima detik, 10.000 node, dan kedalaman 128; root tetap hilang, child berubah/hilang, atau batas tercapai menghasilkan exit 1 tanpa XML sukses. Koneksi dilepas dalam `finally`; kegagalan disconnect juga menghasilkan exit 1. Kolom password dikosongkan.

Root XML memiliki `snapshot-mode="no-idle"` dan waktu uptime mulai. Setiap node menyertakan teks/deskripsi, kelas, package, enabled/clickable/scrollable, visible-to-user, serta bounds native agar parser `smoke.py` dapat digunakan. XML mengamati tree aktif selama traversal dan **bukan snapshot atomik seluruh layar**. Cocokkan dengan PID, screenshot, dan timestamp; status Pindai enabled tidak membuktikan foto diambil atau selesai. Penggunaan helper dan waktu observasi harus dicatat karena pembacaan aksesibilitas juga memberi beban pada runtime. Tidak ada status penerimaan aplikasi atau klaim fisik yang ditetapkan oleh helper.

Build ulang lokal, tanpa ADB:

```powershell
& 'C:\bm-envs\word-symbols-tensorflow\Scripts\python.exe' output/update-symbols-tracking-20260908/ui-dump-no-idle/build.py --java-home 'C:\Users\rifqi\.jdks\brailleme-temurin-17\jdk-17.0.20+8' --android-jar 'C:\Users\rifqi\AppData\Local\Android\Sdk\platforms\android-36\android.jar' --d8 'C:\Users\rifqi\AppData\Local\Android\Sdk\build-tools\36.0.0\d8.bat' --output 'C:\bm-symbols-apk-20260908\evidence\ui-dump-no-idle-build-v4'
```
