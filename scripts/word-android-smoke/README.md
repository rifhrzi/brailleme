# Helper ADB untuk smoke test kata

Helper Python stdlib ini menyiapkan tindakan dan bukti untuk [protokol APK utama](../../docs/testing/WORD-APK-NATIVE-SMOKE-ID.md). Helper tidak menyalakan emulator, memasang APK, memodifikasi aplikasi, atau menetapkan hasil pengujian lulus. Gunakan setelah prasyarat model/build pada protokol terpenuhi.

Semua perintah memerlukan serial eksplisit `emulator-NNNN`. Sebelum berinteraksi, helper memeriksa state ADB dan `ro.kernel.qemu=1`; perangkat fisik ditolak. Gunakan `--adb <path-adb.exe>` jika ADB tidak ditemukan melalui PATH atau direktori Android SDK.

```powershell
$wordSmoke = @('scripts/word-android-smoke/smoke.py', '--serial', 'emulator-5554', '--output', 'C:\bm-word-apk-20260905\evidence\main-smoke')
python @wordSmoke dump --label ready
python @wordSmoke wait --description 'Pindai kata' --enabled --timeout 15
python @wordSmoke tap --description 'Pindai kata'
python @wordSmoke screenshot --label capture-01
python @wordSmoke dump --label capture-01
python @wordSmoke tap --description 'Bacakan'
python @wordSmoke tap --description 'Pindai ulang'
python @wordSmoke logcat --lines 1000
```

`dump` menyimpan XML hierarki dan menampilkan atribut elemen berteks/deskripsi. Pilih `--text` untuk atribut `text`, atau `--description` untuk `content-desc`; pencocokan selalu persis. Elemen tersembunyi, di luar batas induk, dan tombol nonaktif tidak ditap. Dua target yang cocok pada lokasi berbeda menghasilkan error. Titik tap dihitung dari batas elemen yang terlihat, sehingga tidak bergantung pada koordinat layar tetap.

`wait` mengamati kemunculan elemen; tambahkan `--enabled` untuk menunggu tombol aktif. Polling dibatasi 0,25–30 detik, dengan jeda paling lama 0,5 detik dan timeout per operasi ADB paling lama 8 detik. Error dump sementara dicatat dan dicoba lagi selama batas waktu masih tersedia. Serial/QEMU guard memiliki timeout tersendiri sebelum polling.

Untuk percobaan tap ganda, `tap --description 'Pindai kata' --count 2` mengirim dua tap berturut-turut ke lokasi target dari satu hierarki. Catat observasi jumlah pengambilan pada APK; dua perintah input bukan bukti bahwa pencegahan pengambilan ganda berhasil.

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
