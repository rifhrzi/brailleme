# Diagnostik geometri pengambilan kamera

[Arsip v3](../../training/evidence/braille-word-reader-v1/native-camera-capture-diagnostic-v3/README.md) menyimpan delapan pengambilan VisionCamera nyata pada kamera `imagefile` emulator, termasuk dua kegagalan 16:9. APK diagnostik terpisah tidak disimpan di repository. Bukti ini tidak membuktikan keberhasilan APK utama, pengenalan kata, atau kamera fisik.

Verifikasi seluruh hash tanpa dependensi tambahan:

```powershell
python scripts/word-native-capture/verify_archive.py
```

Replay posisi empat penanda, ROI, kesamaan JPEG/Nitro, dan perbandingan crop JPEG memakai NumPy 2.0.2 serta Pillow 12.3.0 dari lingkungan diagnostik:

```powershell
& 'C:/bm-envs/angelina-benchmark/Scripts/python.exe' scripts/word-native-capture/verify_archive.py --replay
```

Replay membuat salinan sementara, memeriksa semua input sebelum menjalankan analyzer asli, dan membandingkan seluruh laporan pengambilan dengan bukti yang dibekukan. Tidak ada ADB, emulator, build, atau inferensi model. File arsip tidak ditulis ulang. Versi library dicantumkan untuk mereproduksi byte PNG, bukan sebagai kewajiban upgrade dependency aplikasi.

`archive-index.json` mengikat setiap file arsip dan sumber verifier ini; `archive-index.sha256` mengikat indeksnya. `report.json` asli tetap identik dengan laporan akhir diagnostik. `source/geometry-entry.tsx` menyimpan UI harness sebenarnya; `source/` juga memuat sumber geometri bersama dan lockfile. `harness/` mempertahankan generator feed, driver ADB, analyzer, dan pembatas job build persis seperti yang dijalankan. Script lama berisi path stage/SDK asli; jangan menjalankannya langsung pada arsip yang dibekukan karena beberapa script menulis hasil.

Untuk pengambilan baru, salin harness dan sumber ke stage diagnostik baru, gunakan dependency sesuai lockfile, lalu perbarui path stage/SDK dalam salinan. Perintah build dan emulator lengkap ada di `report.json` dan `logs/`. Simpan hasil baru terpisah dan ukur keempat penanda yang terlihat pada pratinjau serta foto. Jangan mengubah pemetaan dari hasil model atau menerapkan koreksi 120 piksel. Emulator yang dipakai menaikkan permintaan awal RAM 1536 MiB menjadi 2560 MiB; pengulangan secara eksplisit meminta 2560 MiB setelah pekerjaan ML berhenti.

Pengujian APK utama tetap mengikuti [prosedur native smoke](../../docs/testing/WORD-APK-NATIVE-SMOKE-ID.md) setelah gate model lulus.
