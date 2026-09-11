# Rotasi folder build BrailleMe

Maksimal **dua folder kerja build** di luar repository untuk perbandingan. APK yang disimpan di arsip bukan folder kerja build; arsip tidak ikut dirotasi.

Gunakan PowerShell dari repository:

```powershell
# Lihat jumlah slot yang terisi.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/manage-build-folders.ps1 -Action Status

# Buat slot baru. Jika sudah ada dua, helper mengarsipkan dan menghapus
# folder terlama yang tidak aktif SEBELUM membuat folder ini.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/manage-build-folders.ps1 -Action Create -Path C:\bm-build-20260910-a

# Pastikan slot terdaftar/batas dipenuhi sebelum menyalin source atau build ulang.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/manage-build-folders.ps1 -Action Ensure -Path C:\bm-build-20260910-a

# Simpan snapshot artefak tanpa menghapus folder.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/manage-build-folders.ps1 -Action Archive -Path C:\bm-build-20260910-a
```

`Adopt` khusus untuk mendaftarkan folder lama setelah kepemilikan/isinya diperiksa; `package.json` harus bernama `brailleme`. Nama yang dikelola adalah anak langsung `C:\` dengan awalan `bm-build-`, `bm-apk-`, atau `bm-symbols-apk-`. Folder tak terdaftar memblokir rotasi otomatis. Jangan memakai nama lain untuk melewati batas.

Artefak disimpan di **`artifacts/build-archives/`** dalam repository lokal (diabaikan Git). Folder `blobs` menyimpan satu salinan per SHA-256 sehingga APK identik tidak berulang. Setiap snapshot memiliki `manifest.json` dengan path asli relatif, ukuran, hash, dan lokasi blob. Untuk memulihkan, salin setiap blob ke `tujuan-baru/relativePath` sesuai manifest setelah memeriksa hash. Pertahankan manifest bersama folder blobs; ini arsip lokal, bukan backup cloud.

Yang diamankan: sumber/config/aset lokal, seluruh `evidence`, serta `android/app/build/outputs` termasuk APK/AAB, mapping, dan metadata. Dependency terpasang, cache Gradle/CMake/Kotlin dan output kompilasi perantara dikecualikan karena dapat dibuat ulang. Jangan menyimpan perubahan sumber yang ingin dipertahankan hanya di `node_modules`; pindahkan patch ke repository sebelum rotasi.

Penghapusan memerlukan marker kepemilikan cocok, path valid, tidak ada junction/reparse point, tidak ada proses build/emulator terdeteksi memakai folder, tidak ada lock build aktif atau worktree terdaftar, dan hash sumber/arsip sesuai. Semua proses kelola menggunakan lock tunggal. Bila arsip, pemeriksaan, atau penghapusan gagal, folder baru tidak dibuat. Hentikan build/emulator yang memakai slot lama sebelum mencoba lagi; helper tidak mematikan proses otomatis.

`C:\bm-datasets`, `C:\bm-envs`, `C:\bm-symbols-20260908`, repository, checkpoint dan indeks retensi tetap dilindungi. Bukti lama yang menyebut path staging asli tidak ditulis ulang; manifest arsip menjadi peta lokasi penggantinya.

## Penerapan 9 September 2026

`C:\bm-symbols-apk-20260908` sudah didaftarkan dan dipertahankan sebagai satu dari dua slot. Tidak ada folder build asli dihapus atau dibuat dalam penerapan ini. Arsip memuat 1.236 file (1.167.530.340 byte sebelum deduplikasi), disimpan sebagai 981 blob unik (1.004.041.391 byte). Hash semua file sumber dan blob diperiksa kembali; APK distribusi, kedua checkpoint terbaik, dan indeks retensi/bukti lama cocok dengan hash sebelumnya.

[Laporan verifikasi dan lokasi manifest](build-retention-20260909.json) menyimpan rincian APK, SHA-256, aset yang dilindungi, dan hasil tes. Manifest beserta blobs berada di `artifacts/build-archives/`; log emulator juga sudah diarsipkan setelah emulator pengujian tanpa jendela ditutup.

Helper lokal lama `output/update-symbols-tracking-20260908/stage-source.cjs` dan `build-internal.ps1` sekarang memanggil `Ensure` sebelum menulis ke staging. Salinan asli kedua helper beserta hash ada di `artifacts/build-helper-backup-20260909/`. File dalam `output/` dan `artifacts/` bersifat lokal dan diabaikan Git; aturan permanen berada di `AGENTS.md` dan skrip pengelola yang dilacak repository.

Jalankan tes rotasi terisolasi dengan:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/tests/manage-build-folders.tests.ps1
```

Sepuluh kasus lulus: rotasi dan perlindungan artefak, penolakan tujuan yang sudah ada, path keluar batas, folder tak terdaftar, identitas adopsi, arsip rusak, junction, pengelola bersamaan, lock build aktif, serta log stabil dengan handle penulis bersama. Tes memakai folder kecil di `output/build-folder-tests/`, bukan staging asli. Pemeriksaan sintaks PowerShell/Node juga lulus; aplikasi tidak dibangun ulang untuk perubahan aturan ini.
