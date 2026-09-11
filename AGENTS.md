# Aturan workspace BrailleMe

## Folder build Android: maksimal dua

- Maksimal **2 folder build/staging BrailleMe** sekaligus, untuk membandingkan dua build.
- Buat/daftarkan folder melalui `scripts/manage-build-folders.ps1`; jangan membuat staging baru langsung dengan `mkdir`, penyalinan project, atau worktree yang melewati pemeriksaan batas.
- Saat kedua slot terisi, **arsipkan dan verifikasi artefak penting, hapus satu folder lama, verifikasi penghapusan, baru buat folder baru**. Default: folder terlama yang tidak sedang dipakai proses. Jika tidak ada yang aman dihapus, jangan membuat folder ketiga.
- Simpan APK/AAB, log dan bukti pengujian, konfigurasi, serta snapshot sumber di luar folder yang akan dihapus. Cocokkan SHA-256 arsip dengan sumber sebelum penghapusan. Cache dependency/Gradle/CMake dan hasil kompilasi perantara boleh dibuat ulang.
- Jangan menghapus folder yang dipakai build/emulator/proses lain. Hentikan proses yang dimiliki tugas secara tertib jika diperlukan. Jangan menghapus worktree terdaftar melalui penghapusan direktori biasa.
- Nama folder saja bukan izin menghapus: periksa marker kepemilikan, path absolut, batas direktori, dan reparse point/junction sebelum penghapusan rekursif.
- **Dilindungi dan tidak dihitung sebagai folder build:** repository utama, `C:\bm-datasets`, `C:\bm-envs`, `C:\bm-symbols-20260908`, checkpoint, dataset, lingkungan training, dan arsip bukti. Pertahankan semua indeks `training/checkpoints/retained-*.json` beserta aset yang dirujuknya.
- Jangan mengubah arsip bukti lama untuk memperbarui path. Tambahkan indeks relokasi/arsip baru agar hash dan riwayat lama tetap dapat diperiksa.

Lihat [prosedur dan perintah](docs/maintenance/BUILD-FOLDERS.md).
