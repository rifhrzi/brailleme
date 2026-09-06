# Retensi model dan status pembersihan, 6 September 2026

**Retensi selesai; penghapusan lokal belum dilakukan.** Pemeriksaan otomatis
menolak perintah `Remove-Item` untuk rencana pembersihan, kemudian juga menolak
percobaan yang dibatasi ke satu path eksplisit `run1`. Alasan yang dikembalikan
adalah `blocked by policy`, tanpa rincian tambahan. Tidak ada klaim ruang disk
telah dibebaskan. File lama tetap diabaikan Git dan tidak ikut diunggah.

## Hasil yang dipertahankan

- [Checkpoint terbaik pembaca kata](../../training/checkpoints/braille-word-reader-v1/best-word.keras):
  run20 epoch 09, identik dengan v15/v26, SHA-256
  `cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7`.
- TFLite aktif, fixture startup, manifest, threshold, dan laporan evaluasi
  `experimental-internal-best` tetap utuh. Bobot terbaik memakai decoder v2:
  323/607 kata utuh benar (53,21%); 323/351 hasil diterima benar (92,02%).
- Checkpoint classifier satu sel v2 disimpan terpisah sebagai baseline jalur
  classifier, bukan pembanding angka pembacaan kata.
- Konfigurasi, riwayat, protokol, sumber asli run20, dan catatan dependensi
  tersedia di [training/checkpoints](../../training/checkpoints/README.md).
- Catatan eksperimen lokal dipadatkan dalam
  [arsip historis](../../training/evidence/braille-word-reader-v1/experiment-notes-20260906.zip).
  Semua isi arsip dibandingkan dengan byte sumber; dependency dan cache tensor
  dikecualikan. Bukti evaluasi/diagnosis yang sudah dibekukan tetap dipertahankan.
- Pixel cache training yang masih dibutuhkan, dataset asli di `C:\bm-datasets`,
  lingkungan Python, dependency aplikasi, konfigurasi editor, invoice,
  presentasi, dan APK internal v0.2.0 tanggal 6 September tetap disimpan lokal.

## Rencana penghapusan yang belum terlaksana

Inventaris sebelum penghapusan: **15.094 file, sekitar 9,866 GiB**, dalam 403
target lokal. Rencana lengkap tersedia di [cleanup-plan-20260906.json](cleanup-plan-20260906.json).
Angka tersebut adalah ukuran logis inventaris, bukan ruang disk yang dibebaskan.

- Checkpoint run dan selected selain salinan terbaik yang disimpan di Git.
- Cache tensor, model ekspor duplikat, probe dependency, dan file kerja
  eksperimen yang catatannya sudah masuk arsip.
- APK v0.1.0 tanggal 27 Agustus dan 4 September beserta sidecar hash-nya.
- Hasil export, coverage, state Expo, Android hasil prebuild, dan cache Python.
- Output diagnostik kata lama; file presentasi dan invoice dikecualikan.

Path dalam rencana adalah inventaris pada tanggal ini. Periksa ulang sebelum
penghapusan manual karena file dapat berubah atau dibuat kembali oleh pengujian.
Jangan menghapus seluruh `.braille-training-work`: data yang dipakai run20 dan
lingkungan `test-env` masih berada di dalamnya. Penghapusan di luar workspace
tidak termasuk rencana.

## Verifikasi retensi

- Semua 25 file checkpoint/provenance dan arsip catatan cocok dengan hash
  di [retained-models.json](../../training/checkpoints/retained-models.json).
- Sepuluh snapshot skrip run20 cocok dengan hash konfigurasi asal.
- Seluruh delapan manifest data run20 tersedia sebagai gzip; hash arsip dan
  hash JSON hasil dekompresi cocok dengan konfigurasi training.
- Checkpoint kata berhasil dimuat dengan Keras 3.15.1/Torch, mempunyai 829.307
  parameter, menerima `[1,192,480,3]`, dan menghasilkan `[1,48,120,11]` dengan
  semua nilai finite. Tidak ada training tambahan atau pengukuran ulang akurasi.
- Verifikasi aset model lulus; 10 regresi verifikasi eksperimen serta 218
  kasus gerbang aset dan 4 pemeriksaan FlatBuffer lulus.
- Lint, typecheck, 34 suite / 483 tes aplikasi, 161 tes training, dan
  pemeriksaan konfigurasi rilis lulus.
- Export bundle Android dengan lingkungan preview lulus; ini bukan build APK baru.

Hasil ini mendukung integritas source dan checkpoint. Kandidat tetap eksperimen
internal; target akurasi awal dan validasi kamera fisik belum lulus.
