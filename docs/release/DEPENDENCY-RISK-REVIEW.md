# Dependency Risk Review — 27 Agustus 2026

Status: **menunggu persetujuan release owner**. Dokumen ini bukan persetujuan otomatis.

## Hasil terkini

- `npm audit --omit=dev`: 0 critical, 0 high, 11 moderate.
- `npm audit`: 0 critical, 4 high, 19 moderate, 1 low.
- High pada jalur production telah ditutup dengan pin `metro`, `metro-config`,
  dan `metro-transform-worker` 0.84.5.
- Empat high yang tersisa berada pada dependency development `eas-cli` dan
  transitifnya (`minimatch`, `nanoid`, `tar`).

## Analisis keterjangkauan

Advisori production yang tersisa berada pada Expo CLI/config/plugin/xcode yang
dipakai saat install, config, bundling, atau prebuild. Modul tersebut tidak
menjadi bagian dari runtime JavaScript yang diekspor ke aplikasi Android.
BrailleMe juga memblokir izin jaringan pada build preview dan production.

EAS CLI tetap merupakan permukaan build yang sensitif. Mitigasi yang aktif:

- versi EAS CLI dan integritas package dikunci di `package-lock.json`;
- workflow memakai `npm ci` dan action GitHub pada full commit SHA;
- kredensial checkout tidak dipertahankan;
- input build berasal dari repository yang harus bersih dan sudah di-commit;
- submission memakai build ID spesifik, project/commit/profile yang cocok,
  dan mencatat SHA-256 artefak;
- `npm audit fix --force` tidak digunakan karena saat ini menawarkan downgrade
  Expo/EAS yang breaking dan tidak kompatibel dengan aplikasi.

## Keputusan yang diperlukan

Release owner harus memilih salah satu sebelum GO:

1. menyetujui risiko tooling yang tersisa dengan nama, tanggal, dan tanggal
   kedaluwarsa di bawah; atau
2. menunda release sampai versi Expo/EAS yang kompatibel menutup advisori.

- Keputusan: **belum disetujui; status produksi tetap NO-GO**
- Disetujui oleh: **belum ada release owner yang tercatat**
- Tanggal: **belum berlaku**
- Kedaluwarsa/review ulang paling lambat: `2026-09-27`
- Catatan: audit terakhir tetap menemukan 11 moderate pada jalur production dan 4 high pada tooling development; persetujuan tertulis serta audit ulang exact-build masih wajib.

Audit harus dijalankan ulang setelah setiap perubahan lockfile dan tepat sebelum
build kandidat final.
