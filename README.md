# BrailleMe

[![Platform: Android](https://img.shields.io/badge/platform-Android-3DDC84?logo=android&logoColor=white)](https://developer.android.com/)
[![Expo SDK 57](https://img.shields.io/badge/Expo%20SDK-57-000020?logo=expo&logoColor=white)](https://docs.expo.dev/)
[![Quality](https://github.com/rifhrzi/brailleme/actions/workflows/quality.yml/badge.svg)](https://github.com/rifhrzi/brailleme/actions/workflows/quality.yml)

BrailleMe adalah aplikasi Android offline untuk menyusun satu kata Braille dengan memindai satu sel untuk setiap huruf A-Z. Huruf yang sudah stabil dikonfirmasi pengguna, ditambahkan ke kata, lalu dapat dibacakan menggunakan suara lokal Android.

> [!IMPORTANT]
> Pipeline aplikasi, composer kata, dan release gate tersedia. Kandidat Angelina `0.3.0-candidate.1` kini aktif untuk development lokal, tetapi tetap berstatus `unverified_candidate`. Jalur produksi dan build EAS tertutup sampai hak dataset jelas serta validasi kamera fisik, holdout independen, corpus negatif, benchmark perangkat, TalkBack, dan signing release lulus pada artefak yang sama.

## Kemampuan aplikasi

- Menyusun tepat satu kata hingga 12 huruf, satu sel Braille per langkah.
- Menambahkan huruf stabil secara eksplisit sehingga huruf berulang tetap dapat dipindai.
- Memindai ulang sel, menghapus huruf terakhir, atau menghapus seluruh kata.
- Membacakan kata melalui voice Bahasa Indonesia yang tersedia secara lokal.
- Menolak frame terlalu gelap, terlalu terang, kurang kontras, atau berkeyakinan rendah.
- Menghentikan kamera, polling, dan suara saat layar tidak aktif atau aplikasi masuk background.
- Memproses semuanya di perangkat tanpa akun, analitik, telemetri, penyimpanan hasil, atau unggahan gambar.
- Menyediakan semantik TalkBack, pengumuman status, reduce motion, haptik opsional, dan target sentuh minimal 48 dp.

BrailleMe tidak memotret seluruh kata atau baris sekaligus. Model tetap menerima satu crop sel `28 x 28 x 3` dan menghasilkan satu dari 26 huruf A-Z. Spasi, angka, tanda baca, background, dan kelas unknown belum tersedia.

## Alur penggunaan

1. Selesaikan pengenalan awal, lalu buka **Pindai**.
2. Aktifkan kamera untuk kunjungan tersebut.
3. Tempatkan sel pertama di tengah bingkai dan tunggu sampai huruf stabil.
4. Cocokkan huruf dengan sumber Braille, lalu pilih **Tambah huruf**.
5. Ulangi untuk sel berikutnya. Gunakan **Pindai ulang sel** atau **Hapus huruf terakhir** bila perlu.
6. Setelah kata selesai, pilih **Bacakan** dan verifikasi kembali kata sebelum digunakan.

## Cara kerja

```text
Kamera 640 x 480
  -> crop tengah satu sel
  -> quality gate luminance dan kontras
  -> resize 28 x 28, RGB float32, normalisasi 0..1
  -> TensorFlow Lite, 26 output A-Z
  -> confidence, margin, dan stabilisasi temporal
  -> konfirmasi pengguna per scanEpoch
  -> composer satu kata dan TTS lokal
```

Inferensi dibatasi maksimal 5 Hz. `sessionId` menolak hasil dari sesi kamera lama, sedangkan `scanEpoch` menolak hasil terlambat setelah huruf ditambahkan atau sel dipindai ulang. Draft kata bertahan saat recovery kamera, tetapi kandidat yang belum dikonfirmasi selalu diinvalidasi.

## Menjalankan proyek

Scanner memakai modul native, sehingga Expo Go tidak mencakup runtime lengkapnya. Gunakan Node.js 22, Android SDK 36, dan JDK 17.

```powershell
npm.cmd ci
npm.cmd run model:verify
npm.cmd run android
npm.cmd run start
```

Jalankan ulang `npm.cmd run android` setelah dependency atau konfigurasi native berubah. Pada Windows, build native lebih stabil dari path pendek non-OneDrive, misalnya `C:\bm`.

## Pemeriksaan kualitas

```powershell
npm.cmd run expo:dependencies:check
npm.cmd run doctor
npm.cmd run release:config:verify
npm.cmd run audit:dependencies
npm.cmd run model:verify
npm.cmd run lint
npm.cmd run typecheck
npm.cmd run test:coverage
$env:APP_ENV = 'production'
npm.cmd run export:android
```

Perintah penting:

| Perintah | Fungsi |
| --- | --- |
| `npm.cmd run model:verify` | Memeriksa aset, hash, tensor contract, label, dan evidence model |
| `npm.cmd run release:config:verify` | Memeriksa konfigurasi, permission, ikon, dan profile release |
| `npm.cmd run release:verify` | Gate produksi fail-closed; harus menolak model yang belum terverifikasi |
| `npm.cmd run test:coverage` | Menjalankan seluruh test dan coverage gate |
| `npm.cmd run export:android` | Mengekspor bundle JavaScript untuk Android |
| `npm.cmd run build:preview:android` | Ditolak selama hak redistribusi dataset kandidat belum dikonfirmasi |
| `npm.cmd run build:production:android` | Membuat AAB hanya setelah seluruh production gate lulus |

## Model dan batas keselamatan

Model aktif berada di [`assets/models/braille-cell-classifier-v2`](assets/models/braille-cell-classifier-v2); v1 tetap disimpan sebagai rollback historis:

| Properti | Nilai |
| --- | --- |
| Input | `1 x 28 x 28 x 3`, NHWC, `float32`, RGB |
| Normalisasi | `pixel / 255.0` |
| Output | 26 probabilitas Softmax, label `a` sampai `z` |
| Target | Satu crop sel Braille yang sudah dilokalisasi |
| Runtime kata | Akumulasi hasil satu sel yang dikonfirmasi pengguna |
| External test Angelina | 93,11% top-1 pada 9.625 annotation crop; bukan akurasi kamera |
| Native development | Post-crop BGRA -> RGB `/255` -> TFLite lulus di emulator; kamera fisik belum diuji |

Status `unverified_candidate` tidak boleh diubah hanya agar build lolos. Production GO memerlukan sekurangnya:

- parity A-Z dari jalur kamera native;
- hak penggunaan dan redistribusi seluruh sumber dataset;
- physical holdout independen dan source-independent;
- corpus negatif/non-Braille dengan acceptance criteria yang dibekukan;
- benchmark latency, preview FPS, soak, dan ground truth pada perangkat fisik;
- validasi kata end-to-end termasuk huruf berulang dan koreksi;
- TalkBack, font 200%, landscape, background/lock/reboot, dan recovery nonvisual;
- commit bersih, EAS project identity, AAB store-signed, dan provenance signing.

Hasil tidak boleh menjadi satu-satunya dasar keputusan medis, hukum, finansial, ujian, atau keselamatan.

## Aksesibilitas dan privasi

UI memakai heading semantik, label/hint yang natural, status yang tidak bergantung pada warna saja, kontrol minimal 48 dp, fokus standar tab, TTS lokal, haptik opsional, dan pengumuman yang dibatasi agar TalkBack tidak terus terpotong.

Paket rilis memblokir izin jaringan dan penyimpanan yang tidak diperlukan serta menetapkan `allowBackup=false`. Gambar, huruf, dan kata tidak disimpan. Hanya preferensi suara, getaran, dan status pengenalan awal yang disimpan secara lokal.

## Struktur repository

```text
brailleme/
|- app/                         # Route Expo Router
|- src/
|  |- accessibility/           # Preferensi sistem aksesibilitas
|  |- components/              # Komponen UI bersama
|  |- features/                # Beranda, Live, onboarding, bantuan, privasi, pengaturan
|  |- ml/                      # Model contract, preprocessing, dan stabilizer
|  |- services/                # Port kamera, permission, readiness, dan TTS
|  |- storage/                 # Preferensi lokal dan migrasi schema
|  |- theme/                   # Token desain
|  `- types/                   # Kontrak domain
|- assets/models/              # Model aktif, label, dan manifest
|- modules/braille-local-voice/# Modul voice lokal Android
|- training/                   # Provenance, evidence, dan physical validation pack
|- scripts/                    # Verifier build, model, EAS, dan submission
|- docs/release/               # Checklist serta keputusan rilis
`- .github/workflows/          # Continuous integration
```

Folder native hasil prebuild, dependency lokal, cache Expo, coverage, export, APK/AAB, dan output klien diabaikan Git dan bukan bagian source release.

## Dokumen rilis

- [Production GO checklist](docs/release/GO-CHECKLIST.md)
- [Dependency risk review](docs/release/DEPENDENCY-RISK-REVIEW.md)
- [Play Store listing](docs/release/PLAY-STORE-LISTING-ID.md)
- [Privacy policy](docs/release/PRIVACY-POLICY-ID.md)
- [Physical validation pack](training/physical-validation/README.md)
- [Model evidence manifest](training/evidence/braille-cell-classifier-v2/evidence-manifest.json)
- [Angelina external benchmark dan eksperimen](training/experiments/angelina-2026-09-04/README.md)

## Lisensi

Ketentuan lisensi repository tersedia pada [LICENSE](LICENSE).
