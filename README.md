# BrailleMe

[![Platform: Android](https://img.shields.io/badge/platform-Android-3DDC84?logo=android&logoColor=white)](https://developer.android.com/)
[![Expo SDK 57](https://img.shields.io/badge/Expo%20SDK-57-000020?logo=expo&logoColor=white)](https://docs.expo.dev/)
[![Status: Technical Alpha](https://img.shields.io/badge/status-technical%20alpha-F59E0B)](#status-proyek)
[![Quality](https://github.com/rifhrzi/brailleme/actions/workflows/quality.yml/badge.svg)](https://github.com/rifhrzi/brailleme/actions/workflows/quality.yml)

BrailleMe adalah aplikasi Android eksperimental untuk mengenali **satu sel Braille A-Z** melalui kamera dan menyampaikan kandidat huruf dalam bentuk teks, suara lokal, serta umpan balik haptik. Aplikasi dibangun dengan Expo, React Native, TypeScript, VisionCamera, dan TensorFlow Lite.

> [!IMPORTANT]
> BrailleMe saat ini merupakan **technical alpha untuk pengujian internal**. Model `0.2.0-candidate.1` masih berstatus `unverified_candidate`. Aplikasi belum membaca kata, spasi, baris, atau halaman Braille dan belum boleh dianggap sebagai penerjemah Braille siap produksi.

## Daftar isi

- [Status proyek](#status-proyek)
- [Fitur yang tersedia](#fitur-yang-tersedia)
- [Cara kerja](#cara-kerja)
- [Teknologi](#teknologi)
- [Menjalankan proyek](#menjalankan-proyek)
- [Menggunakan pemindai](#menggunakan-pemindai)
- [Perintah penting](#perintah-penting)
- [Build Android dan EAS](#build-android-dan-eas)
- [Model dan batas penggunaan](#model-dan-batas-penggunaan)
- [Aksesibilitas dan privasi](#aksesibilitas-dan-privasi)
- [Struktur repository](#struktur-repository)
- [Quality gate](#quality-gate)
- [Troubleshooting](#troubleshooting)
- [Dokumentasi lanjutan](#dokumentasi-lanjutan)

## Status proyek

| Area | Status saat ini |
| --- | --- |
| Platform | Android 10+ / min SDK 29 |
| Fungsi utama | Klasifikasi satu crop sel Braille A-Z |
| Model aktif | `0.2.0-candidate.1` (`unverified_candidate`) |
| Pengujian internal | Tersedia melalui development/preview build |
| Rilis produksi biasa | Diblokir oleh release gate sampai model terverifikasi |
| Rilis eksperimental | Tersedia sebagai jalur pengecualian eksplisit, bukan bukti production GO |
| Validasi fisik | Belum lengkap; holdout independen, corpus negatif, native parity, dan benchmark perangkat masih wajib |

Kode aplikasi, pipeline kamera, pengujian otomatis, serta gate build sudah tersedia untuk eksperimen terkontrol. Keputusan produksi tetap **NO-GO** sampai seluruh bukti pada [GO checklist](docs/release/GO-CHECKLIST.md) terpenuhi untuk commit, model, perangkat, dan artefak rilis yang sama.

## Fitur yang tersedia

- Onboarding dan penjelasan batas penggunaan sebelum pemindaian.
- Pemeriksaan kesiapan kamera, model, izin, dan suara lokal.
- Pemindai kamera untuk satu sel Braille di area bidik tengah.
- Quality gate untuk menolak frame terlalu gelap, terlalu terang, atau kurang kontras.
- Validasi confidence dan margin serta stabilisasi prediksi antarsampel.
- Status abstain seperti “belum dapat dipastikan” ketika input belum layak.
- Pembacaan kandidat menggunakan Text-to-Speech lokal Android.
- Pengaturan kecepatan/volume suara, mute, dan haptik yang disimpan hanya di perangkat.
- Semantik aksesibilitas, live-region announcement, dan target sentuh minimal 48 dp pada kontrol utama.
- Halaman bantuan, privasi, pengaturan, model probe, dan diagnostik pengembangan.
- Release gate fail-closed untuk mencegah model kandidat masuk jalur produksi biasa.

## Cara kerja

```text
Kamera 640x480
      ↓
Area bidik satu sel di tengah frame
      ↓
Quality gate luminance dan kontras
      ↓
Resize 28x28, RGB float32, normalisasi 0..1
      ↓
TensorFlow Lite, 26 output A-Z
      ↓
Validasi confidence/margin dan stabilisasi temporal
      ↓
Kandidat huruf, teks status, suara lokal, dan haptik
```

Inferensi dibatasi pada interval 200 ms atau maksimal 5 Hz. Hasil dari sesi kamera lama ditolak menggunakan generation ID, sedangkan kandidat stabil dihapus ketika jeda prediksi melebihi 750 ms atau kualitas input berubah. Kamera dan polling juga dihentikan ketika layar tidak aktif, aplikasi masuk background, atau pemindaian dijeda.

## Teknologi

- Expo SDK 57 dan Expo Router
- React 19 dan React Native 0.86
- TypeScript 6
- React Native VisionCamera
- `react-native-fast-tflite` dan Nitro Modules
- Expo Speech, Expo Haptics, dan modul lokal `braille-local-voice`
- Jest dan Testing Library React Native
- EAS Build untuk development, preview, dan AAB store

## Menjalankan proyek

### Prasyarat

- Node.js 22 dan npm.
- Android Studio, Android SDK 36, dan Build Tools 36.
- JDK 17 yang tersedia melalui `JAVA_HOME`.
- Emulator Android atau perangkat fisik dengan USB debugging.

Scanner memakai modul native, sehingga aplikasi **tidak dapat dijalankan dengan Expo Go**. Gunakan development build.

### Instalasi

Clone repository dan masuk ke folder proyek:

```powershell
git clone https://github.com/rifhrzi/brailleme.git
cd brailleme
```

Pasang dependency secara konsisten dari lockfile dan verifikasi aset model:

```powershell
npm.cmd ci
npm.cmd run model:verify
```

Pastikan sesi PowerShell menggunakan JDK 17. Ganti contoh path berikut sesuai instalasi lokal:

```powershell
$env:JAVA_HOME = 'C:\path\to\jdk-17'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
java -version
```

Bangun dan pasang development client:

```powershell
npm.cmd run android
```

Setelah native binary terpasang, pengembangan JavaScript berikutnya cukup memakai Metro:

```powershell
npm.cmd run start
```

Jalankan `npm.cmd run android` lagi ketika dependency native atau konfigurasi native berubah.

## Menggunakan pemindai

1. Buka BrailleMe dari development client yang terhubung ke Metro.
2. Selesaikan onboarding dan pemeriksaan kesiapan.
3. Dari Beranda, pilih **Pindai satu sel** lalu berikan izin kamera.
4. Tempatkan tepat satu sel Braille di dalam kotak bidik, dengan kamera sejajar dan cahaya yang cukup.
5. Tunggu sampai kandidat stabil. Gunakan tombol baca jika ingin mendengarkan hasil melalui suara lokal.
6. Verifikasi kandidat terhadap sumber Braille asli; hasil technical alpha dapat salah.

Untuk pengukuran akurasi, jangan memakai contoh yang sudah pernah masuk training atau tuning. Ikuti [panduan physical validation](training/physical-validation/README.md).

## Perintah penting

| Perintah | Fungsi |
| --- | --- |
| `npm.cmd run start` | Menjalankan Metro untuk development client |
| `npm.cmd run android` | Prebuild bila perlu, compile, dan memasang debug build Android |
| `npm.cmd run model:verify` | Memeriksa hash, tensor contract, label, dan evidence aset model |
| `npm.cmd run lint` | Menjalankan ESLint |
| `npm.cmd run typecheck` | Memeriksa TypeScript tanpa menghasilkan file build |
| `npm.cmd test` | Menjalankan seluruh test Jest |
| `npm.cmd run test:coverage` | Menjalankan test beserta coverage gate |
| `npm.cmd run doctor` | Menjalankan diagnostik Expo dengan kebijakan proyek |
| `npm.cmd run expo:dependencies:check` | Memeriksa kompatibilitas versi dependency Expo |
| `npm.cmd run audit:dependencies` | Memblokir vulnerability critical pada dependency produksi |
| `npm.cmd run export:android` | Mengekspor bundle JavaScript Android |
| `npm.cmd run release:config:verify` | Memverifikasi konfigurasi build dan permission release |
| `npm.cmd run release:verify` | Gate production biasa; sengaja gagal selama model belum verified |

## Build Android dan EAS

EAS CLI dipatok ke versi `22.0.0` dan `eas.json` mewajibkan source tree yang sudah di-commit.

| Profile | Output | Kegunaan | Model kandidat |
| --- | --- | --- | --- |
| `development` | APK development client | Pengembangan dengan Metro | Diizinkan setelah verifikasi aset |
| `preview` | APK standalone internal | Uji internal tanpa Metro | Diizinkan setelah verifikasi aset |
| `production` | AAB store | Jalur produksi terverifikasi | Ditolak secara fail-closed |
| `production-experimental` | AAB store | Pengecualian risiko eksplisit | Diizinkan hanya dengan risk acceptance yang cocok |

Login dan mulai build:

```powershell
npm.cmd run eas:login
npm.cmd run build:dev:android
npm.cmd run build:preview:android
npm.cmd run build:production:android
npm.cmd run build:production-experimental:android
```

Profile `production-experimental` tidak mengubah status model menjadi verified dan tidak boleh disebut sebagai production GO. Submission memakai build ID yang spesifik, memeriksa project/profile/commit/hash, dan tetap dibuat dalam status draft:

```powershell
$env:EAS_BUILD_ID = 'id-build-yang-selesai'
npm.cmd run submit:internal:android
npm.cmd run submit:production:android
npm.cmd run submit:production-experimental:android
```

## Model dan batas penggunaan

Model aktif berada di [`assets/models/braille-cell-classifier-v1`](assets/models/braille-cell-classifier-v1). Kontrak runtime-nya adalah:

| Properti | Nilai |
| --- | --- |
| Input | `1 x 28 x 28 x 3` |
| Layout | NHWC |
| Tipe | `float32` |
| Channel | RGB |
| Normalisasi | `pixel / 255.0` |
| Output | 26 probabilitas Softmax, label `a` sampai `z` |
| Target | Satu crop sel Braille yang sudah dilokalisasi dan disegmentasi |

Model **tidak ditujukan** untuk:

- seluruh frame kamera tanpa framing;
- lebih dari satu sel Braille;
- baris atau halaman yang belum disegmentasi;
- spasi, angka, tanda baca, background, atau kelas unknown;
- keputusan medis, hukum, finansial, ujian, atau keselamatan.

Training run kandidat mencatat akurasi diagnostik 95,51%, tetapi dataset development hanya memiliki satu source family per kelas. Angka tersebut optimistis dan **bukan bukti generalisasi atau akurasi produk**. Golden test 26/26 hanya membuktikan parity konversi Keras ke TFLite.

Rilis model masih diblokir karena belum tersedia:

- physical holdout independen minimal 260 sampel;
- corpus negatif/non-Braille yang memenuhi acceptance criteria;
- parity A-Z dari jalur kamera native;
- benchmark latency, FPS, soak, dan ground truth pada minimal dua perangkat fisik;
- validasi TalkBack dan recovery nonvisual pada perangkat nyata.

Detail provenance dan hasil evaluasi tersedia di [manifest model](assets/models/braille-cell-classifier-v1/manifest.json) serta [evidence training](training/evidence/braille-cell-classifier-v1/evidence-manifest.json).

## Aksesibilitas dan privasi

Implementasi saat ini menyediakan label dan hint aksesibilitas, live-region announcement, status yang tidak bergantung pada warna saja, kontrol dengan target minimal 48 dp, Text-to-Speech lokal, haptik opsional, serta recovery ketika izin atau runtime tidak tersedia.

Fitur tersebut tetap harus divalidasi dengan TalkBack pada perangkat fisik, termasuk urutan fokus, font 200%, perubahan status dinamis, interupsi TTS, background/lock/reboot, dan pemulihan tanpa bantuan visual.

Untuk build standalone `preview` dan `production`, konfigurasi memblokir permission jaringan, storage lama, overlay, dan phone state yang tidak diperlukan serta mengatur `allowBackup=false`. Development client tetap membutuhkan jaringan untuk Metro dan membawa permission tooling, sehingga tidak dapat dijadikan bukti privasi/offline rilis.

## Struktur repository

```text
brailleme/
├── app/                         # Route Expo Router
├── src/
│   ├── components/              # Komponen UI dan aksesibilitas bersama
│   ├── config/                  # Environment dan feature flags
│   ├── features/                # Layar dan alur fitur
│   ├── ml/                      # Model contract, preprocessing, dan stabilizer
│   ├── services/                # Port kamera, runtime, izin, dan TTS
│   ├── storage/                 # Preferensi lokal
│   ├── theme/                   # Token desain
│   └── types/                   # Tipe domain
├── assets/models/               # Artefak TFLite, label, dan manifest
├── modules/braille-local-voice/ # Expo native module untuk voice lokal Android
├── training/                    # Notebook, evidence, dan physical validation pack
├── scripts/                     # Verifier build, model, EAS, dan submission
├── docs/release/                # Checklist dan dokumen keputusan rilis
├── artifacts/                   # Laporan audit historis; binary diabaikan Git
├── BrailleMe/                   # PRD dan roadmap
└── .github/workflows/           # Continuous integration
```

Folder native `android/` dan `ios/`, output build, APK/AAB, coverage, serta file environment lokal tidak disimpan di Git.

## Quality gate

Workflow [`.github/workflows/quality.yml`](.github/workflows/quality.yml) berjalan pada push dan pull request untuk:

1. memasang dependency dengan `npm ci`;
2. memeriksa dependency Expo dan Expo Doctor;
3. memverifikasi konfigurasi release serta aset model;
4. mengaudit dependency produksi pada level critical;
5. menjalankan lint, typecheck, dan test dengan coverage;
6. mengekspor bundle Android dalam environment production.

Ambang coverage global adalah 70% untuk statements, branches, functions, dan lines. Lulusnya source gate dan export JavaScript tidak otomatis membuktikan kesiapan produksi; AAB store-signed, provenance, validasi fisik, serta pengujian aksesibilitas tetap dinilai terpisah.

Jalankan pemeriksaan utama secara lokal:

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

## Troubleshooting

### `build.ninja still dirty` atau error CMake/AGP

OneDrive dan path Windows yang panjang dapat mengganggu native build. Gunakan JDK 17 dan jalankan build dari salinan kerja ber-path pendek, misalnya `C:\bm`. Source utama tetap dapat dipelihara di repository ini.

### Aplikasi meminta Metro setelah APK dipasang

APK debug dari `npm.cmd run android` adalah development client. Jalankan `npm.cmd run start` lalu hubungkan aplikasi ke Metro. Gunakan profile EAS `preview` jika membutuhkan APK internal standalone.

### `release:verify` gagal pada status model

Itu perilaku yang diharapkan selama manifest berstatus `unverified_candidate`. Jangan mengubah flag menjadi verified tanpa evidence yang diwajibkan oleh [GO checklist](docs/release/GO-CHECKLIST.md).

### PowerShell memblokir `npm.ps1`

Gunakan `npm.cmd`, seperti seluruh contoh perintah dalam README ini.

## Dokumentasi lanjutan

- [Product requirements](BrailleMe/PRD%20-%20BrailleMe.md)
- [Roadmap technical alpha](BrailleMe/Roadmap%20-%20BrailleMe%20-%20Technical%20Alpha.canvas)
- [Production GO checklist](docs/release/GO-CHECKLIST.md)
- [Dependency risk review](docs/release/DEPENDENCY-RISK-REVIEW.md)
- [Physical validation pack](training/physical-validation/README.md)
- [Audit kesiapan internal 20 Agustus 2026](artifacts/readiness-audit-2026-08-20.md)

## Lisensi

Ketentuan lisensi repository tersedia pada file [LICENSE](LICENSE).
