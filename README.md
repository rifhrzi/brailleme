# BrailleMe

[![Platform: Android](https://img.shields.io/badge/platform-Android-3DDC84?logo=android&logoColor=white)](https://developer.android.com/)
[![Expo SDK 57](https://img.shields.io/badge/Expo%20SDK-57-000020?logo=expo&logoColor=white)](https://docs.expo.dev/)
[![Quality](https://github.com/rifhrzi/brailleme/actions/workflows/quality.yml/badge.svg)](https://github.com/rifhrzi/brailleme/actions/workflows/quality.yml)

BrailleMe adalah aplikasi Android offline untuk memindai satu kata atau rangkaian angka Braille Indonesia dasar. Bingkai mengikuti sel terdekat tengah kamera; pengguna menekan **Pindai** untuk memperoleh hasil. Decoder mempertahankan huruf kecil, kapital, dan nol awal angka. Satu hasil memuat 1–12 karakter dari maksimal 24 sel fisik, termasuk indikator, dan dapat dibacakan menggunakan suara lokal Android.

> [!IMPORTANT]
> Update 0.3.0 tetap eksperimen internal. Aturan simbol yang lulus tes tidak membuktikan model kamera mampu mengenali setiap indikator. Target setiap kelompok adalah 75% hasil utuh dan 95% ketepatan hasil diterima. Bukti dan checkpoint sebelum update tetap disimpan: decoder v2 mencapai **323/607 kata utuh (53,21%)**, **323/351 hasil diterima benar (92,02%)**. Evaluasi baru memakai decoder Indonesia dan memisahkan foto publik dari susunan sintetis. Produksi dan cloud build tetap ditutup; pengujian emulator tidak membuktikan akurasi kertas timbul fisik.

Checkpoint untuk melanjutkan training, konfigurasi asli, hash, dan panduan data tersedia di [training/checkpoints](training/checkpoints/README.md). [Catatan pembersihan](docs/maintenance/2026-09-06-repository-cleanup.md) menjelaskan artefak yang dipertahankan dan status penghapusan file sementara.

## Kemampuan aplikasi

- Tab **Penerjemah** mengubah ketikan menjadi gambar titik Braille dan susunan enam titik menjadi teks, langsung dan tanpa internet. Mendukung huruf kecil/kapital, angka bulat terpisah dari huruf, spasi, baris baru, serta `. , ? ! : ; - ( )`. Batas 200 karakter/400 sel; masukan yang belum lengkap tetap tersedia untuk diperbaiki.
- Dalam **Braille ke Teks**, pilih sel lalu ubah tombol titik `1 4 / 2 5 / 3 6`. Tambahkan sel, spasi, atau baris baru setelah pilihan. Draf kedua mode tersimpan di memori selama aplikasi berjalan. Fitur ini menggunakan aturan terpisah dari model kamera.
- Memotret satu kata atau rangkaian digit hingga 12 karakter setelah **Pindai** ditekan.
- Mengikuti posisi dan ukuran satu kelompok Braille secara otomatis, dengan bingkai luar dan kotak setiap sel.
- Menampilkan foto yang dibekukan, kotak sel, dan hasil kata; huruf berulang pada posisi berbeda tetap dipertahankan.
- Mengganti hasil secara utuh melalui **Pindai ulang**.
- Membacakan kata melalui voice Bahasa Indonesia yang tersedia secara lokal.
- Memahami huruf tanpa indikator sebagai huruf kecil, kapital tunggal untuk huruf berikutnya, kapital ganda di awal kata, dan tanda angka untuk digit 0–9. `001` disimpan sebagai teks.
- Menolak foto terlalu gelap, terlalu terang, kurang kontras, atau sel meragukan; menolak indikator menggantung, susunan tidak didukung, batas terpotong, dan celah mencurigakan.
- Menghentikan kamera dan suara serta melepas foto saat layar tidak aktif atau aplikasi masuk background.
- Memproses semuanya di perangkat tanpa akun, analitik jaringan, penyimpanan hasil, atau unggahan gambar. APK preview mencatat metrik kinerja agregat lokal tanpa isi hasil atau piksel.
- Menyediakan semantik TalkBack, pengumuman status, reduce motion, haptik opsional, dan target sentuh minimal 48 dp.

Lingkup pemindaian adalah satu kata atau bilangan bulat tanpa tanda dalam satu baris. Kalimat, halaman, singkatan Braille, desimal, angka negatif, pemisah angka, tanda baca, dan campuran huruf–angka belum didukung. Kelompok tetangga dipakai untuk pemeriksaan batas, bukan digabung menjadi kalimat. Pemeriksaan geometri tidak membuktikan model sama sekali tidak melewatkan suatu sel. Tidak ada pelengkapan dengan kamus.

## Alur penggunaan

1. Selesaikan pengenalan awal, lalu buka **Pindai**.
2. Aktifkan kamera untuk kunjungan tersebut.
3. Arahkan seluruh kata atau angka ke tengah kamera, termasuk indikator dan ruang di kedua ujung.
4. Tunggu bingkai otomatis dan petunjuk **Tahan kamera, lalu tekan Pindai**. Pelacakan berjalan pada preview; hasil tetap memerlukan tombol **Pindai**.
5. Periksa foto, kotak sel, kapital, digit, dan hasil. Pilih **Bacakan** untuk hasil pengambilan terakhir yang berhasil.
6. Jika hasil ditolak atau ingin mengambil kata baru, pilih **Pindai ulang**, perbaiki posisi/cahaya/fokus, lalu ambil foto lagi.

## Cara kerja

```text
Pratinjau YUV -> deteksi geometris maksimal 5 Hz, satu pekerjaan tanpa antrean
  -> kandidat tengah, dua pengamatan, bingkai EMA 0.35
  -> tombol Pindai mengunci koordinat sensor / sesi / waktu
  -> foto dalam memori, target 1920 x 1440, prioritas kualitas
  -> transformasi native frame -> sensor -> preview / foto tegak
  -> deteksi ulang target dalam konteks 2.5:1, maksimal dua perluasan
  -> crop/resize native 22-bit 480 x 192, RGB float32 /255, runtime latar
  -> MobileNetV2 alpha .5 + fitur beberapa skala, grid stride 4
  -> keberadaan sel + kotak + enam probabilitas titik
  -> NMS posisi -> periksa seluruh kelompok sel -> urutkan kiri ke kanan
  -> aturan indikator Indonesia -> hasil utuh atau penolakan -> TTS lokal
```

Event versi 4 membawa `sessionId`, `captureId`, status, teks, dan sel berurutan dengan pola, posisi, confidence, peran, serta karakter keluaran. Indikator mengeluarkan string kosong; validator merekonstruksi teks dari seluruh sel. Tracker memakai interpreter terpisah dengan bobot yang sama. Pengamatan lebih tua dari 500 ms menonaktifkan Pindai dan bingkai hilang setelah 800 ms. Hasil lama diabaikan setelah pindai ulang, pergantian sesi, orientasi/layout, atau background. Foto/model dilepas setelah pekerjaan yang memakainya selesai.

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
$env:APP_ENV = 'preview'
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

Kontrak pengenal kata berada di [`assets/models/braille-word-reader-v1`](assets/models/braille-word-reader-v1). Classifier `braille-cell-classifier-v2` dipertahankan untuk pembanding diagnostik, dan dapat diverifikasi terpisah dengan `npm.cmd run model:verify:classifier`:

| Properti | Nilai |
| --- | --- |
| Input | `1 x 192 x 480 x 3`, NHWC, `float32`, RGB |
| Normalisasi | `pixel / 255.0`, kemudian `Rescaling(2,-1)` di dalam model |
| Output | `1 x 48 x 120 x 11`: objectness, offset x/y, ukuran kotak, titik 1–6 |
| Arsitektur | MobileNetV2 alpha 0.5, bobot awal ImageNet, gabungan fitur stride 4/8/16 |
| Target | Enam titik sel; penerjemahan urutan memakai aturan Braille Indonesia dasar |
| Gate APK internal | Kata utuh ≥75% termasuk penolakan sebagai gagal; ketepatan yang diterima ≥95%; nol hasil diterima pada 44 gambar negatif publik |
| Pemilihan model | Checkpoint dan ambang dipilih pada validation yang terpisah menurut sumber gambar |
| Bukti fisik | Dilaporkan terpisah; benchmark publik bukan holdout kamera fisik |

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
|  |- ml/                      # Kontrak kata, preprocessing foto, decoder, loader; classifier diagnostik
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
- [Panduan pengujian foto satu kata](docs/testing/WORD-SCAN-SELF-TEST-ID.md)
- [Model evidence manifest](training/evidence/braille-cell-classifier-v2/evidence-manifest.json)
- [Angelina external benchmark dan eksperimen](training/experiments/angelina-2026-09-04/README.md)

## Lisensi

Ketentuan lisensi repository tersedia pada [LICENSE](LICENSE).
