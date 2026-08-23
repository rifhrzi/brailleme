# Audit bug, performa, dan kesiapan uji BrailleMe

Tanggal verifikasi: 20 Agustus 2026

## Keputusan

- **GO untuk pengujian internal terkontrol** menggunakan APK preview yang disediakan dalam laporan ini.
- **NO-GO untuk produksi dan klaim akurasi final.** Model masih berstatus `unverified_candidate`, sehingga aplikasi sengaja menolak gate produksi.
- Fitur yang tersedia saat ini adalah **klasifikasi satu sel Braille A–Z yang sudah berada di area bidik**, bukan penerjemahan kata, spasi, baris, atau halaman Braille.

Keputusan ini membedakan kesiapan software untuk diuji langsung dari validitas model untuk dipakai sebagai alat bantu yang dipercaya. Seluruh hasil harus tetap diverifikasi manusia selama fase uji.

## Perbaikan yang diterapkan

### Kamera dan pipeline inferensi

- Mengubah timeout pencarian kamera menjadi status lambat nonfatal. Kamera yang ditemukan setelah deadline tidak lagi terkunci pada layar “pemindaian bermasalah”.
- Menambahkan generation ID pada hasil frame agar hasil lama yang selesai setelah Bersihkan/reset tidak dapat menghidupkan kembali kandidat sebelumnya.
- Membuat error fatal worklet bertahan sampai dibaca, sehingga tidak dapat tertimpa hasil inferensi berikutnya.
- Memperbaiki urutan dan isolasi cleanup frame/crop/resize agar setiap resource tetap dicoba untuk dilepas walau salah satu `dispose()` gagal.
- Menjalankan polling hasil hanya ketika kamera benar-benar aktif; polling berhenti saat jeda, aplikasi masuk background, route kehilangan fokus, atau error fatal.
- Menurunkan resolusi analisis dari 1280×720 menjadi 640×480. Jumlah piksel sumber per frame turun dari 921.600 menjadi 307.200, atau **66,7% lebih sedikit**, sementara tensor model tetap 28×28×3.
- Membatasi cadence inferensi/polling menjadi 200 ms (maksimal 5 Hz) dan meminta format kamera 30 fps.
- Menambahkan quality gate luminance/contrast murah sebelum TFLite. Frame terlalu gelap, terlalu terang, atau tanpa kontras memadai ditolak sebelum inferensi mahal.

### Stabilitas hasil dan kontrak model

- Kandidat stabil sekarang kedaluwarsa/reset ketika jeda antar-prediksi melebihi 750 ms, kualitas berubah, atau sesi di-reset.
- Memvalidasi confidence, margin, timestamp, index kelas, dan urutan label secara fail-closed.
- Versi runtime diturunkan dari manifest model agar tidak dapat drift dari versi artefak.
- Verifikasi aset sekarang memeriksa urutan label A–Z secara tepat, bukan hanya jumlah 26 label.
- Scene kamera yang tidak menyerupai sel Braille pada smoke test final menghasilkan status abstain “Belum dapat dipastikan”, bukan dipaksa menjadi huruf.

### Lifecycle, UI, suara, dan navigasi

- Menghentikan kamera dan kontrol aktif ketika terjadi error fatal; aksi pemulihan ditampilkan sebelum konten stale.
- Mendeduplikasi event UI secara semantik dan mememoisasi komponen kamera untuk mengurangi render seluruh layar pada setiap sampel.
- Menambahkan latest-request-wins untuk permintaan izin kamera serta pemeriksaan TTS agar respons async lama tidak menimpa status terbaru.
- Menghentikan TTS ketika aplikasi masuk background atau layar kehilangan fokus.
- Menambahkan cache/dedup pemeriksaan voice lokal beserta invalidasi generation-aware setelah kembali dari pengaturan TTS.
- Menambahkan fallback navigasi `replace('/')` ketika halaman dibuka lewat deep link tanpa history.
- Memungkinkan retry lazy import benar-benar memuat ulang modul setelah kegagalan import.
- Mengubah copy beranda agar tidak menyatakan scanner/model “terpasang dan siap” sebelum probe runtime dijalankan.

### Build dan keamanan artefak

- Menyelaraskan seluruh dependency Expo patch-level; `expo install --check` dan Expo Doctor kembali hijau.
- Build development/preview dan hook EAS nonproduksi sekarang selalu menjalankan `model:verify`; produksi menjalankan gate penuh `release:verify`.
- Memverifikasi konfigurasi preview dan production dari manifest hasil introspeksi, bukan hanya file konfigurasi sumber.
- Manifest APK standalone hanya meminta izin platform `CAMERA` dan `VIBRATE`. INTERNET, network state, storage, overlay, dan phone state tidak ada.
- `allowBackup=false` dan `debuggable=false` terkonfirmasi pada APK final.

## Hasil verifikasi final

| Pemeriksaan | Hasil |
| --- | --- |
| ESLint | Lulus |
| TypeScript `tsc --noEmit` | Lulus |
| Jest | 21 suite, 224/224 tes lulus |
| Coverage global | 78,62% statement; 75,77% branch; 78,79% function; 79,21% line |
| Coverage modul ML | 98,94% statement; 95,20% branch |
| Expo Doctor | 21/21 pemeriksaan lulus |
| `expo install --check` | Dependency sesuai |
| Verifikasi aset model | Lulus: 1×28×28×3 → 26 kelas; hash cocok |
| Verifikasi config release | Lulus untuk preview APK dan production AAB |
| Gate model produksi | Ditolak sesuai desain: `unverified_candidate` |
| Gradle `assembleRelease` | Lulus dengan Temurin 17 dari staging path pendek |
| Cold-start APK tanpa Metro | Lulus |
| Live camera >10 detik | Lulus; tidak ada false-fatal |
| Halaman pengaturan/TTS lokal | Lulus; native module tersedia |
| Crash/fatal log selama smoke test | Tidak ditemukan |

Pengukuran emulator sebelumnya menunjukkan CPU sekitar 23–26% saat pemindaian aktif dan 4–5% saat dijeda. Angka ini hanya indikator regresi pipeline, bukan benchmark perangkat fisik atau bukti thermal/ketahanan.

## Audit APK final

- File: `output/android/BrailleMe-0.1.0-internal-preview-2026-08-20.apk`
- Ukuran: 130.380.842 byte
- SHA-256: `34B4DCEDC7FD512591AD746557BE7E9291D9A3C0E8752AC0287E0518F8F70004`
- Package: `com.brailleme.app`
- Versi: `0.1.0` / version code `1`
- Android: min SDK 29, target/compile SDK 36
- ABI: `arm64-v8a` dan `x86_64`
- Zip alignment: lulus
- Signature scheme: APK Signature Scheme v2
- Model di APK: 979.488 byte, SHA-256 `096d2822321732fd8e0e46fd6ca53e8569cc533f81a5593827d439f1a51f18f5`, identik dengan source
- `BrailleLocalVoiceModule` hadir dan terdaftar pada generated Expo module package list
- Sertifikat: **Android Debug RSA 2048-bit**; hanya untuk internal testing, bukan Play Store/produksi

Instalasi melalui ADB:

```powershell
adb install -r .\output\android\BrailleMe-0.1.0-internal-preview-2026-08-20.apk
```

## Batas akurasi yang belum dapat ditutup oleh perbaikan kode

Manifest model masih tidak memiliki provenance dataset/training, channel order, value range, normalization, lisensi, atau laporan evaluasi. Flag label-order against training, preprocessing parity, golden-sample parity, dan device benchmark juga masih `false`.

Model memiliki 26 keluaran Softmax A–Z tanpa kelas background/unknown/space. Ia juga ditujukan untuk crop sel yang sudah dilokalisasi dan disegmentasi, sedangkan runtime saat ini masih memakai area bidik tengah—belum memiliki detector/grid/cell segmentation yang tervalidasi. Threshold dan quality gate mengurangi false positive dengan cara abstain, tetapi tidak membuktikan bahwa huruf yang diterima sudah benar.

Diagnostik historis proyek juga mencatat fixture berlabel A pernah menghasilkan V dengan confidence 87,74%. Karena itu gate produksi tidak boleh dibuka hanya karena tes software dan build lulus.

## Acceptance gate berikutnya

Sebelum mengubah status model menjadi `verified` atau menyatakan fitur akurat:

1. Pastikan training code, dataset version/license, urutan label, channel order, range, dan normalisasi dapat direproduksi.
2. Jalankan golden parity pada file `.tflite` aktual, bukan mock, untuk seluruh A–Z serta negative/background/unknown.
3. Ukur precision, recall, confusion matrix, false-accept rate, dan abstention coverage pada holdout yang tidak pernah masuk training.
4. Tambahkan detector/segmenter atau buktikan framing acceptance untuk crop satu sel secara terukur.
5. Uji pada beberapa perangkat fisik untuk pencahayaan, glare, blur, rotasi, jarak, material Braille, p50/p95 latency, FPS, dropped frame, memory, dan thermal soak.
6. Jalankan sesi TalkBack nyata: urutan fokus, announcement, TTS overlap, recovery nonvisual, serta target sentuh.
7. Gunakan production keystore dan provenance build. Repo saat audit belum memiliki commit sementara EAS memakai `requireCommit=true`, sehingga build EAS sengaja belum dapat dijalankan.

## Risiko dependency tersisa

`npm audit --omit=dev` melaporkan 8 high dan 8 moderate, tanpa critical. Semuanya berada pada rantai Expo/Metro/transitif; perbaikan otomatis yang ditawarkan memaksa downgrade/major change yang tidak kompatibel. Tidak dilakukan `npm audit fix --force`. Risiko ini perlu ditinjau ulang ketika Expo menerbitkan upgrade yang kompatibel.
