# APK BrailleMe dengan UI bersih dan model ID-v5

Pembaruan 11 September 2026 menyederhanakan petunjuk pada Beranda, pengenalan
awal, aktivasi kamera, pemindai, Bantuan, dan Privasi. Label tahap development
dihapus dari tampilan. Petunjuk tetap menjelaskan cara membingkai Braille,
memeriksa hasil, dan memindai ulang. Bantuan mencakup kapital, angka, serta
`. , ? ! : ; - ( )`.

Hasil dengan tanda baca dipertahankan dalam teks, draft, dan permintaan Bacakan.
Contoh `Baca?`, `ABC!`, dan `(001)` diuji melalui event berisi pola titik yang
valid. Gabungan huruf-angka, desimal, angka negatif, pasangan kurung tidak
lengkap, whitespace, dan teks di atas 12 karakter tetap ditolak.

## APK internal

[Unduh BrailleMe 0.3.0 dengan model ID-v5 (r2)](../../output/android/BrailleMe-0.3.0-model-v5-2026-09-11-r2.apk).

- Ukuran: 136.598.928 byte, sekitar 130,3 MiB.
- SHA-256: `71e3b4c69b8041860ae2054f16a5494c420e20ca4bc71536d7f4e7fae8955771`.
- Package: `com.brailleme.app`; versi `0.3.0`, version code `3`.
- Minimum Android: API 29; ABI: `arm64-v8a` dan `x86_64`.
- Penandatangan: sertifikat Android Debug, untuk pemasangan dan pengujian internal.

Seluruh 27 pemeriksaan pada [laporan integritas APK](../../output/apk-id-v5-20260911/apk-integrity-r2/report.json)
lulus. Model v5 dan fixture di dalam APK cocok dengan hash manifest. Pemeriksaan
mencakup validitas arsip dan tanda tangan, metadata package, izin kamera, ketiadaan
izin internet/storage, serta alignment ZIP dan segmen ELF 16 KiB. Hasil ini
memverifikasi artefak APK; kesiapan rilis produksi belum ditetapkan.

Build r2 selesai dalam 2 menit 39 detik; 1.064 dari 1.122 tugas Gradle memakai
hasil yang sudah tersedia. APK awal dan lognya tetap disimpan sebagai bukti
sebelum perbaikan. Gunakan tautan r2 di atas untuk pemasangan terbaru.

## Model yang terkemas

- Model: `assets/models/braille-word-reader-id-v5/word-reader.tflite`.
- SHA-256: `6847c91fe186169ba5208a7444f9c8bedc753228bc42b78c691471a85453a6cb`.
- Metode: `cascade_zoom_fallback`, melalui `word_overall_v5_zoom`.
- Akurasi benchmark TFLite terpisah: 438/500 teks utuh (87,6%).
- Paritas implementasi aplikasi: seluruh 540 contoh validation, tanpa selisih
  teks, penolakan, mask, peran simbol, atau rute; keenam transformasi piksel identik.
- Fixture awal: `Yys`, diambil dari validation yang sudah digunakan, bukan TEST.

Angka 87,6% berasal dari komposisi potongan scan publik. Akurasi kamera ponsel
fisik belum diukur. Metode terlatih beserta pembacaan ulang diperlukan untuk
mereproduksi hasil tersebut; mengganti berkas model saja tidak cukup.

## Pemeriksaan sumber

`model:verify`, `release:config:verify`, lint, typecheck, dan `test:coverage`
lulus. Sebanyak 39 suite / 874 tes lulus. Coverage: statements 91,48%, branches
86,38%, functions 90,56%, lines 92,66%. Empat peringatan urutan import terdapat
hanya pada harness pengujian paritas; lint tidak menghasilkan error.

Bukti sumber dan build berada di `output/apk-id-v5-20260911/`. Staging memakai
ulang `C:\bm-symbols-apk-20260908`, satu-satunya folder build terdaftar. Sebanyak
1.236 berkas penting diarsipkan dan diverifikasi sebelum staging digunakan lagi:
`artifacts/build-archives/bm-symbols-apk-20260908/20260911T023144026Z-03cd675d/manifest.json`.
Checkpoint, dataset, model lama, dan arsip training tidak ditimpa.

Pemeriksaan ulang r2 tercatat di `source-checks-r2.json`. Perbaikan kamera
membaca deteksi terbaru yang sudah selesai ketika tombol Pindai ditekan, sehingga
hasil baru tidak perlu menunggu jadwal pembaruan tampilan berikutnya. Tidak ada
inferensi tambahan. Batas usia 500 ms serta pemeriksaan geometri dan sesi tetap
berlaku. Lima tes regresi mencakup bingkai baru, bingkai kedaluwarsa, perubahan
area kamera, dan callback setelah kamera ditutup. Model, kebijakan decoding,
manifest, dan bukti paritas tetap memiliki hash yang sama.

## Hasil emulator dan batas pengujian

Peninjauan sumber UI dan pemeriksaan layar emulator telah mencakup pengenalan
awal, Beranda, aktivasi kamera, Pengaturan, Bantuan, dan Privasi. Tampilan tersebut
bersih dari label development. Audit 106 string aplikasi pada tujuh halaman
menemukan nol caption development; XML, screenshot, dan hash sumber tercatat
di `ui-copy-audit.json`. Ke-243 berkas staging cocok dengan sumber akhir.

APK r2 terpasang pada emulator API 36 dan SHA-256 berkas terpasang cocok dengan
APK yang diberikan. Wi-Fi dan data seluler dimatikan; JavaScript sudah berada
di APK. Pemindai melewati pemeriksaan awal model, membuka kamera, dan membaca
fixture kamera virtual menjadi **Yys**, tepat sesuai teks acuan termasuk huruf
kapital, dalam **2.411 ms**. Hasil terlihat pada screenshot dan hierarki UI.

Fixture pertama terpotong di kanan oleh kamera virtual. Fixture kedua memakai
patch validation yang sama, tanpa mengubah piksel atau skala, dengan posisi yang
disesuaikan terhadap area kamera emulator. Ini satu kasus integrasi kamera
perangkat lunak, bukan estimasi akurasi kamera atau tambahan data TEST.

Ada dua batas runtime yang masih terlihat:

- Setelah Pindai ulang dan kembali dari background, preview muncul lagi, tetapi
  satu ketukan ditolak karena snapshot tracking yang segar tidak tersedia.
  Batas 500 ms tetap berlaku; keandalan pemindaian berulang belum dinyatakan lulus.
- Bacakan gagal saat offline. Google TTS melaporkan nama suara lokal tersedia,
  tetapi log menunjukkan paket Indonesia belum tersedia, peralihan ke suara
  server, dan kegagalan sintesis `-4` ketika jaringan dimatikan. Pemeriksaan
  ketersediaan suara dapat menghasilkan positif palsu pada konfigurasi ini.
  Emulator juga dijalankan tanpa keluaran audio; suara terdengar belum dinilai.

Tidak ada fatal exception aplikasi dalam log r2 yang dikumpulkan. Dialog ANR
System UI saat boot dan timeout beberapa perintah UIAutomator dicatat sebagai
batas lingkungan pengujian. Keseluruhan smoke, TalkBack, dan kamera ponsel fisik
belum dinyatakan lulus. Angka akurasi 87,6% tetap khusus benchmark terpisah.

[Laporan QA akhir](../../output/apk-id-v5-20260911/final-qa.json) memuat hash,
hasil, dan batas tersebut. Folder build terdaftar tetap **satu**, tanpa staging
baru atau penghapusan checkpoint/dataset.
