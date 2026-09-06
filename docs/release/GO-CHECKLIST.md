# BrailleMe Android Production GO Checklist

Keputusan GO hanya boleh diberikan untuk satu commit, model, dan AAB yang sama. Isi bukti; jangan mencentang berdasarkan rencana atau hasil emulator saja.

Build production selalu fail-closed: model harus berstatus `verified` dan seluruh evidence gate wajib lulus. Tidak ada profile pengecualian untuk model kandidat.

## Identitas kandidat

- [ ] Git commit SHA: `________________________________________`
- [ ] Model kata version: `________________________________________`
- [ ] Model kata SHA-256: `________________________________________`
- [ ] EAS build ID: `________________________________________`
- [ ] AAB SHA-256: `________________________________________`
- [ ] Attestation exact-build: `artifacts/release/________________________.json`
- [ ] Android versionName/versionCode: `________________ / ________________`
- [ ] Upload certificate SHA-256: `________________________________________`

## Gate otomatis

- [ ] `npm ci`
- [ ] `npm run expo:dependencies:check`
- [ ] `npm run doctor`
- [ ] `npm run lint`
- [ ] `npm run typecheck`
- [ ] `npm run test:coverage`
- [ ] `npm run release:config:verify`
- [ ] `npm run model:verify`
- [ ] `npm run release:model:verify`
- [ ] `npm run export:android` dengan `APP_ENV=production`
- [ ] `DEPENDENCY-RISK-REVIEW.md` sudah diperbarui dan disetujui release owner.

## Model dan physical holdout

- [ ] Benchmark publik blok lengkap: akurasi kata utuh ≥75% (penolakan dihitung gagal), ketepatan hasil diterima ≥95%; keduanya dilaporkan menurut panjang 1–12 huruf, dengan panjang tanpa sampel dinyatakan jelas.
- [ ] Batas kata/sel benchmark diperiksa sebelum inferensi; split menurut sumber dan pemilihan checkpoint/ambang hanya pada validation dibuktikan.
- [ ] Seluruh 66 blok eksperimen lama dilaporkan sebagai regresi; bukan otomatis bukti batas kata lengkap.
- [ ] Seluruh 44 negatif publik menghasilkan nol kata diterima; data ini tidak dinyatakan sebagai holdout independen.
- [ ] Kesetaraan backend training–Keras TensorFlow–TFLite serta crop, orientasi, resize, urutan kanal, dan pemetaan kotak Android diuji pada artefak yang sama.

- [ ] Hak penggunaan dan redistribusi seluruh dataset training dikonfirmasi tertulis.
- [ ] Minimum 260 sampel holdout fisik independen, minimum 10 per kelas A-Z.
- [ ] Holdout berasal dari sesi/sumber yang tidak pernah dipakai training atau tuning.
- [ ] Akurasi dan macro-F1 masing-masing minimal 90%.
- [ ] Recall setiap kelas minimal 80%.
- [ ] Minimum 12 scene negatif; nol false accept pada confidence 0,90.
- [ ] Golden native camera A-Z lulus 26/26 pada model SHA yang sama.
- [ ] Threshold dibekukan sebelum holdout dibuka.

## Perangkat dan aksesibilitas

- [ ] Minimum dua perangkat fisik target dan versi Android dicatat.
- [ ] Satu kata end-to-end lulus untuk 1-12 huruf, termasuk huruf berulang.
- [ ] Satu tekanan Pindai kata hanya mengambil satu foto; hasil lama tidak dapat masuk setelah pindai ulang atau pergantian sesi.
- [ ] Pindai ulang mengganti hasil secara utuh; Bacakan hanya aktif untuk foto terakhir yang berhasil; interupsi TTS lulus.
- [ ] Lebar bingkai dapat disesuaikan untuk satu huruf dan kata panjang; area pilihan sesuai foto pada seluruh orientasi yang didukung.
- [ ] Foto hanya di memori dan dilepas setelah pekerjaan latar selesai, termasuk pindai ulang, keluar layar, background, dan error.
- [ ] Preview minimum 24 FPS dan p95 tekanan Pindai kata hingga hasil/penolakan maksimum 2 detik; capture dan inferensi dilaporkan terpisah.
- [ ] Soak kamera minimum 10 menit tanpa crash, ANR, stall, atau false utterance.
- [ ] Fresh install, izin grant/deny/permanent deny/revoke, background, lock, dan reboot lulus.
- [ ] Cold start dalam mode pesawat lulus; kamera, inferensi, dan voice lokal tetap berfungsi.
- [ ] TalkBack traversal, fokus setelah panel dinamis, font 200%, TTS interruption, serta recovery nonvisual lulus.
- [ ] Gelap, glare, blur, rotasi, jarak, material, dan sel off-center diuji.

## Artefak dan Google Play

- [ ] Path model, evidence, verifier, workflow, dan konfigurasi rilis dilindungi required independent review/CODEOWNERS-equivalent.
- [ ] Approval model berasal dari attestation immutable bertanda tangan oleh protected CI/reviewer dan mengikat model SHA, evidence digest, commit, serta device report yang sama.
- [ ] EAS/Expo owner dan `extra.eas.projectId` terkonfirmasi.
- [ ] Google Play App Signing aktif dan upload key memiliki backup aman.
- [ ] AAB store-signed berasal dari commit yang dicatat di atas.
- [ ] Merged manifest hanya memuat izin yang disetujui; backup/debuggable/network diperiksa ulang.
- [ ] Privacy policy memiliki URL HTTPS aktif dan dapat diakses tanpa login.
- [ ] Data safety, ads, app access, target audience, content rating, kategori, negara, harga, dan kontak selesai.
- [ ] Store listing Indonesia, ikon 512 x 512, feature graphic 1024 x 500, screenshot aktual, dan release notes selesai.
- [ ] Play pre-launch report tidak memiliki error yang belum diselesaikan.
- [ ] Jika akun personal baru terpengaruh, closed test 12 tester selama 14 hari telah selesai.

## Keputusan

- [ ] Product owner menyetujui keterbatasan: satu kata Braille dasar A–Z, satu baris, 1–12 huruf; kalimat, halaman, singkatan, angka, dan tanda baca belum didukung.
- [ ] Accessibility reviewer menyetujui bukti TalkBack/perangkat.
- [ ] Model reviewer menyetujui holdout dan negative corpus.
- [ ] Release owner menyetujui signing, policy, dan staged rollout.

**Status:** `NO-GO` sampai seluruh item wajib di atas memiliki bukti yang dapat ditelusuri.
