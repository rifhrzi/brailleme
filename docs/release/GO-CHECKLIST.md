# BrailleMe Android Production GO Checklist

Keputusan GO hanya boleh diberikan untuk satu commit, model, dan AAB yang sama. Isi bukti; jangan mencentang berdasarkan rencana atau hasil emulator saja.

Build production selalu fail-closed: model harus berstatus `verified` dan seluruh evidence gate wajib lulus. Tidak ada profile pengecualian untuk model kandidat.

## Identitas kandidat

- [ ] Git commit SHA: `________________________________________`
- [ ] Model version: `0.3.0-candidate.1`
- [ ] Model SHA-256: `7a41942cf330c9558884385221deb69f282adfd8d39be00a257c75ec9572c318`
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
- [ ] Tambah huruf idempoten; hasil lama tidak dapat masuk setelah pindai ulang atau pergantian sesi.
- [ ] Hapus huruf terakhir, hapus kata, pindai ulang, pembacaan kata, dan interupsi TTS lulus.
- [ ] Preview minimum 24 FPS dan p95 ROI-layak-ke-stabil maksimum 2 detik.
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

- [ ] Product owner menyetujui keterbatasan: input satu sel A-Z per langkah, output maksimum satu kata; bukan pemindaian kata/baris/halaman sekaligus.
- [ ] Accessibility reviewer menyetujui bukti TalkBack/perangkat.
- [ ] Model reviewer menyetujui holdout dan negative corpus.
- [ ] Release owner menyetujui signing, policy, dan staged rollout.

**Status:** `NO-GO` sampai seluruh item wajib di atas memiliki bukti yang dapat ditelusuri.
