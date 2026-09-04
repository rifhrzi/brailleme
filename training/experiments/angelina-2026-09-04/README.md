# Angelina Training Experiment — 2026-09-04

Model aktif BrailleMe pertama-tama diuji tanpa retraining pada 69.056 crop A-Z
dari 290 halaman Angelina dan hanya mencapai top-1 22,26%. Eksperimen kemudian
melatih kandidat baru dari partisi upstream train Angelina ditambah dataset
Kaggle lama. Pemilihan epoch memakai partisi upstream val yang berbagi source
family dengan train, sehingga angka validation 99,40% hanya diagnostik dan
berpotensi optimistis. Penilaian eksternal memakai 9.625 crop dari 50 halaman
`uploaded/test.txt`.

Kandidat TFLite mencapai top-1 93,11% pada external test. Pada threshold 0,90,
coverage 72,75% dan accepted accuracy 99,30%. Ini melewati target karakter
internal 75%, tetapi bukan bukti akurasi kamera native.

Kandidat kini mengganti model aktif aplikasi **hanya untuk development lokal**
sebagai `0.3.0-candidate.1`. Model v1 tidak ditimpa dan tetap tersedia untuk
rollback. Kandidat belum boleh didistribusikan karena:

- repository Angelina tidak memiliki `LICENSE` eksplisit pada revision yang
  dipin;
- external test tidak memiliki Y dan hanya memiliki 9 contoh V; recall V hanya
  11,11%;
- `uploaded/test.txt` kini sudah memengaruhi keputusan promosi dan tidak boleh
  dipakai lagi sebagai holdout promosi yang pristine untuk kandidat berikutnya;
- satu dari 44 scene non-Braille salah diterima sebagai C pada confidence
  0,9014, dan 3 dari 50 halaman external test berada di bawah 75%;
- crop berasal dari bounding box ground-truth, bukan ROI kamera aplikasi;
- self-test post-crop native baru lulus pada emulator; physical holdout, native
  camera golden parity, dan benchmark perangkat fisik/release belum ada.

Detail ringkas dan hash artefak tersedia di [RESULTS.json](RESULTS.json).
Artefak aktif dan evidence berada di `assets/models/braille-cell-classifier-v2`
dan `training/evidence/braille-cell-classifier-v2`. Script
`training/scripts/train_angelina_candidate.py` menyediakan resep training yang
dapat dijalankan ulang; hash hasil retraining tidak dijamin identik lintas
environment.
