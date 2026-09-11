# Eksperimen huruf kapital dan angka

**Hasil 8 September 2026: epoch 10 terpilih oleh training, tetapi tidak dikirim
dalam APK.** Promosinya dibatalkan karena regresi huruf publik dari 323/607
menjadi 306/607 dengan decoder yang sama. APK memakai checkpoint baseline
yang dipertahankan bersama decoder baru. Lihat
[laporan deployment baseline](evidence/braille-word-reader-v1/indonesian-symbols-baseline-20260908/EXPERIMENT-ID.md).

Pipeline ini melatih lokasi sel dan enam pola titik dengan arsitektur yang sama.
Penerjemah `six-dot-id-v3` kemudian membaca aturan `indonesian-basic-v1`: huruf
tanpa tanda menjadi kecil, titik 6 untuk kapital, dan titik 3456 untuk angka.
Pola titik tidak diganti menjadi label 36 kelas.

## Data dan batas klaim

- AngelinaDataset dipulihkan pada revisi `e13038d483bb58e17046d640e078dc4768276e9b`
  di `C:\bm-datasets\AngelinaDataset`.
- DSBI TRAIN dipulihkan pada revisi `6937eb8973efe5ff337e5c018485cb626303f5c2`
  di `C:\bm-datasets\DSBI-train-6937eb8`. Hash setiap sumber dicocokkan dengan
  manifest lama. Piksel DSBI TEST tidak diunduh.
- Semua halaman sumber dibagi sebelum pembuatan potongan atau komposisi. Data
  publik eksternal 607 contoh tidak digunakan untuk gradien atau pemilihan epoch.
- Data tambahan adalah **komposisi sintetis dari potongan foto publik**, bukan
  foto kata kapital/angka Indonesia yang ditemukan pada sumber asli. Tanda
  titik 6 pada teks Rusia/matematika tidak diberi makna kapital Indonesia.
- Validation asli tidak memiliki titik 6. Pola yang tidak tersedia dilengkapi
  menggunakan refleksi/rotasi potongan sel dari sumber validation yang sama,
  dengan pemetaan bit serta transformasi dicatat secara eksplisit. Ini tetap
  data sintetis; hasilnya bukan akurasi kertas Braille fisik.
- Komposisi baru: 3.096 training dan 516 validation. Validation terdiri dari
  140 huruf kecil, 140 kapital tunggal, 140 kapital seluruh kata, 64 angka,
  serta 32 rangkaian yang wajib ditolak.

## Menjalankan percobaan baru

Jalankan di akar repo menggunakan lingkungan Torch CUDA yang sudah dipulihkan:

```powershell
$env:KERAS_BACKEND = 'torch'
$trainingPython = 'C:\bm-envs\word-symbols-torch\Scripts\python.exe'
$experiment = 'C:\bm-symbols-' + (Get-Date -Format 'yyyyMMdd-HHmmss')

& $trainingPython training/scripts/word_symbol_prepare.py --output $experiment
if ($LASTEXITCODE -ne 0) { throw 'Persiapan data gagal' }

& $trainingPython training/scripts/word_symbol_train.py `
  --experiment $experiment --epochs 10 --batch-size 24
if ($LASTEXITCODE -ne 0) { throw 'Percobaan gagal; periksa log sebelum melanjutkan' }
```

Persiapan menyimpan hash checkpoint/evidence lama dan salinan baseline sebelum
melakukan training. Jangan menghapus folder percobaan atau menimpa checkpoint
lama. Percobaan yang sudah memiliki konfigurasi training tidak dijalankan ulang
secara diam-diam. Batch 12 hanya diizinkan setelah laporan GPU kehabisan memori
pada batch 24; percobaan pengganti dimulai dari baseline asli di folder berbeda.

Driver mengevaluasi baseline terlebih dahulu. Bila salah satu kelompok gagal
75% teks utuh, 95% ketepatan hasil diterima, atau kasus wajib ditolak, driver
menjalankan satu putaran maksimal 10 epoch dengan learning rate `0.00002`.
Setiap epoch disimpan. Data training lama, pembobotan sumber, serta supervisi
DSBI parsial dipertahankan dan ditambah komposisi baru.

Pemilihan memakai threshold baseline yang dibekukan. Kandidat yang menurunkan
akurasi teks utuh atau ketepatan hasil diterima pada salah satu kelompok huruf
lama (natural/stress) tidak menggantikan baseline. Di antara kandidat yang
memenuhi aturan tersebut, semua kelompok harus mencapai ketepatan 95% dan
kasus wajib ditolak harus lolos sebelum prioritas bergeser ke teks utuh.
Jika belum terpenuhi, pembanding utama adalah ketepatan kelompok terlemah.
**Model terpilih tidak otomatis lulus target atau siap rilis.**

## Konversi dan bukti

Gunakan lingkungan TensorFlow terpisah:

```powershell
$exportPython = 'C:\bm-envs\word-symbols-tensorflow\Scripts\python.exe'
$selection = "$experiment/selection"
$data = '.braille-training-work/word-scan-v1-20260905/data'

& $exportPython training/scripts/word_export.py --run $selection --data $data
if ($LASTEXITCODE -ne 0) { throw 'Kesetaraan konversi gagal' }

& $exportPython training/scripts/word_benchmark.py `
  --run $selection --data $data --plain-latin-lowercase `
  --legacy training/evidence/braille-word-reader-v1/experimental-internal-best/public-benchmark.json
```

Oracle standar tetap berisi 405 contoh validation. Oracle komposisi simbol
disimpan terpisah. `word_symbol_replay.py` menyediakan tensor, piksel, dan
keputusan Python aktual untuk `scripts/word-decoder-parity/replay-id.cjs`;
`word_symbol_evidence.py` menghitung laporan simbol dari baris keputusan TFLite.
Laporan final harus mencantumkan penolakan, baseline, setiap epoch, dan batas
validasi fisik. Model baru hanya masuk APK setelah bukti konversi serta
kesetaraan decoder dibuat ulang untuk file model yang sama.

Setelah pemilihan training selesai, bandingkan juga baseline dan kandidat pada
607 crop publik menggunakan **decoder dan threshold yang sama**. Data publik
ini sudah pernah diamati selama pengembangan dan bukan holdout independen.
Jika kandidat menurunkan regresi huruf dasar, **batalkan promosinya** dan
pertahankan baseline. Jangan mengganti sejarah `checkpoint-selection.json`
atau memilih epoch lain berdasarkan hasil eksternal. Catat veto pada
`deployment-selection.json` terpisah dan buat ulang semua bukti untuk model
yang benar-benar dimasukkan ke APK.

Pada percobaan 8 September 2026, epoch 10 terpilih oleh validation training,
tetapi tidak dipromosikan karena turun dari 323 menjadi 306 teks benar pada
607 crop dengan decoder yang sama. Checkpoint epoch 10 tetap disimpan untuk
riset. Evaluasi baseline tanpa training tambahan dapat dibuat menggunakan:

```powershell
& $trainingPython training/scripts/word_symbol_deployment_baseline.py `
  --experiment $experiment --output "$experiment/deployment-baseline"
```

Gunakan direktori tersebut untuk konversi, benchmark, oracle, diagnostik, dan
startup baru. Hasil kandidat tersimpan pada evidence `indonesian-symbols-20260908`;
hasil model untuk APK pada `indonesian-symbols-baseline-20260908`.

Lingkungan Python/CUDA, dataset, dan percobaan menggunakan ruang disk di luar
repo. Folder tersebut masih diperlukan untuk melanjutkan atau mengaudit training.

## Retensi untuk cleanup berikutnya

[Indeks retensi 8 September 2026](checkpoints/retained-models-20260908.json)
melengkapi [indeks historis 6 September 2026](checkpoints/retained-models.json).
Gunakan keduanya saat merencanakan cleanup. Indeks baru mencatat hash model
baseline untuk APK, checkpoint epoch 10 yang tidak dipromosikan, sumber,
konfigurasi, bukti, seluruh 10 epoch mentah, serta lokasi dataset dan lingkungan
Python. Status tidak dipromosikan bukan alasan menghapus hasil training;
keduanya dipertahankan untuk pengembangan berikutnya.
