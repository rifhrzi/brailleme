# Angelina External Benchmark

Benchmark ini menguji model TFLite BrailleMe yang sudah dibekukan pada
`uploaded/test.txt` dari
[AngelinaDataset](https://github.com/IlyaOvodov/AngelinaDataset). Subset ini
tidak dipakai untuk training atau pemilihan epoch kandidat aktif.

## Batas penggunaan

- Revision dataset wajib `e13038d483bb58e17046d640e078dc4768276e9b`.
- Default hanya `uploaded/test.txt`. Partisi `books` dan `handwritten`
  train/val dipakai untuk melatih atau memilih epoch kandidat aktif; partisi
  tersebut hanya boleh dipilih eksplisit sebagai diagnostik, bukan sebagai
  external-test evidence.
- Angelina dominan teks Rusia. Pola titik Latin Y tidak muncul di
  `uploaded/test.txt`, sedangkan V hanya memiliki 9 contoh; cakupan kelas ini
  tidak cukup kuat untuk klaim recall produksi.
- Hanya pola enam titik yang memetakan tepat ke A-Z yang dinilai.
- Crop dibuat dari bounding box ground-truth Angelina; hasilnya bukan bukti
  deteksi ROI atau akurasi kamera native BrailleMe.
- Repository Angelina pada revision tersebut tidak memiliki file `LICENSE` di
  root. Jangan vendor atau redistribusikan gambarnya dari repository ini tanpa
  izin/lisensi yang jelas. Cantumkan sitasi penulis saat memakai hasilnya.
- Target 75% adalah target benchmark karakter internal. Status produksi dan
  physical holdout di `GO-CHECKLIST.md` tidak berubah.

## Menjalankan

Siapkan Python 3.12 dan dependency yang dibekukan pada
`training/evidence/braille-cell-classifier-v2/training-source/training_requirements.txt`,
lalu:

```powershell
git clone https://github.com/IlyaOvodov/AngelinaDataset.git C:\bm-datasets\AngelinaDataset
git -C C:\bm-datasets\AngelinaDataset checkout e13038d483bb58e17046d640e078dc4768276e9b

$env:BRAILLE_ANGELINA_DATASET_DIR = 'C:\bm-datasets\AngelinaDataset'
python training/scripts/benchmark_angelina.py
```

Secara default, keluaran dibuat di
`.braille-training-work/angelina-benchmark/`:

- `benchmark_report.json`: ringkasan metrik dan batas interpretasi;
- `data_quality.json`: cakupan halaman, kelas, anotasi, dan revision;
- `classification_report.json`: precision/recall/F1 per kelas;
- `confusion_matrix.json`: matriks kebingungan A-Z;
- `page_accuracy_summary.json`: jumlah dan detail halaman di bawah target 75%;
- `predictions.csv`: audit setiap crop berlabel;
- `negative_predictions.csv`: guardrail pada halaman non-Braille.

Benchmark dinyatakan memenuhi target internal hanya dari `top1_accuracy` semua
crop A-Z, bukan dari accuracy pada prediksi yang lolos threshold saja. Coverage
dan accepted accuracy tetap dilaporkan agar abstention tidak menyamarkan hasil.
Kelas yang tidak tersedia, termasuk Y pada external test, dicatat sebagai
`missing_classes`; benchmark tetap berjalan dan tidak menganggap recall kelas
yang hilang sebagai bukti keberhasilan.

## Eksperimen training terisolasi

Jika baseline gagal, eksperimen berikut melatih kandidat baru dari partisi
Angelina train, memilih epoch pada partisi val, lalu menilai satu kali pada
`uploaded/test.txt`. Dataset Kaggle lama tetap dicampurkan agar pola V dan Y
yang sangat langka di Angelina tidak hilang.

```powershell
$env:BRAILLE_DATASET_DIR = 'C:\path\to\Braille Dataset'
python training/scripts/train_angelina_candidate.py
```

Output tetap berada di `.braille-training-work/angelina-finetune` dan script
training tidak pernah mengganti model aplikasi otomatis. Kandidat yang sekarang
dipasang sebagai v2 dilakukan lewat promosi manual dan tetap berstatus
`unverified_candidate`; kandidat tidak boleh didistribusikan sebelum hak
penggunaan dataset dipastikan.
