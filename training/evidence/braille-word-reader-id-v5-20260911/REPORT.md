# Model eksperimen Braille ID-v5 — 11 September 2026

Hasil test utama TFLite: **438/500 (87.6%) teks utuh tepat**. Target keseluruhan 80% tercapai.

Presisi di antara teks positif yang diterima: 92.41%. Input tidak valid yang diterima: 0/40. Kedua hasil ini dilaporkan terpisah dari target akurasi keseluruhan.

| Kelompok | Tepat / total | Akurasi teks utuh | Presisi diterima |
| --- | ---: | ---: | ---: |
| capital_single | 87/100 | 87.0% | 89.69% |
| capital_word | 82/100 | 82.0% | 89.13% |
| lowercase | 84/100 | 84.0% | 88.42% |
| numbers | 94/100 | 94.0% | 98.95% |
| punctuation | 91/100 | 91.0% | 95.79% |

Audit konteks asli terpisah: 185/200 tepat. Audit ini terdiri atas potongan pendek dari sel asli yang berurutan; cakupannya tidak seimbang dan tidak memiliki kapital alami. Angkanya tidak dimasukkan ke denominator 500.

| Regresi historis | Baseline tepat / total | Kandidat tepat / total |
| --- | ---: | ---: |
| external607 | 323/607 | 350/607 |
| legacy405 | 355/405 | 343/405 |
| negative44 | 0/44 invalid diterima | 0/44 invalid diterima |

Model dan kebijakan inferensi disimpan sebagai kandidat eksperimen; aset model aplikasi tidak diganti dan tidak ada APK atau folder build baru.

Kebijakan inferensi terpilih: `cascade_zoom_fallback`. Gunakan kebijakan TTA, parser, dan ambang yang ikut dibekukan; berkas model saja tidak mewakili seluruh metode yang diuji.

Test utama memakai 500 rangkaian seimbang dari potongan sel hasil pemindai DSBI yang disusun ulang, 96% multikarakter, dengan 40 urutan invalid terpisah. Halaman sumbernya baru dan terpisah dari sumber latihan serta test lama. Ini belum membuktikan akurasi kamera ponsel pada kertas Braille Indonesia nyata.

Checkpoint SHA-256: `ac9e30cd27cb22ca4f97db42637d3a9e2a27ed47d13a962a661c3bb01c0e66bb`
TFLite SHA-256: `6847c91fe186169ba5208a7444f9c8bedc753228bc42b78c691471a85453a6cb`

Pemeriksaan konversi awal **gagal** pada `atol=0.0001, rtol=0.0001`. Selisih absolut maksimum Torch/TFLite adalah `0.0002515912`. Sebelum test dibuka, tinjauan terpisah menerima batas empiris `atol=0.0003, rtol=0` untuk seluruh 540 citra validasi dalam 6 orientasi/skala; seluruh 2700 keputusan kebijakan identik, termasuk teks, penolakan, urutan mask fisik, dan rute. Tinjauan ini tidak mengubah model atau ambang inferensi, dan bukan jaminan kesetaraan pada semua input baru. Akurasi test utama di atas diukur langsung dengan TFLite dan dilaporkan terpisah dari pemeriksaan konversi.

Bukti asli tetap tersedia di [export-parity-original-failed.json](export-parity-original-failed.json); hasil tinjauan di [export-parity.json](export-parity.json) dan skripnya di [review_export_float32.py](review_export_float32.py).
