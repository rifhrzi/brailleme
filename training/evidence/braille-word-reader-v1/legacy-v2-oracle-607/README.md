# Diagnosis classifier v2 dengan batas sel dari anotasi

Ini **pengujian segmentasi oracle, bukan akurasi pemindai aplikasi**. Setiap sel
diberikan kepada classifier dari batas anotasi yang benar, kemudian hasilnya
dirangkai menurut urutan anotasi. Kamera tidak memiliki informasi tersebut.
Model, ambang, fixture, dan aplikasi tidak diubah; tidak ada fallback yang dipasang.

Seluruh 607 blok publik yang dibekukan memuat 2.242 sel unik dari 46 halaman
`uploaded/test.txt`. Model v2 yang dipertahankan menghasilkan:

| Ukuran | Benar / penyebut | Hasil |
| --- | ---: | ---: |
| Huruf, argmax tanpa penolakan | 2.077 / 2.242 | 92,64% |
| Blok oracle utuh, argmax tanpa penolakan | 499 / 607 | 82,21% |
| Blok oracle utuh, setiap sel harus mencapai ambang lama 0,90; penolakan gagal | 324 / 607 | 53,38% |
| Ketepatan blok oracle yang diterima pada ambang lama 0,90 | 324 / 329 | 98,48% |

Jadi dua sasaran 75% kata utuh dan 95% ketepatan diterima belum dipenuhi sekaligus
pada ambang yang sudah ada, bahkan dengan batas sel yang diketahui. Angka 82,21%
tidak dapat dipakai sebagai hasil fitur atau alasan mengaktifkan model lama.

| Panjang | Jumlah blok | Huruf benar | Blok oracle utuh, argmax | Benar / diterima pada 0,90 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 152 | 146/152 (96,05%) | 146/152 (96,05%) | 116/116 |
| 2 | 109 | 202/218 (92,66%) | 95/109 (87,16%) | 62/62 |
| 3 | 73 | 198/219 (90,41%) | 59/73 (80,82%) | 35/37 |
| 4 | 66 | 249/264 (94,32%) | 54/66 (81,82%) | 31/33 |
| 5 | 70 | 315/350 (90,00%) | 48/70 (68,57%) | 28/29 |
| 6 | 43 | 247/258 (95,74%) | 35/43 (81,40%) | 15/15 |
| 7 | 35 | 226/245 (92,24%) | 23/35 (65,71%) | 14/14 |
| 8 | 24 | 169/192 (88,02%) | 14/24 (58,33%) | 6/6 |
| 9 | 15 | 128/135 (94,81%) | 11/15 (73,33%) | 7/7 |
| 10 | 13 | 122/130 (93,85%) | 9/13 (69,23%) | 7/7 |
| 11 | 5 | 54/55 (98,18%) | 4/5 (80,00%) | 3/3 |
| 12 | 2 | 21/24 (87,50%) | 1/2 (50,00%) | 0/0 |

Perbandingan terhadap keluaran v13 yang sudah tersimpan menunjukkan bahwa
lokalisasi dan geometri merupakan hambatan utama pada data ini. Dengan aturan
pencocokan diagnostik yang sama seperti `word_failure_analysis.py` (pusat
prediksi terdekat yang belum digunakan, jarak paling jauh 0,45 tinggi sel GT):

- 2.046 sel memiliki pasangan prediksi v13. Masker v13 benar 1.919/2.046
  (93,79%); classifier lama dengan crop ideal benar 1.928/2.046 (94,23%).
  Selisihnya hanya sembilan sel pada populasi pasangan yang sama.
- Dari 196 sel GT tanpa pasangan v13, 149 dapat dibaca classifier lama.
  Sebanyak 149 dari selisih bersih 158 sel benar (94,3%) berasal dari kelompok
  sel yang tidak ditemukan, bukan peningkatan pembacaan pada sel berpasangan.
- V13 menolak 204 blok sebagai terpotong; 148 di antaranya dapat dibaca utuh
  oleh classifier lama dengan crop ideal. Ini mendukung perbaikan deteksi dan
  penanganan batas sel. Ini tidak membenarkan melonggarkan penolakan geometris:
  crop oracle juga membuang persoalan sel lain, bagian terpotong, dan penentuan jumlah sel.

Hasil kata v13 yang sebenarnya pada fixture ini tetap 262/607 (43,16%), dengan
262/291 (90,03%) hasil diterima benar. Dua model menerima representasi gambar
yang berbeda dan dilatih dengan data berbeda, sehingga ini bukan perbandingan
arsitektur yang terkontrol. Terdapat 98 sel yang dibaca benar v13 tetapi salah
oleh classifier lama, serta 107 kesalahan masker v13 yang dibaca benar oleh
classifier lama. Mengganti pembaca titik saja tidak menjelaskan selisih hasil kata.

| Prediksi | GT huruf tersebut | Benar | Salah masuk kelas | Kesalahan masuk terbesar |
| --- | ---: | ---: | ---: | --- |
| C | 7 | 7 | 4 | A→C: 3; I→C: 1 |
| H | 28 | 27 | 11 | E→H: 4; G→H: 4 |
| J | 33 | 31 | 15 | G→J: 8; T→J: 3 |

Secara total hanya 95 dari 2.242 prediksi berupa C/H/J; 30 di antaranya salah
(18,18% dari seluruh 165 kesalahan). Tidak terjadi keluaran yang didominasi
tiga huruf tersebut pada crop ideal. Kesalahan terbanyak justru N→O (24).
Jumlah C sangat kecil, V hanya lima contoh, dan Y tidak ada. Data ini tidak
membuktikan penyebab C/H/J di perangkat pengguna karena tidak ada foto kamera
fisik pengguna yang diuji.

Model SHA-256:
`7a41942cf330c9558884385221deb69f282adfd8d39be00a257c75ec9572c318`.
Crop mengikuti fungsi murni dari arsip training yang diverifikasi hash:
kotak persegi berpusat anotasi, sisi `1.25 × max(lebar, tinggi)`, dibatasi dan
digeser ke dalam halaman, batas dibulatkan dengan `round`, RGB asli diubah ke
28×28 memakai PIL bilinear, lalu float32 `/255`. Bukan crop dari gambar kata
480×192 yang sudah diperkecil. Setiap crop dan baris CSV memiliki hash/batas
di `frozen-crops.json.gz`; seluruh probabilitas ada di `probabilities.npz`.

Training historis v2 memakai 191 halaman train upstream (47.338 sel A–Z)
ditambah 1.560 gambar Kaggle, dan 49 halaman validation upstream. Pemisahan ini
berbeda dari pemisahan keluarga sumber model kata. Seluruh 46 halaman diagnosis
tidak terdapat pada daftar train/validation kedua jalur; namun `uploaded/test`
sudah memengaruhi promosi historis v2 dan kini menjadi regresi pengembangan,
bukan holdout baru. Batas 607 blok juga bersifat geometris dan belum direview
manusia sebagai batas kata leksikal.

Negatif dilaporkan terpisah tanpa inferensi ulang: evaluasi lama mencatat
satu penerimaan keliru dari 44 gambar publik pada ambang 0,90 (C, 0,9014),
sedangkan v13 menolak 44/44 gambar yang juga dipakai untuk training/regresi
model kata. Kontrak crop dan penolakan berbeda; keduanya bukan bukti negatif
kamera fisik independen.

Inferensi satu proses CPU TFLite/XNNPACK, satu thread, TensorFlow 2.20.0,
Pillow 12.3.0, NumPy 2.0.2; tanpa GPU. Peak working set Windows 345.571.328 byte
(sekitar 330 MiB), selesai sekitar sembilan detik. Protokol, input, dan aturan
ringkasan dibekukan sebelum inferensi. Arsip script `diagnose.py` dijalankan
dari struktur kerja semula `.braille-training-work/word-scan-v1-20260905/legacy-v2-oracle-607-diagnostic/`;
arsip ini mempertahankan sumbernya, bukan mengubah lokasi relatif agar bisa
langsung dieksekusi dari folder evidence.

Catatan pelaporan v13: ringkasan lama mencatat `no_grid=57` dan
`low_confidence=20`, sedangkan baris final mencatat masing-masing 28 dan 49.
Sebanyak 29 penolakan kualitas diberi placeholder kosong saat agregasi.
Erratum terpisah mengikat hash laporan asli dan memperbaiki pembacaan alasan
tanpa menulis ulang laporan atau mengubah hitungan benar/diterima/panjang.
