# Diagnosis geometri v15 dari keluaran tersimpan

Ini **simulasi memakai kotak anotasi, bukan akurasi model atau kalibrasi**. Semua keluaran berasal dari inferensi v15 yang sudah tersimpan; tidak ada inferensi baru, training, perubahan threshold, relabeling, kamus, atau perubahan aplikasi.

Hasil aktual tetap **325/607 blok benar (53,54%)**, dengan **325/353 hasil diterima benar (92,07%)**.

Sebanyak **93 penolakan** sudah memiliki semua sel GT dengan masker benar dan yakin, tanpa sel tambahan: 92 ditolak sebagai terpotong dan satu karena jarak. Mengganti hanya kotak pasangan dengan kotak GT memulihkan **92 blok**, menghasilkan **417/607 (68,70%)** dan **417/461 (90,46%)** hasil diterima benar. Semua 325 keberhasilan lama tetap benar.

Batas sebelum pemeriksaan kelayakan geometri adalah **418/607 (68,86%)** kumpulan sel lengkap, benar, dan yakin. Satu blok HHDJ masih ditolak saat memakai kotak GT: rasio jarak pusat terhadap median tinggi anotasi 2,005, melewati batas tetap 1,2. Jadi mencapai sedikitnya 456/607 (75%) memerlukan **38 blok benar tambahan di luar batas kumpulan sel tetap**, atau 39 di luar simulasi kotak GT yang benar-benar dihitung. Angka417 bukan batas universal semua kemungkinan perubahan geometri.

Proposal setelah selection/NMS dibekukan identitas dan jumlahnya; score dan keenam probabilitas titik tetap. Hanya XYXY/ukuran pasangan GT diganti, lalu disortir dan diperiksa oleh pernyataan decoder v15 yang sama. Sel hilang tidak ditambahkan, sel ekstra tidak dibuang, NMS tidak diulang, dan penolakan kualitas tetap. Perpindahan kotak oracle bisa melampaui kemampuan offset grid model nyata.

Dari 2.242 sel GT terdapat **2.085 pasangan**, **157 sel tidak ditemukan**, **118 masker pasangan salah**, dan **73 pasangan kurang yakin** (dua kelompok terakhir dapat tumpang tindih). Dari157 sel hilang,44 berasal dari29 penolakan kualitas,112 memiliki puncak lokal terdekat di bawah ambang proposal0,25, dan satu mempunyai puncak di atas ambang tetapi tidak memperoleh pasangan setelah selection/NMS. Terdapat46sel ekstra:30duplikat dekat target,14tekstur/sel tanpa anotasi pasangan, satu konteks parsial, dan satu konteks lengkap. Kategori berbasis jarak tidak membuktikan titik verso.

Pencocokan nearest-center memakai batas tetap0,45tinggi GT. Greedy kiri-ke-kanan, greedy terbalik, dan assignment optimal berpasangan menghasilkan pasangan identik untuk seluruh607blok. Replay decoder utuh dan tahap setelah NMS sama persis dengan setiap baris laporan yang dibekukan.

Perbaikan kotak juga membuka16hasil salah yang sebelumnya ditolak; bersama28kesalahan diterima lama, tersisa44hasil salah dari461penerimaan. Pembacaan titik/keyakinan dan deteksi proposal masih harus membaik. Simulasi ini tidak mengevaluasi negatif baru dan tidak memenuhi75/95.

| Panjang | Blok | Benar aktual | Pulih dari penolakan | Benar dengan kotak GT | Diterima dengan kotak GT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 152 | 94 | 10 | 104 | 111 |
| 2 | 109 | 63 | 10 | 73 | 89 |
| 3 | 73 | 43 | 8 | 51 | 59 |
| 4 | 66 | 37 | 12 | 49 | 54 |
| 5 | 70 | 31 | 16 | 47 | 49 |
| 6 | 43 | 19 | 10 | 29 | 31 |
| 7 | 35 | 14 | 10 | 24 | 26 |
| 8 | 24 | 8 | 7 | 15 | 16 |
| 9 | 15 | 6 | 7 | 13 | 13 |
| 10 | 13 | 6 | 2 | 8 | 8 |
| 11 | 5 | 4 | 0 | 4 | 5 |
| 12 | 2 | 0 | 0 | 0 | 0 |

Seluruh607fixture adalah regresi pengembangan publik dengan batas geometris, bukan holdout kamera fisik atau batas kata yang direview manusia. Counterfactual ini bukan ramalan peningkatan residual11logit; perubahan selection/NMS/objectness/dots dalam model nyata berada di luar intervensi terbatas ini.

`summary.json` memuat ukuran ringkas; `report.json.gz` mempertahankan seluruh pasangan, sel hilang/tambahan, dan hitungan menurut sumber tanpa menyalin tensor output besar. `protocol.json` mengikat hash model, decoder, fixture, keluaran tersimpan dan script. Script arsip dijalankan dari struktur kerja asli `selected-v15-independent-diagnosis/`; ia membaca frame tensor satu per satu. Peak working set86.687.744byte; TensorFlow/Torch/Keras tidak diimpor.
