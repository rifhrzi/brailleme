# Parity decoder pada keluaran model sebenarnya

## Decoder Bahasa Indonesia, kapital, dan angka

`replay-id.cjs` menjalankan parser, decoder, dan validasi event aplikasi yang sekarang. Mode `--semantics` memakai fixture semantik yang ditulis terpisah dan tensor sintetis; ini tidak mengukur model atau kamera.

```powershell
node scripts/word-decoder-parity/replay-id.cjs --semantics <laporan-baru.json>
node scripts/word-decoder-parity/replay-id.cjs <folder-oracle-python> <laporan-baru.json>
node --test scripts/word-decoder-parity/replay-id.test.cjs
```

Folder oracle harus memuat `python-reference.json`, `outputs.f32le` (N×48×120×11 float32 little-endian), dan `inputs.rgb` (N×192×480×3 byte RGB). Referensi mengikat versi decoder/protokol, model, ambang, hash data mentah, source Python termasuk `word_symbols.py`, serta teks/alasan decoder dan pemeriksaan kualitas setiap baris. Koordinat `selection` menggunakan `[x,y,right,bottom]`. Bila `python_cells` tersedia pada hasil diterima, role, karakter keluaran, mask, dan kotak juga dibandingkan; kasus tanpa referensi sel ditandai belum dibandingkan.

Harness memeriksa ukuran arsip, urutan baris, dan hash source sebelum replay; perbedaan menghasilkan laporan gagal dan exit code bukan nol. Laporan lama tidak dapat ditimpa. Hash seluruh modul TypeScript yang benar-benar dimuat ikut dicatat. Bukti decoder lama di bawah tetap historis dan tidak membuktikan dukungan kapital/angka.

Harness ini memutar ulang `public-word-outputs.npz` dan gambar input dari benchmark yang sudah selesai. Tidak ada pemuatan TensorFlow, model, atau inferensi ulang. Python menjalankan decoder yang sebenarnya serta fungsi pemeriksaan kualitas yang diambil langsung dari AST `word_benchmark.py`; Node menjalankan TypeScript aplikasi melalui loader transpile yang sama dengan harness geometri.

```powershell
C:\bm-envs\angelina-benchmark\Scripts\python.exe scripts/word-decoder-parity/prepare.py --run .braille-training-work/word-scan-v1-20260905/selected-v1 --data .braille-training-work/word-scan-v1-20260905/data --output .braille-training-work/word-scan-v1-20260905/selected-v1/ts-python-parity
node scripts/word-decoder-parity/compare.cjs .braille-training-work/word-scan-v1-20260905/selected-v1/ts-python-parity training/evidence/braille-word-reader-v1/failed-selected-v1-decoder-parity.json
```

Gunakan direktori laporan baru untuk kandidat berikutnya. `prepare.py` memeriksa bahwa replay Python masih sama dengan benchmark beku, menyimpan tensor little-endian float32 dan pixel RGB, serta mengikat hash semua input dan source. File mentah sekitar 322 MB tetap di direktori eksperimen yang diabaikan Git. `compare.cjs` membaca satu blok pada satu waktu dan gagal bila teks/alasan penolakan berbeda.

Ada keterbatasan arsip kandidat `selected-v1`: benchmark awal mengganti tensor dengan nol pada 29 gambar yang ditolak pemeriksaan kontras. Karena itu laporan membedakan **578 keluaran model asli** dari **29 kasus yang melewati decoder karena kualitas gambar**. Jangan menyebut 607 tensor tersimpan sebagai 607 keluaran model asli. Pemeriksaan kualitas dan hasil pipeline lengkap tetap dibandingkan pada seluruh 607 gambar memakai pixel input yang sama.

Arsip berikutnya menyimpan seluruh tensor asli dalam `outputs` dan alasan kualitas secara terpisah dalam `quality_reasons`. Harness mengenali kontrak tersebut, memeriksa alasan yang diarsipkan, dan membandingkan decoder pada semua tensor asli, termasuk gambar yang ditolak pemeriksaan kualitas runtime. `quality_bypass_cases` tetap mencatat bypass runtime; nilai itu tidak mengurangi jumlah tensor asli yang bisa diuji secara diagnostik.

Hasil 5 September 2026 pada kandidat gagal tersebut: teks dan alasan decoder cocok 578/578; pemeriksaan kualitas, teks pipeline, dan alasan pipeline cocok 607/607; tidak ada mismatch. Laporan terikat ke model SHA-256 `36fa9c2532ad526b8e3cf93a4b4fd43758aeb1b78ecfac80dee047ba732d0dc8`. Tidak ada perubahan decoder atau ambang untuk memperoleh hasil ini. Parity menunjukkan dua implementasi konsisten; kandidat tersebut tetap gagal kriteria akurasi dan tidak boleh dipromosikan.

Bukti ini berasal dari Node dan array keluaran TFLite yang sudah diarsipkan. Pengujian kamera native, perangkat fisik, dan performa aplikasi tetap terpisah.

Replay kandidat gagal `selected-v6` juga selesai: decoder cocok **607/607 keluaran asli**, dan kualitas/teks/alasan pipeline cocok **607/607**. Ada 29 bypass kualitas dan nol placeholder. Laporan `failed-selected-v6-decoder-parity.json` terikat ke model `ccf93a70ad0a7b0cda882fda2525f7d14632d682ec8188bab65ee6bb76d6b73f`; kandidat tetap gagal kriteria pengenalan dan negatif. Hash `source_sha256` pada laporan mengidentifikasi **source yang dipakai saat replay**. Source training/inferensi yang membentuk kandidat tetap diarsipkan tersendiri di `selected-v6/source`; perubahan augmentation setelah inferensi tidak dianggap sebagai source pembentukan kandidat tersebut.

## Regresi batas numerik

```powershell
node scripts/word-decoder-parity/numerical-boundaries.cjs C:\bm-envs\word-scan-torch-only\Scripts\python.exe training/evidence/braille-word-reader-v1/numerical-boundaries-after-geometry-fix.json
```

Pemeriksaan ini membangun 162 tensor sintetis yang disimpan secara sparse sebagai bit IEEE-754, lalu menjalankan decoder TypeScript dan Python sebenarnya. Kasus mencakup nilai float32 sebelum/pada/sesudah proposal `.35`, penerimaan `.4`, keyakinan titik `.7`, titik tidak aktif `.3`, proposal lemah di tengah, huruf berulang, ukuran minimum, local maximum, ROI, batas foto, jarak antarsel, susunan baris, NMS, dan overlap. Koordinat ROI `xyxy` dari benchmark juga diuji melalui konversi `xywh` yang dilakukan replay aplikasi, tanpa mengubah koordinat sumber fixture.

Regresi awal mengungkap dua perbedaan pembulatan pada batas kanan/bawah: ekspresi Python `1-right < margin` tidak selalu identik dengan ekspresi aplikasi `right > 1-margin` pada bilangan berdekatan satu ULP. Python kemudian mengikuti promosi float32 ke float64, ukuran kotak mentah, urutan operasi geometri, dan konversi ROI aplikasi. Source TypeScript dan ambang aktif tidak diubah.

Laporan awal 112/114 cocok dipertahankan di `numerical-boundaries-before-geometry-fix.json`, bersama fixture dan snapshot driver awal. Laporan setelah perbaikan mencatat **162/162 cocok**, dengan hash decoder Python `bd301c5afeb4ba2e6866371665324d2de7f0a6ee252ef76ce02a5553193d63c5`. File `.fixtures.json` pendamping mengikat semua input; laporan mengikat source kedua decoder. Tidak ada model, Torch, TensorFlow, atau GPU yang dimuat oleh regresi ini. Bukti sintetis tersebut melengkapi replay keluaran model sebenarnya dan tidak menjadi bukti akurasi pengenalan.

## Kelayakan geometri dengan anotasi ideal

```powershell
C:\bm-envs\word-scan-torch-only\Scripts\python.exe scripts/word-decoder-parity/geometry-ceiling.py --data .braille-training-work/word-scan-v1-20260905/data --output .braille-training-work/word-scan-v1-20260905/ideal-geometry-ceiling
node scripts/word-decoder-parity/geometry-ceiling.cjs .braille-training-work/word-scan-v1-20260905/ideal-geometry-ceiling training/evidence/braille-word-reader-v1/ideal-annotation-decoder-feasibility.json
node --test scripts/word-decoder-parity/geometry-ceiling.test.cjs
```

Audit ini mensimulasikan prediksi sempurna dari seluruh sel anotasi yang lengkap dalam konteks: objectness 1, enam bit 0/1, serta pusat dan ukuran kotak yang ditulis ke tensor Float32 stride 4. Sel lengkap yang memotong area pilihan tetap dimasukkan. Sel parsial di batas konteks dihitung terpisah dan tidak disimulasikan. Tabrakan slot grid dilaporkan tanpa memperbaiki anotasi. Decoder dan quality check TypeScript aktual dijalankan tanpa model, GPU, atau Android.

Hasil: decoder menerima tepat 134/135 natural, 402/405 stress, dan 604/607 eksternal. Setelah pemeriksaan kualitas RGB, jumlahnya 134, 372, dan 575. Tidak ada hasil diterima yang salah atau tabrakan grid. Penolakan geometri adalah satu `PRIMERA` dan dua variasinya karena rentang pusat vertikal; tiga kata eksternal `DHFJ`, `HHDJ`, dan `RV` ditolak karena rasio jarak pusat terhadap tinggi median masing-masing 1.9888, 2.0050, dan 1.2475 melebihi ambang 1.2. Ini tidak menyatakan fixture salah; seluruh fixture dan ambang dipertahankan.

Laporan ringkas mengikat ketiga manifest, source, RGB, dan hash seluruh hasil (`all-outcomes.json` di direktori diagnostik). Natural 135 juga termasuk dalam stress 405 sehingga kelompok tidak independen. Angka ini hanya menjelaskan kelayakan representasi anotasi ideal; tidak boleh dipakai sebagai akurasi model, bukti native, atau penyaringan benchmark.
