# Penerjemah Braille dua arah

Tab utama berurutan **Beranda, Pindai, Penerjemah, Setelan**. Penerjemah bekerja
lokal tanpa kamera, jaringan, pelatihan tambahan, atau dependency native baru.
Tampilan memakai komponen native dengan Braille di atas dan teks di bawah.

## Aturan dan batas

Pemetaan alfabet, kapital, angka, dan sembilan tanda baca mengikuti tabel
Indonesia pada [World Braille Usage, edisi ketiga, halaman 70](https://www.perkins.org/wp-content/uploads/2021/07/world-braille-usage-third-edition.pdf#page=86).
Konstanta yang sudah ada di `src/ml/word-model-contract.ts` dan
`src/ml/word-symbols.ts` hanya dibaca. Decoder kamera tetap terpisah.

| Masukan | Perilaku |
| --- | --- |
| `a` sampai `z`, `A` sampai `Z` | Kapital memakai indikator sebelum setiap huruf besar |
| `001`, `Aku punya 12 buku`, `(001)` | Angka bulat tanpa tanda, nol awal dipertahankan |
| `. , ? ! : ; - ( )` | Sembilan tanda baca; kurung berpasangan berurutan tanpa tingkat bersarang |
| Spasi dan LF | Jumlah spasi dan baris baru dipertahankan |
| `A1`, `1a`, `-10`, `12,5`, `12.5` | Belum valid, disertai lokasi dan penjelasan |
| Emoji, aksen, pola tidak dikenal, indikator menggantung | Masukan dipertahankan untuk diperbaiki |

Indikator kapital ganda di awal kata juga diterima decoder. Mode angka berakhir
di spasi, baris baru, atau tanda baca. Pola kurung buka/tutup identik sehingga
decoder membacanya bergantian; sebuah kurung tanpa pasangan belum valid.

`encodeText()` dan `decodeCells()` adalah fungsi TypeScript murni pada
`src/features/translator/codec.ts`. Hasil memuat `value`, `valid`, dan `issues`
dengan indeks mulai nol: codepoint masukan untuk teks, indeks item untuk Braille.
UI menampilkan posisi mulai satu. Dokumen membedakan `{ kind: 'cell', mask }`,
`{ kind: 'space' }`, dan `{ kind: 'newline' }`. Mask memakai enam bit dengan titik
1 sebagai bit terendah. Mask nol adalah sel belum selesai, bukan spasi.

Batas valid adalah 200 codepoint teks dan 400 item dokumen, termasuk pemisah.
Teks tempelan berlebih tidak dipotong. Nilai parsial dari codec tidak boleh
dipakai sebagai hasil valid ketika `valid` bernilai `false`.

## Editor dan draf

Pilih sel dalam pratinjau lalu ubah tombol titik berurutan `1 4 / 2 5 / 3 6`.
Keenam tombol berperan sebagai checkbox dengan nomor, status aktif, dan area
sentuh minimal 56 dp. Tambah sel, Spasi, dan Baris baru menyisipkan setelah
pilihan; Hapus sel menghapus pilihan tersebut.

Hasil valid menjadi masukan mode tujuan. Perpindahan dari masukan belum valid
mempertahankan kedua draf, termasuk saat kembali tanpa mengedit mode tujuan.
Draf berada dalam memori proses, tetap tersedia saat berganti tab, dan tidak
ditulis ke penyimpanan. Hapus semua mengosongkan mode aktif. Ekspor, riwayat,
dan Bacakan tidak termasuk fitur ini.

## Bukti implementasi

Log pemeriksaan, hash sumber sebelum perubahan, staging, build, dan pemeriksaan
emulator disimpan pada `output/translator-20260911/`.

Pemeriksaan sumber `model:verify`, `release:config:verify`, lint, typecheck, dan
`test:coverage` lulus, dengan 44 suite dan 1.002 tes. Coverage seluruh aplikasi:
92,06% statements, 87,33% branches, 91,31% functions, dan 93,14% lines.
Empat peringatan lint lama hanya terdapat pada harness paritas model; kode
penerjemah tidak menghasilkan peringatan lint.

Penerjemah memiliki 100 tes codec dan 25 tes editor/draf/keyboard. Codec mencapai 100%
coverage statements, branches, functions, dan lines. Kasusnya mencakup alfabet,
angka, sembilan tanda baca, 64 pola sel, 4.096 kombinasi dua sel, kapital
campuran, nol awal, kurung lintas kata, masukan belum lengkap, batas panjang,
dan tempelan 20.000 karakter tidak didukung tanpa pemotongan. Tes interaksi
mencakup pembaruan langsung, sel identik pada indeks berbeda, pemisah, hapus,
perpindahan mode, draf setelah remount, dan dimensi 320 dp dengan font 200%.

Hasil ini menguji aturan digital dan interaksi editor; angka benchmark kamera
tidak digunakan sebagai ukuran ketepatan fitur ini. Bukti visual dan TalkBack
APK dicatat terpisah dari tes komponen.

APK awal mengungkapkan kotak ketikan tertutup keyboard ketika pratinjau bertambah.
Perbaikan menyesuaikan tinggi area gulir Android dan menjaga input yang sedang
difokuskan terlihat setelah keyboard atau ukuran konten berubah. Sel pratinjau
yang tidak berubah juga tidak dirender ulang. Regresi keyboard lulus pada
konfigurasi Android maupun konfigurasi tes default.

Pada APK kedua, ketikan `Aku  007` diikuti baris baru `BACA!` menghasilkan tepat
20 item Braille dan kembali menjadi teks yang sama, termasuk dua spasi dan nol
awal. Kotak ketikan terlihat di atas Gboard; draf, hasil, dan pilihan sel tetap
ada setelah membuka Setelan lalu kembali ke Penerjemah. Wi-Fi dan data seluler
dimatikan; hash APK terpasang dicocokkan sebelum pemeriksaan.

## APK akhir dan layar kecil

Artefak yang diserahkan adalah
[`BrailleMe-0.3.0-translator-2026-09-11-r3.apk`](../../output/android/BrailleMe-0.3.0-translator-2026-09-11-r3.apk),
berukuran 136.623.168 byte. SHA-256:

```text
e35bfe87d08a930c22932bf73ec2f3734c4bff4cada6c0b65d89701948e7e3bc
```

APK internal memakai sertifikat Android Debug, package `com.brailleme.app`,
versi 0.3.0/code 3, minimum Android 10 (API 29), dan ABI `arm64-v8a` serta
`x86_64`. Pemeriksaan integritas lulus 27/27, termasuk tanda tangan, CRC arsip,
alignment ELF 16 KB, metadata, dan hash model serta fixture yang dibundel.
APK terpasang pada emulator API 36 dicocokkan dengan hash tersebut; pengujian
dijalankan tanpa Metro, dengan Wi-Fi dan data seluler mati.

Pada ukuran 320 x 640 dp dan font 200%, empat tab, ikon, dan label terlihat utuh.
Ketikan `Aku 007` diikuti baris baru `BACA!` menghasilkan 19 item Braille.
Kotak ketikan tetap berada di atas keyboard. Editor enam titik mempertahankan
dua kolom dan dapat digulir sampai baris ketiga serta kontrol di bawahnya;
area tombol titik yang terukur sekitar 117 x 140 dp.

Revisi akhir hanya mengubah ukuran wadah ikon tab untuk font besar dan tes tab
terkait. Typecheck, lint, dan tiga tes tab dijalankan kembali dan lulus.
Hash 251 berkas lainnya sama dengan sumber pada pemeriksaan lengkap 1.002 tes,
termasuk seluruh codec, editor, draf, dan perbaikan keyboard. Hubungan bukti ini
dicatat dalam `validation-r3/source-preflight.json` dan
`final-source-audit.json`; hasil pemeriksaan lengkap tidak diklaim sebagai
eksekusi ulang setelah revisi ikon.

Ketiga APK penerjemah, lima APK sebelum tugas ini, serta 80 aset terlindungi
tetap utuh. Seluruh 253 berkas sumber yang dipakai build cocok dengan staging
dan repository. Build memakai kembali `C:\bm-symbols-apk-20260908`; tidak ada
folder staging baru, duplikasi project, atau perubahan pada aset training.

Pemeriksaan ini merupakan bukti APK internal pada emulator. Pengoperasian
perangkat fisik dan ketepatan kamera pada kondisi nyata belum diuji oleh
pekerjaan fitur penerjemah ini.

## Batas verifikasi TalkBack

Hierarki Android memuat keenam tombol sebagai checkbox bernomor dengan label
aktif/tidak aktif dan nilai `checked` yang sesuai. TalkBack terpasang dan
terhubung, dengan touch exploration aktif. Namun, injeksi `adb input tap`
mengubah titik langsung tanpa memindahkan fokus TalkBack; tangkapan layar
tidak memperlihatkan indikator fokus. Tidak ada UIAutomator yang dijalankan
selama percobaan sentuhan tersebut.

Karena jalur injeksi ini tidak membuktikan gestur TalkBack, pengoperasian enam
titik melalui fokus dan ketuk dua kali **belum dinyatakan lulus**. Percobaan
dan struktur aksesibilitas disimpan pada `talkback-r3/`. Pengujian sentuhan
dan pengucapan TalkBack pada perangkat fisik masih diperlukan; hasil tes
komponen dan sentuhan biasa tidak menggantikan bukti tersebut.

## Pemeriksaan regresi kamera

Kamera native dapat dibuka dan menampilkan fixture pada APK akhir, tanpa
Metro atau jaringan. Kesiapan model/self-test disimpulkan dari berhasilnya
melewati gerbang aplikasi menuju kamera; ini bukan log keberhasilan self-test
yang berdiri sendiri.

Dua percobaan terbatas, termasuk satu setelah proses aplikasi dimulai ulang,
berhenti pada pemeriksaan bingkai otomatis sebelum prediksi foto berhasil.
`BRAILLE_WORD_PHOTO_METRICS` mencatat `status: error` pada 2 ms dan 5 ms;
layar meminta pengguna memilih Pindai ulang karena bingkai belum dapat
mengikuti Braille. Hasil fixture yang diharapkan, `Yys`, tidak muncul.
Percobaan kedua tidak menjalankan UIAutomator saat pratinjau aktif.

Karena itu, keberhasilan pindai kamera **belum dinyatakan lulus** untuk sesi
emulator APK akhir ini. Bukti tersimpan dalam `camera-r3/`, termasuk kegagalan
pertama dan satu percobaan ulang. Decoder, model, dan aset training tidak
diubah untuk mengatasi temuan di luar implementasi penerjemah ini. Bukti
tersebut tidak menentukan akurasi kamera fisik atau ketepatan codec digital.

Pengaturan ukuran layar, font, dan aksesibilitas emulator dipulihkan ke nilai
awal. Emulator milik tugas ditutup secara tertib setelah bukti dikumpulkan.
Ringkasan beserta hash laporan terkait tersedia pada
[`final-qa.json`](../../output/translator-20260911/final-qa.json). APK dapat
dipakai untuk menguji fitur penerjemah; seluruh kriteria penerimaan native
belum terpenuhi karena batas verifikasi TalkBack dan hasil pindai di atas.
