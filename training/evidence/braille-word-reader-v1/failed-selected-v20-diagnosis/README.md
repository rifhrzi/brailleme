# Diagnosis kandidat fitur awal run25

Seleksi v20 mempertahankan kontrol v15. Kandidat baru menghasilkan 356/405 kata benar dan 356/361 hasil diterima benar; kontrol menghasilkan 357/405 dan 357/362. Hanya ZWUKOW berubah dari benar menjadi ditolak karena confidence titik; tidak ada kata benar baru.

Jumlah pasangan anotasi 1.518, anotasi tidak berpasangan 111, dan deteksi tambahan empat tetap. Angka 111 mencakup sel pada 30 gambar yang ditolak kualitasnya dan diberi keluaran nol; angka ini tidak menunjukkan 111 sel terlewat pada gambar yang lulus kualitas. Pencocokan kotak bersifat diagnosis perkiraan, bukan AP.

[Laporan lengkap](comparison.json.gz) mempertahankan JSON asli, SHA-256 setelah dekompresi `3d071cb913cf88736322b75d0dac23a86b48df1ad76d980e1dfd7ac9ce171417`. [Tabel asli](original-README.md.txt) dan sumber evaluator diikat oleh [indeks hash](archive-index.json). Tensor cache tetap di direktori eksperimen dan hash asalnya tercatat dalam laporan; tensor tersebut tidak disalin ke arsip ini.

Tidak ada inferensi baru, pencarian ambang, penggunaan 607 blok eksternal, aktivasi model, APK, atau bukti perangkat fisik dari diagnosis ini. Syarat penerimaan tetap berlaku.
