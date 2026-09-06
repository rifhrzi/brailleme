# Diagnosis cache selected-v21

Kandidat variasi lebar memulihkan enam pasangan sel dan mengurangi pola salah dari delapan menjadi tujuh. Pada 1518 pasangan yang sama, mean IoU naik 0,042902 dan rasio lebar turun 0,069640 mendekati anotasi. Tujuh kata benar hilang (lima karena confidence titik dan dua terpotong), sementara lima kata membaik. Perbaikan geometri ini belum membuat kandidat unggul dalam pembacaan kata utuh.

Kontrol: 357/405 kata benar, 362 diterima. Kandidat: 355/405 kata benar, 358 diterima. Kata benar yang hilang: 7; kata benar baru: 5.

Pencocokan kotak bersifat diagnosis perkiraan, bukan AP. Anotasi tidak berpasangan mencakup sel pada gambar yang ditolak kualitasnya dan diberi keluaran nol; angka itu tidak boleh seluruhnya disebut sel yang terlewat oleh detector pada gambar yang lulus kualitas.

[Laporan lengkap](comparison.json.gz) mempertahankan JSON asli, SHA-256 setelah dekompresi `870999502ba0f02d65391a1d02d106235b28373dd3703b83fe84d256f1c24f6a`. [Indeks hash](archive-index.json) mengikat salinan metadata dan sumber evaluator. Tensor cache tetap di direktori eksperimen; hash asalnya tercatat dalam laporan.

Pengarsipan ini tidak menjalankan inferensi baru, pencarian ambang, evaluasi eksternal, aktivasi model, pembuatan APK, atau pengujian fisik. Syarat penerimaan tetap berlaku.
