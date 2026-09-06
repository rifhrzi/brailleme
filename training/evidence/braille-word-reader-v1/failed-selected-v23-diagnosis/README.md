# Diagnosis cache selected-v23

Run27 continued horizontal-aspect weights with deliberately repeated sample realizations. This comparison uses the best new eligible checkpoint and the unchanged control at equal thresholds on frozen held validation. Eleven correct words are lost and six gained; checkpoint selection keeps the control.

Kontrol: 357/405 kata benar, 362 diterima. Kandidat: 352/405 kata benar, 355 diterima. Kata benar yang hilang: 11; kata benar baru: 6.

Pencocokan kotak bersifat diagnosis perkiraan, bukan AP. Anotasi tidak berpasangan mencakup sel pada gambar yang ditolak kualitasnya dan diberi keluaran nol; angka itu tidak boleh seluruhnya disebut sel yang terlewat oleh detector pada gambar yang lulus kualitas.

[Laporan lengkap](comparison.json.gz) mempertahankan JSON asli, SHA-256 setelah dekompresi `fcfd028833d1dd93408b28e4fd91073c4b32aeac7494bd4c8e932928f076979f`. [Indeks hash](archive-index.json) mengikat salinan metadata dan sumber evaluator. Tensor cache tetap di direktori eksperimen; hash asalnya tercatat dalam laporan.

Pengarsipan ini tidak menjalankan inferensi baru, pencarian ambang, evaluasi eksternal, aktivasi model, pembuatan APK, atau pengujian fisik. Syarat penerimaan tetap berlaku.
