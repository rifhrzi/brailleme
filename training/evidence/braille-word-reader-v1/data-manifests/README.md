# Manifest data yang dibekukan

Arsip `.json.gz` mempertahankan byte JSON asli secara utuh. [Indeks](index.json)
mencatat lokasi kerja semula, ukuran, SHA-256 hasil dekompresi, serta SHA-256
arsip. Seluruh arsip telah didekompresi dan dibandingkan byte demi byte saat
dibuat. Tidak ada foto dataset di direktori ini.

Simpan hasil dekompresi sesuai `originalWorkRelativePath` di direktori kerja
training baru, lalu cocokkan SHA-256 sebelum merekonstruksi piksel dari
AngelinaDataset pada revisi yang tercatat. Jangan menimpa fixture eksperimen
yang sudah dibekukan. Path gambar pada manifest merujuk pada direktori kerja
training dan bukan path gambar yang dibundel ke aplikasi.

Manifest training awal, tambahan mask 16, serta cakupan seluruh 69.317 sel
disimpan terpisah. Manifest validation natural, variasi kontras/blur, dan
eksternal juga terpisah. Menyimpan data ini tidak menyatakan bahwa setiap model
telah mempelajari seluruh versi data atau memenuhi kriteria akurasi; cocokkan
hash dengan konfigurasi run yang bersangkutan.
