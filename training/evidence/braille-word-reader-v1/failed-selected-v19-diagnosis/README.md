# Diagnosis kandidat BatchNorm run24

Kontrol v15 tetap dipilih oleh seleksi v19. Keduanya menghasilkan **357/405 kata benar dan 357/362 hasil diterima benar**, dengan pembagian natural/stress identik. Aturan seri yang telah ditetapkan memilih kontrol.

Hanya dua keputusan kata berubah: ZWUKOW ditolak setelah confidence W turun dari 0,761 menjadi 0,644; NA diterima setelah confidence N naik dari 0,676 menjadi 0,704. Pola kedua sel tersebut tetap sama. Total deteksi 1.522 dan pasangan GT 1.518 tidak berubah. Pola salah pada pasangan GT naik dari delapan menjadi sembilan; mean IoU naik 0,002618 dan galat pusat turun 0,04734 piksel. Ini perubahan numerik nyata tanpa peningkatan kata utuh.

**111 GT tidak berpasangan mencakup 103 sel pada 30 gambar yang ditolak kualitasnya dan diberi keluaran nol. Hanya delapan GT tidak berpasangan pada gambar yang lulus kualitas.** Angka 111 tidak boleh disebut jumlah sel terlewat oleh detector. Lima hasil diterima yang salah tetap berupa kata lebih pendek; PTICA menjadi RC juga memuat salah pola T menjadi R.

[Audit independen](independent-audit/independent-aggregate-audit.json) memeriksa hash dan menghitung ulang agregat 405 baris tanpa mengulang inferensi atau pencarian ambang. Set ini terdiri atas 135 crop natural dan 270 varian stress yang berkorelasi, bukan 405 foto independen. Penolakan kualitas tetap dihitung gagal.

[Laporan lengkap terkompresi](comparison.json.gz) mempertahankan JSON asli persis. SHA-256 setelah dekompresi: `0d2b9dd0126f0a25d3deadfc3fbc0ae79de059f57fb44419267f2658f7b3fc48`; gzip: `c5dd20ba5488d82041bcdcc089e7fe520ccd4f71104621083ead64982ed2a200`. [Dokumen asli](original-documents/FINDINGS.md.txt) dan [tabel asli](original-documents/README.md.txt) dipertahankan tanpa perubahan. Metadata cache, seleksi, sumber evaluator, dan audit berada dalam [indeks hash](archive-index.json).

Tidak ada inferensi baru, evaluasi 607 eksternal, perubahan ambang, aktivasi model, atau APK dari pengarsipan ini. Target 75%/95%, 0/44 negatif, dan seluruh 14 kasus penolakan tetap berlaku; hasil eksternal masih milik kandidat v15 yang gagal.
