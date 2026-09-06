# Pemeriksaan cermin horizontal

[Laporan utama](preflight.json) mempertahankan konfigurasi, protokol, source snapshot, dan audit sebelum training run22. Data dan parent v15 sama; perubahan perilaku hanya pencerminan horizontal dengan peluang 0,5 serta pertukaran kolom titik yang sesuai. Tidak ada rotasi 180 derajat atau pembalikan intensitas.

Audit memeriksa seluruh 63 pola nonkosong, piksel, kotak, label, 252 kasus kelangsungan titik melalui transformasi, serta 63 kasus label parsial yang keenam bitnya tidak diketahui. Lima batch nyata mencakup 30 contoh per mode dari 12 kelompok sumber. Mode tanpa cermin identik dengan run20; mode cermin mempertahankan urutan dan seluruh pengambilan angka acak untuk augmentasi selanjutnya.

Protokol mengikat audit **sebelum launch** dengan hash `efd565a8d372c141a5f5e3456a855f4e59d0486da75a9f09ab807e19d52a62c6`. Salinan persisnya dipertahankan. Tambahan pemeriksaan negatif dan reproducer statistik frekuensi ditulis sesudah launch dan disimpan di `post-launch-addenda`; keduanya tidak mengganti audit awal atau mengubah training.

[Pemeriksaan verifier pada arsip asli](verification/real-mirror-preflight-verifier-check.json) lulus. Source dan hasil pemeriksaan disimpan untuk peninjauan; skrip replay saat itu juga menggunakan ancestry run20 dan arsip data yang disebutkan di laporannya. Ini bukan klaim bahwa seluruh ancestry kandidat akhir sudah dirakit.

Semua bukti di sini memeriksa ketepatan transformasi dan provenance. Akurasi model terlatih, dukungan runtime Android, APK utama, dan kertas fisik masih memerlukan bukti terpisah. [Indeks hash](archive-index.json) mencakup seluruh file yang dipertahankan.
