# Uji APK pemindai satu kata

Gunakan APK internal beserta SHA-256 yang tercantum dalam laporan penyerahan. Catat model ponsel, versi Android, nama APK, tanggal, kondisi cahaya, dan jenis kertas. Hasil pengujian publik/offline serta emulator dilaporkan terpisah dari pengujian kertas timbul berikut.

## Cara memindai

1. Pasang APK, selesaikan pengenalan awal, lalu buka Pindai dan izinkan kamera.
2. Letakkan kertas Braille dasar A–Z pada permukaan rata. Arah baca titik harus normal dari kiri ke kanan. Arahkan cahaya miring dari ujung atas kertas, yaitu tepi yang paling jauh dari Anda, agar bagian terang dan bayangan titik terlihat. Posisikan kamera tepat di atas kata.
3. Masukkan **seluruh satu kata** dalam satu baris ke bingkai. Atur **Lebar bingkai** agar kata atau baris tetangga tidak ikut masuk. Sisakan sedikit ruang di sekitar titik terluar.
4. Tahan ponsel. Bila perlu coba senter dan periksa apakah bayangan titik tetap terlihat, lalu tekan **Pindai kata** satu kali.
5. Periksa foto, kotak sel, dan kata yang muncul. Pilih **Bacakan** atau **Pindai ulang**.
6. Bila ditolak, catat penolakan dahulu sebelum mencoba lagi. Percobaan ulang tetap merupakan pengambilan baru dalam laporan akurasi.

Petunjuk arah cahaya mengikuti [panduan pembuat AngelinaDataset](https://angelina-reader.com/help/). Pengujian mandiri tetap perlu mencatat kondisi sebenarnya; kesesuaian petunjuk tidak menjamin akurasi pada ponsel atau kertas Anda.

## Daftar percobaan

| Kasus | Hasil yang diperiksa |
| --- | --- |
| Huruf tunggal A–Z, masing-masing beberapa pengambilan | Huruf sesuai sumber; tidak selalu C, H, atau J |
| MAMA dan KAKAK | Huruf berulang tetap ada dan berurutan |
| Kata sepanjang 1 sampai 12 huruf | Jumlah kotak sel sama dengan panjang hasil |
| Kata baru setelah hasil berhasil | Hasil baru mengganti seluruh hasil sebelumnya |
| Dua tekanan cepat Pindai kata | Hanya satu foto/proses; tidak ada hasil ganda |
| Pindai ulang saat hasil muncul | Hasil dan foto lama hilang; Bacakan belum aktif |
| Huruf di tepi atau sebagian titik keluar bingkai | Ditolak; bingkai perlu diperbaiki |
| Dua baris atau dua kata dalam bingkai | Ditolak bila terdeteksi sebagai susunan di luar satu kata |
| Celah satu sel di tengah kata | Ditolak bila celah terdeteksi; tidak dilengkapi kamus |
| Lebih dari 12 sel, pola angka/tanda baca, Braille singkatan | Ditolak; jangan dianggap sebagai terjemahan yang didukung |
| Kertas kosong, meja, tulisan biasa, pola tekstur | Tidak menghasilkan kata yang diterima |
| Cahaya redup, pantulan, blur, miring, jarak berubah | Catat keberhasilan/penolakan; perbaiki kondisi sesuai petunjuk |
| Keluar layar, Home, lock, kembali saat pemrosesan | Tidak crash dan tidak menampilkan/membacakan hasil lama |
| Izin kamera ditolak atau dicabut melalui Settings | Petunjuk pemulihan muncul; aplikasi tetap dapat digunakan |

Pemeriksaan celah tidak menjamin bahwa semua sel yang sama sekali gagal dideteksi, khususnya sel di ujung kata, akan diketahui. Catat hasil pendek yang keliru sebagai **diterima tetapi salah**, bukan sebagai keberhasilan parsial.

## Offline dan aksesibilitas

- Pastikan suara Bahasa Indonesia lokal telah tersedia melalui pengaturan suara Android. Aktifkan mode pesawat, tutup aplikasi sepenuhnya, buka lagi, lalu ulangi pemindaian dan Bacakan tanpa Metro.
- Dengan TalkBack, periksa urutan fokus, pengaturan lebar bingkai, status pemrosesan, hasil, penolakan, Bacakan, dan Pindai ulang. Tidak boleh ada pengumuman berulang yang menghalangi navigasi.
- Atur ukuran font 200%. Pastikan seluruh petunjuk serta tombol dapat dicapai dengan menggulir.
- Saat suara sedang dibacakan, pindai ulang, keluar layar, atau ubah pengaturan suara. Pembacaan lama harus berhenti.
- Gunakan pemindai selama 10 menit; catat panas berlebih, keterlambatan, kamera macet, crash, atau suara yang keluar tanpa diminta.

## Format hasil

Salin tabel ini untuk setiap pengambilan. Jangan hanya mencatat pengambilan yang berhasil.

Tersedia juga [template CSV](WORD-SCAN-RESULTS-TEMPLATE.csv) dengan contoh kata sepanjang 1–12 huruf. Ganti contoh sesuai kertas yang tersedia. Setiap pengambilan, termasuk pengulangan kata yang sama, mendapat baris baru. Isi status `diterima` atau `ditolak`; biarkan hasil aplikasi kosong untuk penolakan dan tulis petunjuk yang muncul pada kolom alasan.

| Nomor | Kata sebenarnya | Panjang | Hasil aplikasi atau alasan ditolak | Benar seluruhnya? | Kondisi |
| --- | --- | --- | --- | --- | --- |
| 1 | MAMA | 4 | | | |
| 2 | KAKAK | 5 | | | |

**Keberhasilan kata utuh** = jumlah hasil tepat seluruhnya ÷ semua pengambilan. Penolakan masuk penyebut sebagai gagal.

**Ketepatan hasil diterima** = jumlah hasil tepat seluruhnya ÷ semua pengambilan yang menghasilkan kata. Hasil salah dengan satu huruf berbeda tetap salah.

Laporkan kedua angka menurut panjang kata. Pisahkan kertas timbul, gambar cetak/layar, negatif, serta variasi cahaya. APK internal dan pengujian mandiri ini belum menggantikan seluruh persyaratan produksi pada [GO checklist](../release/GO-CHECKLIST.md).
