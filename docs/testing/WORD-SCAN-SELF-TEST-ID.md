# Uji APK pemindai Braille Indonesia dasar

Protokol untuk update 0.3.0 eksperimen internal. Catat nama APK, SHA-256, ponsel, Android, tanggal, kondisi cahaya, dan jenis sumber. Pengujian gambar publik, contoh sintetis, emulator, dan kertas timbul harus dilaporkan terpisah.

## Cara memindai

1. Pasang APK, selesaikan pengenalan, buka Pindai, dan izinkan kamera.
2. Letakkan sumber pada permukaan rata dengan titik tegak dan arah baca kiri ke kanan. Kamera tepat di atas sumber. Arahkan cahaya miring dari ujung atas kertas agar bayangan titik terlihat.
3. Arahkan satu kata atau rangkaian angka ke tengah kamera. Sertakan indikator dan ruang di kedua ujung. Bingkai luar mengikuti kelompok sel; kotak kecil mewakili setiap sel fisik. Tidak ada slider lebar.
4. Tunggu status **Mencari Braille**, **Braille terdeteksi**, kemudian **Tahan kamera, lalu tekan Pindai**. Bingkai menunjukkan deteksi posisi, bukan jaminan terjemahan benar.
5. Tekan **Pindai** satu kali. Periksa foto, kotak indikator, urutan sel, kapital, digit, dan nol awal. Pilih **Bacakan** atau **Pindai ulang**.
6. Catat setiap pengambilan, termasuk hasil salah dan penolakan, sebelum mencoba lagi. Rangkaian yang terpotong tetapi diterima adalah hasil salah, bukan keberhasilan parsial.

## Aturan hasil

Aturan mengikuti [panduan Braille dasar Indonesia](https://bakorplbbanyumas.files.wordpress.com/2013/11/membaca-menulis-braille-lengkap-ok.pdf). Tanda kapital adalah titik 6; tanda angka adalah titik 3-4-5-6.

| Kasus sumber | Hasil yang diperiksa |
| --- | --- |
| A–Z tanpa indikator, masing-masing satu sel | Huruf kecil a–z |
| Tanda kapital tunggal sebelum setiap A–Z | Huruf kapital A–Z; indikator tidak menghasilkan karakter |
| `⠁⠃⠉`, `⠠⠁⠃⠉`, `⠠⠠⠁⠃⠉` | `abc`, `Abc`, `ABC` |
| Tanda kapital tunggal di tengah kata | Hanya huruf berikutnya menjadi kapital |
| Tanda kapital ganda di tengah kata, tiga indikator, indikator tanpa isi | Seluruh hasil ditolak |
| Tanda angka diikuti setiap pola a–j | Digit 1–9 dan 0 |
| `⠼⠃⠚⠃⠋`, `⠼⠚⠚⠁` | `2026`, `001`; nol awal tetap ada di hasil/draft/TTS |
| Tanda angka saja, campuran huruf–angka, tanda negatif/desimal/pemisah | Seluruh hasil ditolak |
| 12 karakter, termasuk 12 huruf dengan masing-masing tanda kapital (24 sel) | Dapat diterima jika ukuran dan keyakinan memadai |
| Lebih dari 12 karakter atau 24 sel; rangkaian tidak muat/terlalu kecil | Ditolak |
| Satu indikator atau sel lain berkeyakinan rendah | Seluruh hasil ditolak; tidak membuang simbol lalu menerima sisa |

Jumlah sel fisik dapat lebih besar dari panjang hasil. Indikator ditampilkan sebagai kotak sel dengan keluaran kosong. Jangan mengharuskan jumlah kotak sama dengan panjang teks.

## Pelacakan dan kamera fisik berikutnya

| Percobaan | Yang harus diamati |
| --- | --- |
| Satu sel, kata berulang `mama`/`kakak`, indikator di depan | Kotak mengikuti seluruh kelompok termasuk indikator |
| Dua kata atau beberapa baris | Memilih satu kelompok terdekat tengah; tidak menggabungkan semua baris |
| Geser kamera pelan/cepat, pindahkan sasaran melewati batas area pencarian | Kotak mengikuti posisi/ukuran tanpa lompatan target berulang |
| Kandidat baru lebih dekat tengah | Pergantian setelah dua pengamatan konsisten |
| Target hilang atau tertutup | Pindai nonaktif setelah 500 ms; bingkai hilang setelah 800 ms |
| Ujung kata/indikator keluar bidang kamera | Ditolak; tidak menerima potongan kata |
| Tekan Pindai ketika tracking berjalan; tekan dua kali cepat | Satu foto dan satu hasil, tanpa crash atau model berebut akses |
| Pindai ulang | Hasil, foto, dan target lama hilang; pencarian baru dimulai |
| Rotasi portrait/landscape, perubahan layout, kembali dari Home/lock | Tidak memakai koordinat atau hasil sesi lama |
| Fokus sentuh; senter hidup/mati; kamera tanpa senter | Preview tetap berjalan dan kontrol yang tersedia dapat digunakan |
| Pembukaan pertama setelah force-stop; izin ditolak/dicabut | Startup stabil atau pesan pemulihan yang dapat ditindaklanjuti |
| Kertas kosong, meja, huruf cetak biasa, tekstur, blur/pantulan | Tidak menghasilkan hasil yang diterima secara keliru |

Bandingkan posisi kotak pada preview dan foto memakai penanda independen; catatan matriks saja tidak membuktikan kotak sel benar. Mirror/rotasi dan kamera dari beberapa produsen perlu bukti perangkat tersendiri.

## Suara, aksesibilitas, dan kinerja

- Gunakan suara Bahasa Indonesia yang tersedia lokal. Uji mode pesawat tanpa Metro. Cocokkan teks yang dikirim ke TTS dan dengarkan pengucapan digit; penerimaan event bukan bukti audio terdengar.
- Dengan TalkBack, pastikan status hanya diumumkan saat berubah. Bingkai bergerak tidak menerima fokus. Tombol Pindai, Bacakan, Pindai ulang, senter, dan Bantuan dapat dicapai.
- Atur font 200%; seluruh petunjuk dan tombol tetap dapat dicapai dengan menggulir. Target sentuh minimal 48 dp.
- Pindai ulang/keluar saat TTS berbicara harus menghentikan pembacaan lama.
- Uji 10 menit: catat FPS preview, latensi tracking, waktu tombol hingga hasil, memori, suhu, crash/ANR. Target preview minimal 24 FPS dan p95 pengambilan hingga hasil maksimal 2 detik. Pisahkan angka emulator dari ponsel fisik.

## Catatan setiap pengambilan

| Nomor | Sumber/split | Teks sebenarnya | Sel fisik | Hasil/alasan penolakan | Tepat seluruhnya? | Kondisi |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | kertas fisik | Abc | 4 | | | |
| 2 | kertas fisik | 001 | 4 | | | |

**Keberhasilan hasil utuh** = tepat seluruhnya / semua pengambilan. **Ketepatan hasil diterima** = tepat seluruhnya / pengambilan yang menghasilkan teks. Laporkan terpisah untuk huruf kecil, kapital tunggal, seluruh kata kapital, angka, dan negatif. Sampel sintetis atau susunan ulang foto publik tidak membuktikan akurasi kertas Indonesia fisik. APK ini tetap eksperimen; persyaratan produksi ada pada [GO checklist](../release/GO-CHECKLIST.md).