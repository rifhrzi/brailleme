# Rencana arsip runtime final

Belum ada penyalinan. `runtime-curation-plan-v1.json` mencatat **97 basename input unik**, **430 berkas**, perkiraan **53.496.168 byte** salinan, dan **266.629.238 byte** APK yang hanya diindeks di lokasi asal. Ukuran belum mencakup `index.json`, berkas plan yang ikut dipertahankan, atau bukti UI/single-feed tambahan. Dump refleksi `emulator-runtime.json` 75 MB tidak dipilih; bobot/tensor model duplikat juga ditolak.

Tujuan yang direncanakan adalah `training/evidence/braille-word-reader-v1/native-symbols-runtime-20260908`. Curator asli tetap melakukan pemeriksaan path/reparse point, batas 10 MB per berkas, pengecualian package APK, deteksi perubahan sumber, serta verifikasi hash salinan. Skrip baru hanya menyiapkan input dan menjalankan curator itu setelah mode `copy` secara eksplisit dipanggil. Semua input harus mempunyai basename unik; tidak ada folder parent stage/output yang disalin seluruhnya.

| Kelompok | Sumber utama dalam rencana |
| --- | --- |
| APK final dan pembangunan | `apk-integrity-v6`, `build-20260908T173305Z`, `checks-1788888766290` dengan 671 tes, `apk-delivery.json`, `replaced-delivery-v2`; APK hanya diindeks |
| Runtime lengkap | `runtime-600-v6-complete`, seluruh `actions-600-v6`, `timing-600-v6`, analisis runtime dan tinjauan independen |
| Preview awal dan native | Analisis/sampel preview awal, ART resizer, verifier model 42 kasus, identitas host ringkas |
| Geometri dan penyebab kegagalan | Source binding v4/v5/v6, review piksel/overlay/proyeksi/localization, screenshot/log v6, reproduksi Worklets, bytecode/JAR/sumber/aljabar CameraX |
| Input kamera | `calibrated-feeds-v1`, `labelled-feeds-v1`, input startup asli/besar, manifest/hash/kalibrasi, peluncuran emulator dan boot |
| Helper dan prosedur | Source/DEX/report UiAutomation v3, XML native pertama, harness dan 17 tesnya, collector/verifier/analisis, prosedur dan skrip kurasi |
| Riwayat yang harus tetap terlihat | Prepare v3/v4/v5, crash worklet, collector awal tanpa summary, cold-start dengan stock dump yang gagal, boot ANR, helper UiAutomation v1, dan PNG crunch |

Bukti background, font200, cold-start selesai, rotasi, serta setiap single-feed ditambahkan oleh root setelah UI QA selesai. Daftar dasar terbaru juga mempertahankan empat TSX kamera/screen asli milik pengguna, `pre-existing.patch`, dan `protected-hashes.json` sebagai file individual; bukan seluruh folder baseline atau aset model. Audit preservasi final, audit sumber lokalisasi, dan laporan approximate replay JSON ikut disertakan; RGB/tensor besarnya tetap tidak disalin. Helper rotasi dan tesnya juga menjadi input. Rencana v1 tetap merupakan snapshot daftar sebelumnya; tambahan tersebut serta perubahan akhir metadata delivery memerlukan **plan baru**, tidak menimpa v1.

```powershell
$wordCurationPython = 'C:\bm-envs\word-symbols-tensorflow\Scripts\python.exe'
$wordCurationScript = 'output/update-symbols-tracking-20260908/prepare-runtime-curation.py'
& $wordCurationPython $wordCurationScript prepare --plan output/update-symbols-tracking-20260908/runtime-curation-plan-final.json --add-input C:\bm-symbols-apk-20260908\evidence\actions-background-v6 --add-input C:\bm-symbols-apk-20260908\evidence\actions-cold-complete-v6 --add-input C:\bm-symbols-apk-20260908\evidence\actions-font200-v6 --add-input C:\bm-symbols-apk-20260908\evidence\actions-font200-scroll-v6
```

Ulangi `--add-input` untuk folder kasus final lainnya. Perintah `prepare` hanya membaca/menghitung hash dan menulis plan/companion proposal. Tiga tes offline memeriksa konflik basename, pengecualian APK/model, dan perubahan sumber; tidak menjalankan copy arsip atau ADB.

**Hanya setelah root menginstruksikan penyalinan final**, jalankan:

```powershell
& $wordCurationPython $wordCurationScript copy --plan output/update-symbols-tracking-20260908/runtime-curation-plan-final.json
```

Mode copy memeriksa ulang seluruh tree/hash terhadap plan; drift menghentikannya sebelum curator membuat tujuan. Folder tujuan harus belum ada. Curator tidak menulis `index.json` lengkap jika pemeriksaan gagal. Plan final ikut disalin sebagai input agar pemilihan sumber dapat diaudit.

`runtime-curation-plan-v1-retention-proposed.json` adalah proposal pendamping retensi, bukan indeks final. Ia menunjuk `training/checkpoints/retained-models-20260908.json` dengan SHA-256 **29890d735d734184135c6e725a5065e46c67c827678bf66cb04534e9c53843ed**, yang tetap tidak berubah. Setelah arsip selesai, buat pendamping final baru yang mengikat hash aktual `index.json`; pertahankan seluruh file dari indeks model lama dan arsip runtime baru. Tidak ada izin penghapusan, penggantian model, atau kesimpulan penerimaan runtime/perangkat fisik dari kurasi ini.
