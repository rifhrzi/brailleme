# Physical Validation Pack

Folder ini mendefinisikan data yang harus dikumpulkan sebelum model dapat diberi status `verified`. Jangan memakai kembali gambar training, diagnostic, atau gambar yang pernah dilihat saat memilih threshold.

## Struktur input notebook

```text
braille-physical-holdout/
  holdout_manifest.csv
  holdout/
    a/
    b/
    ...
    z/
  negative/
    empty/
    non-braille/
    partial-cell/
    multiple-cells/
    blur/
    off-center/
```

Gunakan `holdout_manifest.template.csv` sebagai header. Setiap gambar holdout wajib memiliki label, sumber fisik, sesi pengambilan, perangkat, material, pencahayaan, dan SHA-256. Nama file harus unik. Minimum release adalah 260 gambar—10 per kelas—dengan variasi sumber dan sesi yang benar-benar independen.

Gunakan `negative_manifest.template.csv` untuk mencatat scene negatif. Notebook membaca semua gambar di folder `negative`; manifest negatif dipakai untuk provenance dan review. Minimum awal adalah 12 scene dan nol false accept pada confidence 0,90.

## Aturan anti-leakage

1. Bekukan model, threshold, commit, dan acceptance criteria sebelum membuka holdout.
2. Jangan gunakan hasil holdout untuk tuning. Jika model atau threshold berubah setelah hasil dilihat, kumpulkan holdout baru.
3. Satu benda Braille yang sama pada augmentasi atau foto beruntun tetap satu `source_id`.
4. `capture_session` harus membedakan sesi nyata, bukan hanya nama file.
5. Simpan file mentah read-only dan hitung SHA-256 sebelum training/evaluasi.

## Eksekusi

Tambahkan dataset ini sebagai Kaggle Input, atau jalankan lokal dengan:

```powershell
$env:BRAILLE_PHYSICAL_HOLDOUT_DIR = 'C:\path\to\braille-physical-holdout\holdout'
$env:BRAILLE_NEGATIVE_DIR = 'C:\path\to\braille-physical-holdout\negative'
```

Setelah notebook selesai, salin report baru ke folder evidence kandidat dan perbarui hash melalui proses review. `release:model:verify` akan tetap menolak jika metrik atau bukti perangkat belum memenuhi ambang.
