"""Retain already-computed run24 held405 diagnosis; no inference or selection."""
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path.cwd().resolve()
WORK = ROOT / '.braille-training-work/word-scan-v1-20260905'
SOURCE = WORK / 'run24/held-diagnosis/bn-affine-comparison'
OUT = ROOT / 'training/evidence/braille-word-reader-v1/failed-selected-v19-diagnosis'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Keep existing evidence immutable')
    raw = (SOURCE / 'comparison.json').read_bytes()
    assert sha(raw) == '0d2b9dd0126f0a25d3deadfc3fbc0ae79de059f57fb44419267f2658f7b3fc48'
    report = json.loads(raw)
    assert report['model_execution'] is False and report['external_outputs_used'] is False
    assert report['summary']['baseline']['thresholds'] == report['summary']['candidate']['thresholds']
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0, compresslevel=9) as handle:
        handle.write(raw)
    compressed = stream.getvalue()
    assert gzip.decompress(compressed) == raw
    planned = {'comparison.json.gz': compressed}
    origins = {'comparison.json.gz': SOURCE / 'comparison.json'}

    def retain(source, name):
        planned[name] = source.read_bytes()
        origins[name] = source

    for name in ('README.md', 'FINDINGS.md', 'decision-transitions.json', 'summarize.py'):
        retain(SOURCE / name, 'original-documents/' + name + ('.txt' if name.endswith('.md') else ''))
    for name in ('compare_held_cache.py', 'cache_best_new_held_v2.py'):
        retain(WORK / name, 'scripts/' + name)
    retain(Path(__file__).resolve(), 'scripts/archive_bn_affine_diagnosis.py')
    retain(WORK / 'run24/held-diagnosis/cache-report.json', 'candidate-cache-report.json')
    retain(WORK / 'selected-v19/checkpoint-selection.json', 'checkpoint-selection.json')
    retain(WORK / 'selected-v19/selected-validation-fullprecision.json', 'selected-validation-fullprecision.json')
    retain(WORK / 'run24/sampling-lineage-final.json', 'candidate-sampling-lineage-final.json')
    for path in sorted((WORK / 'selected-v19/source').glob('*.py')):
        retain(path, 'evaluator-source/' + path.name)
    audit_dir = WORK / 'v19-held-diagnosis'
    for path in sorted(audit_dir.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            retain(path, 'independent-audit/' + path.relative_to(audit_dir).as_posix())
    for name, origin in origins.items():
        if origin.relative_to(ROOT).as_posix() in report['source_bindings']:
            assert sha(origin.read_bytes()) == report['source_bindings'][origin.relative_to(ROOT).as_posix()]
    planned['README.md'] = f'''# Diagnosis kandidat BatchNorm run24

Kontrol v15 tetap dipilih oleh seleksi v19. Keduanya menghasilkan **357/405 kata benar dan 357/362 hasil diterima benar**, dengan pembagian natural/stress identik. Aturan seri yang telah ditetapkan memilih kontrol.

Hanya dua keputusan kata berubah: ZWUKOW ditolak setelah confidence W turun dari 0,761 menjadi 0,644; NA diterima setelah confidence N naik dari 0,676 menjadi 0,704. Pola kedua sel tersebut tetap sama. Total deteksi 1.522 dan pasangan GT 1.518 tidak berubah. Pola salah pada pasangan GT naik dari delapan menjadi sembilan; mean IoU naik 0,002618 dan galat pusat turun 0,04734 piksel. Ini perubahan numerik nyata tanpa peningkatan kata utuh.

**111 GT tidak berpasangan mencakup 103 sel pada 30 gambar yang ditolak kualitasnya dan diberi keluaran nol. Hanya delapan GT tidak berpasangan pada gambar yang lulus kualitas.** Angka 111 tidak boleh disebut jumlah sel terlewat oleh detector. Lima hasil diterima yang salah tetap berupa kata lebih pendek; PTICA menjadi RC juga memuat salah pola T menjadi R.

[Audit independen](independent-audit/independent-aggregate-audit.json) memeriksa hash dan menghitung ulang agregat 405 baris tanpa mengulang inferensi atau pencarian ambang. Set ini terdiri atas 135 crop natural dan 270 varian stress yang berkorelasi, bukan 405 foto independen. Penolakan kualitas tetap dihitung gagal.

[Laporan lengkap terkompresi](comparison.json.gz) mempertahankan JSON asli persis. SHA-256 setelah dekompresi: `{sha(raw)}`; gzip: `{sha(compressed)}`. [Dokumen asli](original-documents/FINDINGS.md.txt) dan [tabel asli](original-documents/README.md.txt) dipertahankan tanpa perubahan. Metadata cache, seleksi, sumber evaluator, dan audit berada dalam [indeks hash](archive-index.json).

Tidak ada inferensi baru, evaluasi 607 eksternal, perubahan ambang, aktivasi model, atau APK dari pengarsipan ini. Target 75%/95%, 0/44 negatif, dan seluruh 14 kasus penolakan tetap berlaku; hasil eksternal masih milik kandidat v15 yang gagal.
'''.encode('utf-8')
    OUT.mkdir(parents=True)
    files = []
    for name, data in sorted(planned.items()):
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        assert path.read_bytes() == data
        item = {'file': name, 'sizeBytes': len(data), 'sha256': sha(data)}
        if name in origins:
            item.update(originalFile=origins[name].relative_to(ROOT).as_posix(), originalSha256=sha(origins[name].read_bytes()))
        if name == 'comparison.json.gz':
            item.update(encoding='gzip', uncompressedSizeBytes=len(raw), uncompressedSha256=sha(raw))
        files.append(item)
    index = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Cached held405 diagnosis of unselected run24 epoch9; no new inference, external evaluation, activation, or physical evidence.',
             'files': files}
    encoded = (json.dumps(index, indent=2) + '\n').encode('utf-8')
    (OUT / 'archive-index.json').write_bytes(encoded)
    (OUT / 'archive-index.sha256').write_text(sha(encoded) + '  archive-index.json\n', encoding='ascii')
    print(json.dumps({'archive': str(OUT), 'payload_files': len(files), 'payload_bytes': sum(x['sizeBytes'] for x in files), 'index_sha256': sha(encoded)}, indent=2))


if __name__ == '__main__':
    main()
