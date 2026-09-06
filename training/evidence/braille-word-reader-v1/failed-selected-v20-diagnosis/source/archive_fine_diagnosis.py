"""Retain the completed run25 cache diagnosis without new model execution."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd().resolve()
WORK = ROOT / '.braille-training-work/word-scan-v1-20260905'
SOURCE = WORK / 'run25/held-diagnosis/fine-branch-comparison'
OUT = ROOT / 'training/evidence/braille-word-reader-v1/failed-selected-v20-diagnosis'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError('Existing evidence must remain unchanged')
    raw = (SOURCE / 'comparison.json').read_bytes()
    assert sha(raw) == '3d071cb913cf88736322b75d0dac23a86b48df1ad76d980e1dfd7ac9ce171417'
    report = json.loads(raw)
    assert report['model_execution'] is False and report['external_outputs_used'] is False
    assert report['summary']['baseline']['words']['correct'] == 357
    assert report['summary']['candidate']['words']['correct'] == 356
    assert report['summary']['baseline']['thresholds'] == report['summary']['candidate']['thresholds']
    zipped = gzip.compress(raw, compresslevel=9, mtime=0)
    assert gzip.decompress(zipped) == raw
    payload = {'comparison.json.gz': zipped}
    origins = {}
    for name, digest in report['source_bindings'].items():
        path = ROOT / name
        if path.suffix != '.py' and path.name not in {'checkpoint-selection.json', 'cache-report.json'}:
            continue
        data = path.read_bytes()
        assert sha(data) == digest, f'Changed source {path}'
        target = 'source/' + digest[:12] + '-' + path.name
        payload[target] = data
        origins[target] = name
    payload['original-README.md.txt'] = (SOURCE / 'README.md').read_bytes()
    payload['source/archive_fine_diagnosis.py'] = Path(__file__).read_bytes()
    payload['README.md'] = (
        '# Diagnosis kandidat fitur awal run25\n\n'
        'Seleksi v20 mempertahankan kontrol v15. Kandidat baru menghasilkan 356/405 kata benar '
        'dan 356/361 hasil diterima benar; kontrol menghasilkan 357/405 dan 357/362. '
        'Hanya ZWUKOW berubah dari benar menjadi ditolak karena confidence titik; tidak ada kata benar baru.\n\n'
        'Jumlah pasangan anotasi 1.518, anotasi tidak berpasangan 111, dan deteksi tambahan empat tetap. '
        'Angka 111 mencakup sel pada 30 gambar yang ditolak kualitasnya dan diberi keluaran nol; '
        'angka ini tidak menunjukkan 111 sel terlewat pada gambar yang lulus kualitas. '
        'Pencocokan kotak bersifat diagnosis perkiraan, bukan AP.\n\n'
        '[Laporan lengkap](comparison.json.gz) mempertahankan JSON asli, SHA-256 setelah dekompresi '
        f'`{sha(raw)}`. [Tabel asli](original-README.md.txt) dan sumber evaluator diikat oleh '
        '[indeks hash](archive-index.json). Tensor cache tetap di direktori eksperimen dan hash asalnya '
        'tercatat dalam laporan; tensor tersebut tidak disalin ke arsip ini.\n\n'
        'Tidak ada inferensi baru, pencarian ambang, penggunaan 607 blok eksternal, aktivasi model, '
        'APK, atau bukti perangkat fisik dari diagnosis ini. Syarat penerimaan tetap berlaku.\n'
    ).encode('utf-8')
    OUT.mkdir(parents=True)
    files = []
    for name, data in sorted(payload.items()):
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assert target.read_bytes() == data
        entry = {'file': name, 'sizeBytes': len(data), 'sha256': sha(data)}
        if name in origins:
            entry['originalFile'] = origins[name]
        if name == 'comparison.json.gz':
            entry.update(encoding='gzip', uncompressedSha256=sha(raw), uncompressedSizeBytes=len(raw))
        files.append(entry)
    index = (json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n').encode()
    (OUT / 'archive-index.json').write_bytes(index)
    (OUT / 'archive-index.sha256').write_text(sha(index) + '  archive-index.json\n', encoding='ascii')
    print(json.dumps({'path': str(OUT), 'files': len(files), 'bytes': sum(x['sizeBytes'] for x in files), 'index_sha256': sha(index)}))


if __name__ == '__main__':
    main()
