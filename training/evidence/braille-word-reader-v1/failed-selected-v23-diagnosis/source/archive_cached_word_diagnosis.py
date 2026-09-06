"""Archive an existing cache comparison and its evaluator sources; no inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('run', 'selection', 'comparison', 'expected-sha', 'description'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    for name in (args.run, args.selection, args.comparison):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Expected a direct folder name')
    root = Path.cwd()
    work = root / '.braille-training-work/word-scan-v1-20260905'
    source = work / args.run / 'held-diagnosis' / args.comparison
    out = root / 'training/evidence/braille-word-reader-v1' / ('failed-' + args.selection + '-diagnosis')
    if out.exists():
        raise FileExistsError('Existing evidence must remain immutable')
    raw = (source / 'comparison.json').read_bytes()
    assert sha(raw) == args.expected_sha
    report = json.loads(raw)
    assert report['model_execution'] is False and report['external_outputs_used'] is False
    assert report['summary']['baseline']['thresholds'] == report['summary']['candidate']['thresholds']
    zipped = gzip.compress(raw, compresslevel=9, mtime=0)
    assert gzip.decompress(zipped) == raw
    payload = {'comparison.json.gz': zipped, 'original-README.md.txt': (source / 'README.md').read_bytes()}
    for name, digest in report['source_bindings'].items():
        path = root / name
        if path.suffix != '.py' and path.name not in {'checkpoint-selection.json', 'cache-report.json'}:
            continue
        data = path.read_bytes()
        assert sha(data) == digest
        payload['source/' + digest[:12] + '-' + path.name] = data
    payload['source/archive_cached_word_diagnosis.py'] = Path(__file__).read_bytes()
    baseline = report['summary']['baseline']['words']
    candidate = report['summary']['candidate']['words']
    transition = report['transitions']
    payload['README.md'] = (
        '# Diagnosis cache ' + args.selection + '\n\n' + args.description + '\n\n'
        f"Kontrol: {baseline['correct']}/{baseline['total']} kata benar, {baseline['accepted']} diterima. "
        f"Kandidat: {candidate['correct']}/{candidate['total']} kata benar, {candidate['accepted']} diterima. "
        f"Kata benar yang hilang: {transition['baseline_correct_candidate_wrong']}; "
        f"kata benar baru: {transition['baseline_wrong_candidate_correct']}.\n\n"
        'Pencocokan kotak bersifat diagnosis perkiraan, bukan AP. Anotasi tidak berpasangan mencakup '
        'sel pada gambar yang ditolak kualitasnya dan diberi keluaran nol; angka itu tidak boleh '
        'seluruhnya disebut sel yang terlewat oleh detector pada gambar yang lulus kualitas.\n\n'
        f'[Laporan lengkap](comparison.json.gz) mempertahankan JSON asli, SHA-256 setelah dekompresi `{sha(raw)}`. '
        '[Indeks hash](archive-index.json) mengikat salinan metadata dan sumber evaluator. '
        'Tensor cache tetap di direktori eksperimen; hash asalnya tercatat dalam laporan.\n\n'
        'Pengarsipan ini tidak menjalankan inferensi baru, pencarian ambang, evaluasi eksternal, '
        'aktivasi model, pembuatan APK, atau pengujian fisik. Syarat penerimaan tetap berlaku.\n'
    ).encode('utf-8')
    files = []
    out.mkdir(parents=True)
    for name, data in sorted(payload.items()):
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        assert target.read_bytes() == data
        row = {'file': name, 'sha256': sha(data), 'sizeBytes': len(data)}
        if name == 'comparison.json.gz':
            row.update(encoding='gzip', uncompressedSizeBytes=len(raw), uncompressedSha256=sha(raw))
        files.append(row)
    index = (json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n').encode()
    (out / 'archive-index.json').write_bytes(index)
    print(json.dumps({'files': len(files), 'bytes': sum(r['sizeBytes'] for r in files), 'index_sha256': sha(index)}))


if __name__ == '__main__':
    main()
