"""Archive existing run23 cached-output diagnosis without model execution."""
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import zlib

ROOT=Path.cwd().resolve()
WORK=ROOT/'.braille-training-work/word-scan-v1-20260905'
ORIGIN=WORK/'run23/held-diagnosis'
REPORT=ORIGIN/'stretch-comparison'
OUT=ROOT/'training/evidence/braille-word-reader-v1/failed-selected-v18-diagnosis'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compress(data):
    stream=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',fileobj=stream,compresslevel=9,mtime=0) as handle:
        handle.write(data)
    return stream.getvalue()


def main():
    assert not OUT.exists(), 'Do not overwrite frozen evidence'
    raw=(REPORT/'comparison.json').read_bytes()
    assert sha(raw)=='2447c27fc97981b448e0b7dc8dc3e570981a11e187b352088fcfb93119e723d8'
    report=json.loads(raw)
    assert report['model_execution'] is False and report['external_outputs_used'] is False
    assert report['summary']['baseline']['thresholds']==report['summary']['candidate']['thresholds']
    compressed=compress(raw)
    assert compressed==compress(raw) and gzip.decompress(compressed)==raw
    OUT.mkdir(parents=True)
    files=[]
    def put(relative,data,origin=None,encoding='identity',uncompressed=None):
        path=OUT/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        row={'file':relative,'sizeBytes':len(data),'sha256':sha(data),'encoding':encoding}
        if origin:
            row.update(originalFile=origin.relative_to(ROOT).as_posix(),originalSha256=sha(origin.read_bytes()))
        if uncompressed is not None:
            row.update(uncompressedSizeBytes=len(uncompressed),uncompressedSha256=sha(uncompressed))
        files.append(row)
    def copy(source,relative):
        put(relative,source.read_bytes(),source)
    put('comparison.json.gz',compressed,REPORT/'comparison.json','gzip',raw)
    for name in ('README.md','FINDINGS.md'):
        copy(REPORT/name,'original-documents/'+name+'.txt')
    copy(REPORT/'execution.log','execution.log')
    copy(WORK/'compare_held_cache.py','scripts/compare_held_cache.py')
    copy(Path(__file__).resolve(),'scripts/archive_stretch_diagnosis.py')
    copy(ORIGIN/'cache-report.json','candidate-cache-report.json')
    copy(WORK/'selected-v18/checkpoint-selection.json','checkpoint-selection.json')
    copy(WORK/'selected-v18/selected-validation-fullprecision.json','selected-v18-validation-fullprecision.json')
    copy(WORK/'selected-v17/selected-validation-fullprecision.json','baseline-selected-validation-fullprecision.json')
    copy(WORK/'run23/sampling-lineage-final.json','candidate-sampling-lineage-final.json')
    for source in sorted((WORK/'selected-v18/source').glob('*.py')):
        copy(source,'evaluator-source/'+source.name)
    copy(WORK/'selected-v17/source/word_data.py','baseline-evaluator-source/word_data.py')
    copy(ROOT/'training/scripts/benchmark_angelina.py','evaluator-source/benchmark_angelina.py')
    findings=(REPORT/'FINDINGS.md').read_text(encoding='utf-8').replace('](README.md)','](TABLES.md)').replace('](comparison.json)','](comparison.json.gz)')
    findings=findings.replace('SHA-256 laporan JSON:', 'SHA-256 JSON setelah dekompresi:')
    put('FINDINGS.md',findings.encode('utf-8'))
    tables=(REPORT/'README.md').read_text(encoding='utf-8').replace('[comparison.json](comparison.json)','[comparison.json.gz](comparison.json.gz)')
    tables=tables.replace('Full row-level diagnosis:', 'Full row-level diagnosis (gzip; the following hash is for the decompressed JSON):')
    put('TABLES.md',tables.encode('utf-8'))
    thresholds=report['summary']['baseline']['thresholds']
    threshold_table='\n'.join(f'| {name} | {value} | {value} |' for name,value in thresholds.items())
    readme=f'''# Failed selected-v18: diagnosis vertical stretch

Run23 epoch9 dan kontrol v15 sama-sama benar pada **357 dari 405 blok**. Kontrol tetap dipilih karena skor kelompok natural/stress yang dibobot sama lebih tinggi. Arsip ini terpisah dari bukti seleksi v18 lainnya dan tidak mengubah model aktif.

Baca [temuan](FINDINGS.md) dan [tabel per panjang/pola](TABLES.md). Satu blok natural `ZWUKOW` menjadi penolakan confidence; satu varian stress `WRAQ` membaik setelah sel A yang terlewat ditemukan. Analisis menggunakan cache tersimpan dan matching satu-ke-satu berbasis IoU >=0,25, tanpa label untuk menentukan pasangan. Ini diagnosis aproksimasi, bukan AP.

Set ini berisi 135 crop natural dan 270 varian stress yang berkorelasi, bukan 405 sumber independen. Tiga puluh penolakan kualitas dihitung gagal pada kedua model. Panjang 10–12 dan huruf F/V tidak terwakili. Tidak ada inferensi baru, external607, pencarian ambang, APK, atau bukti perangkat fisik.

Ambang kontrol dan kandidat persis sama:

| Parameter | Kontrol v15 | Stretch run23 epoch9 |
|---|---:|---:|
{threshold_table}

Identitas checkpoint dan hasil seleksi asli tersimpan dalam [checkpoint-selection.json](checkpoint-selection.json), [laporan kontrol](baseline-selected-validation-fullprecision.json), [laporan v18](selected-v18-validation-fullprecision.json), dan [metadata cache kandidat](candidate-cache-report.json). [Decoder yang diarsipkan](evaluator-source/word_decode.py) tetap sama untuk kedua kandidat. Kriteria APK 75%/95%, negatif 0/44, diagnostik penolakan 0/14, dan uji fisik tidak berubah.

## Membaca laporan lengkap

[comparison.json.gz](comparison.json.gz) berisi seluruh 405 keputusan per model, kotak, probabilitas titik, matching, dan hash sumber/fixture/cache/checkpoint. Gzip deterministik memakai level 9, `mtime=0`, tanpa nama file dalam header. Ukuran asli {len(raw):,} byte; SHA-256 setelah dekompresi `{sha(raw)}`. Gzip berukuran {len(compressed):,} byte; SHA-256 `{sha(compressed)}`.

```python
import gzip, json
from pathlib import Path
archive = Path("training/evidence/braille-word-reader-v1/failed-selected-v18-diagnosis")
with gzip.open(archive / "comparison.json.gz", "rt", encoding="utf-8") as handle:
    report = json.load(handle)
print(report["summary"]["baseline"]["words"])
print(report["summary"]["candidate"]["words"])
```

[archive-index.json](archive-index.json) memuat ukuran/hash setiap file dan asal salinannya, termasuk hash mentah dan gzip. [Hash index](archive-index.sha256) disimpan terpisah. [Log eksekusi](execution.log) berasal dari replay cache yang sudah selesai; pengarsipan tidak menjalankan ulang diagnosis.

Byte dokumen asli dipertahankan sebagai [README.md.txt](original-documents/README.md.txt) dan [FINDINGS.md.txt](original-documents/FINDINGS.md.txt). Dokumen presentasi di akar arsip hanya memperbarui tautan dan keterangan gzip. Semua JSON asli tetap identik setelah dekompresi.

[Skrip diagnosis asli](scripts/compare_held_cache.py) disimpan persis; untuk replay ia memerlukan cache tensor, checkpoint, dan fixture workspace yang tidak disalin ke arsip ini. Tidak ada tensor, model, atau APK dalam arsip.
'''
    put('README.md',readme.encode('utf-8'))
    index={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Run23 vertical-stretch best new candidate cached held405 diagnosis; no activation, new inference, external607, APK, or physical proof.',
        'selectedControlCheckpointSha256':report['summary']['baseline']['checkpoint_sha256'],
        'unselectedStretchCheckpointSha256':report['summary']['candidate']['checkpoint_sha256'],
        'fixtureSha256':json.loads((ORIGIN/'cache-report.json').read_bytes())['fixture_sha256'],
        'modelActivated':False,'physicalCameraValidated':False,'internalApkProduced':False,
        'gzip':{'mtime':0,'filename':'','compresslevel':9,'python':platform.python_version(),'zlib':zlib.ZLIB_VERSION,'repeatedCompressionIdentical':True},
        'files':sorted(files,key=lambda row:row['file'])}
    encoded=(json.dumps(index,indent=2)+'\n').encode('utf-8')
    (OUT/'archive-index.json').write_bytes(encoded)
    (OUT/'archive-index.sha256').write_text(sha(encoded)+'  archive-index.json\n',encoding='ascii')
    for row in files:
        contents=(OUT/row['file']).read_bytes()
        assert sha(contents)==row['sha256'] and len(contents)==row['sizeBytes']
        if row['encoding']=='gzip':
            contents=gzip.decompress(contents)
            assert sha(contents)==row['uncompressedSha256'] and len(contents)==row['uncompressedSizeBytes']
        if 'originalFile' in row:
            assert contents==(ROOT/row['originalFile']).read_bytes()
            if row['originalFile'] in report['source_bindings']:
                assert sha(contents)==report['source_bindings'][row['originalFile']]
    links=0
    for name in ('README.md','FINDINGS.md','TABLES.md'):
        for target in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text(encoding='utf-8')):
            assert (OUT/target).is_file(),(name,target)
            links+=1
    assert {p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}=={row['file'] for row in files}|{'archive-index.json','archive-index.sha256'}
    print(json.dumps({'archive':str(OUT),'payload_files':len(files),'payload_bytes':sum(row['sizeBytes'] for row in files),
        'gzip_bytes':len(compressed),'gzip_sha256':sha(compressed),'index_sha256':sha(encoded),'originals_preserved':True,'links_checked':links},indent=2))


if __name__=='__main__':
    main()
