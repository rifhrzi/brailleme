"""Retain the completed cached mirror diagnosis; never execute a model."""
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
ORIGIN=WORK/'run22/held-diagnosis'
REPORT=ORIGIN/'mirror-comparison'
OUT=ROOT/'training/evidence/braille-word-reader-v1/failed-selected-v17-diagnosis'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compress(data):
    output=io.BytesIO()
    with gzip.GzipFile(filename='',mode='wb',fileobj=output,compresslevel=9,mtime=0) as stream:
        stream.write(data)
    return output.getvalue()


def main():
    assert not OUT.exists(), 'Never overwrite frozen evidence'
    assert sha((REPORT/'comparison.json').read_bytes())=='072eb832e019ff416ac2d8fd7d3018d51f06b532a55fcd45a4b2e98db8a76363'
    assert sha((REPORT/'pattern-followup.json').read_bytes())=='7cfbb9b7b46324db556931b4d9aa9087fd1a5af8f18fd44b14edd3b16a7ae078'
    OUT.mkdir(parents=True)
    files=[]
    def put(relative,data,origin=None,encoding='identity',raw=None):
        path=OUT/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        record={'file':relative,'sizeBytes':len(data),'sha256':sha(data),'encoding':encoding}
        if origin:
            record['originalFile']=str(origin.relative_to(ROOT)).replace('\\','/')
            record['originalSha256']=sha(origin.read_bytes())
        if raw is not None:
            record['uncompressedSizeBytes']=len(raw)
            record['uncompressedSha256']=sha(raw)
        files.append(record)
    def copy(source,target):
        put(target,source.read_bytes(),source)
    raw=(REPORT/'comparison.json').read_bytes()
    source_bindings=json.loads(raw)['source_bindings']
    compressed=compress(raw)
    assert compressed==compress(raw) and gzip.decompress(compressed)==raw
    put('comparison.json.gz',compressed,REPORT/'comparison.json','gzip',raw)
    copy(REPORT/'pattern-followup.json','pattern-followup.json')
    for name in ('FINDINGS.md','README.md'):
        copy(REPORT/name,'original-documents/'+name+'.txt')
    for name in ('compare_control.py','pattern_followup.py'):
        copy(ORIGIN/name,'scripts/'+name)
    copy(WORK/'compare_held_cache.py','scripts/compare_held_cache.py')
    copy(Path(__file__).resolve(),'scripts/archive_mirror_diagnosis.py')
    copy(ORIGIN/'cache-report.json','candidate-cache-report.json')
    copy(WORK/'selected-v17/checkpoint-selection.json','checkpoint-selection.json')
    copy(WORK/'selected-v17/selected-validation-fullprecision.json','baseline-selected-validation-fullprecision.json')
    copy(WORK/'run22/sampling-lineage-final.json','candidate-sampling-lineage-final.json')
    copy(ORIGIN/'generic-replay-final/replay-proof.json','generic-replay-proof.json')
    for path in sorted((WORK/'selected-v17/source').glob('*.py')):
        copy(path,'evaluator-source/'+path.name)
    copy(ROOT/'training/scripts/benchmark_angelina.py','evaluator-source/benchmark_angelina.py')
    findings=(REPORT/'FINDINGS.md').read_text(encoding='utf-8').replace('[README.md](README.md)','[TABLES.md](TABLES.md)').replace('[comparison.json](comparison.json)','[comparison.json.gz](comparison.json.gz)')
    findings=findings.replace('`../compare_control.py` dan `../pattern_followup.py`','[compare_control.py](scripts/compare_control.py) dan [pattern_followup.py](scripts/pattern_followup.py)')
    put('FINDINGS.md',findings.encode('utf-8'))
    tables=(REPORT/'README.md').read_text(encoding='utf-8').replace('[comparison.json](comparison.json)','[comparison.json.gz](comparison.json.gz)')
    tables=tables.replace('Full row-level diagnosis:', 'Full row-level diagnosis (gzip; the following hash is for the decompressed JSON):')
    put('TABLES.md',tables.encode('utf-8'))
    readme=f'''# Failed selected-v17: diagnosis mirror pada validasi405

Run22 epoch8 mirror gagal mengungguli kontrol v15: **314/405 versus357/405** blok benar. Seleksi v17 tetap memilih kontrol. Ini arsip diagnosis dari keluaran tersimpan, bukan model yang diaktifkan, APK, atau pengujian perangkat fisik.

Baca [temuan](FINDINGS.md) dan [tabel per panjang/pola](TABLES.md). Sebanyak27 kesalahan pola pantulan A/B/K/L berkaitan dengan kotak bergeser ke kiri; penjelasan ini aproksimasi dan tidak mengisolasi penyebab optimisasi/normalisasi. Matching memakai pasangan satu-ke-satu IoU >=0,25 setelah NMS, tanpa menggunakan label untuk mencocokkan. Ini bukan AP.

Seluruh405 blok berasal dari135 crop natural dan270 varian stress yang berkorelasi. Tiga puluh gambar ditolak kualitas pada kedua kandidat dan tetap dihitung gagal. Panjang10–12 dan hurufF/V tidak terwakili. Tidak ada607 eksternal, inferensi baru, pencarian ambang, atau perubahan model dalam diagnosis ini.

Ambang tetap milik masing-masing kandidat:

| Parameter | Kontrol v15 | Mirror run22 epoch8 |
|---|---:|---:|
| proposalScore | 0.25 | 0.15 |
| acceptScore | 0.25 | 0.20 |
| dotConfidence | 0.70 | 0.70 |
| nmsIou | 0.25 | 0.25 |

Seluruh ambang geometri serta identitas checkpoint ada dalam [tabel seleksi asli](checkpoint-selection.json), [laporan baseline](baseline-selected-validation-fullprecision.json), dan [laporan cache kandidat](candidate-cache-report.json). Decoder persis yang digunakan tersedia di [evaluator-source/word_decode.py](evaluator-source/word_decode.py). Dataset tetap public development validation; kriteria APK75%/95%, negatif0/44, diagnostik penolakan0/14, dan uji fisik tidak berubah.

## Membaca JSON terkompresi

[comparison.json.gz](comparison.json.gz) berisi JSON asli seluruh405 keputusan per model, enam probabilitas titik, kotak, matching, statistik, dan hash sumber. Kompresi gzip memakai level9, `mtime=0`, tanpa nama file di header. Hash JSON setelah dibuka adalah `{sha(raw)}` ({len(raw):,} byte); hash file gzip adalah `{sha(compressed)}` ({len(compressed):,} byte).

Contoh membaca langsung tanpa ekstraksi atau inferensi:

```python
import gzip, json
from pathlib import Path
archive = Path("training/evidence/braille-word-reader-v1/failed-selected-v17-diagnosis")
with gzip.open(archive / "comparison.json.gz", "rt", encoding="utf-8") as handle:
    report = json.load(handle)
print(report["summary"]["control"]["words"])
print(report["summary"]["mirror"]["words"])
```

[pattern-followup.json](pattern-followup.json) merinci27 kasus pantulan dan16 kasus clipped yang kehilangan hasil benar. [Hash index](archive-index.json) mencatat hash/ukuran terkompresi dan asli serta asal setiap salinan; [hash index itu sendiri](archive-index.sha256) tersedia terpisah.

Dokumen asli dipertahankan byte demi byte sebagai [FINDINGS.md.txt](original-documents/FINDINGS.md.txt) dan [README.md.txt](original-documents/README.md.txt). FINDINGS/TABLES pada akar arsip menyesuaikan tautan untuk gzip; tidak ada perubahan pada laporan JSON asli. Ekstensi `.txt` pada salinan asli mencegah tautan lokasi kerja lama diperlakukan sebagai tautan arsip.

Skrip asli [compare_control.py](scripts/compare_control.py) dan [pattern_followup.py](scripts/pattern_followup.py), serta CLI [compare_held_cache.py](scripts/compare_held_cache.py), disimpan persis. Skrip asli mengacu pada tata letak workspace ketika dijalankan; replay membutuhkan cache tensor dan fixture asal yang tidak disalin ke arsip ini. CLI menerima lokasi eksplisit. [Bukti replay CLI](generic-replay-proof.json) mencatat kesetaraan810 keputusan dan3258 baris sel GT dengan diagnosis asli. Archive tidak mengandung checkpoint, tensor, atau APK.
'''
    put('README.md',readme.encode('utf-8'))
    index={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Failed run22 horizontal-mirror candidate held405 cached-output diagnosis; no inference, activation, external607, APK, or physical proof.',
        'selectedControlCheckpointSha256':'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7',
        'failedMirrorCheckpointSha256':'6dfcc85d16ffccd5fda0d926aa8b93d0b49e899569a175f98039c3bd0b23e463',
        'fixtureSha256':'e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625',
        'modelActivated':False,'physicalCameraValidated':False,'internalApkProduced':False,
        'gzip':{'mtime':0,'filename':'','compresslevel':9,'python':platform.python_version(),'zlib':zlib.ZLIB_VERSION,'repeatedCompressionIdentical':True},
        'files':sorted(files,key=lambda row:row['file'])}
    encoded=(json.dumps(index,indent=2)+'\n').encode('utf-8')
    (OUT/'archive-index.json').write_bytes(encoded)
    (OUT/'archive-index.sha256').write_text(sha(encoded)+'  archive-index.json\n',encoding='ascii')
    # Read-only archive verification, including original source preservation.
    for record in files:
        contents=(OUT/record['file']).read_bytes()
        assert len(contents)==record['sizeBytes'] and sha(contents)==record['sha256']
        if record['encoding']=='gzip':
            contents=gzip.decompress(contents)
            assert len(contents)==record['uncompressedSizeBytes'] and sha(contents)==record['uncompressedSha256']
        if 'originalFile' in record:
            assert contents==(ROOT/record['originalFile']).read_bytes()
            if record['originalFile'] in source_bindings:
                assert sha(contents)==source_bindings[record['originalFile']]
    followup=json.loads((OUT/'pattern-followup.json').read_bytes())
    assert sha((OUT/'scripts/pattern_followup.py').read_bytes())==followup['source_sha256']
    assert sha((OUT/'scripts/compare_held_cache.py').read_bytes())==json.loads((OUT/'generic-replay-proof.json').read_bytes())['script_sha256']
    links=0
    for name in ('README.md','FINDINGS.md','TABLES.md'):
        for target in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text(encoding='utf-8')):
            assert (OUT/target).is_file(), (name,target)
            links+=1
    assert set(p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file())=={r['file'] for r in files}|{'archive-index.json','archive-index.sha256'}
    print(json.dumps({'archive':str(OUT),'payload_files':len(files),'payload_bytes':sum(r['sizeBytes'] for r in files),
        'gzip_bytes':len(compressed),'gzip_sha256':sha(compressed),'index_sha256':sha(encoded),
        'originals_preserved':True,'links_checked':links},indent=2))


if __name__=='__main__':
    main()
