"""Retain finished diagnostic and verify a reason-label erratum, no inference."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
REPO = WORK.parent.parent
sys.path.insert(0, str(REPO / 'training/scripts'))
from word_benchmark import quality


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


benchmark = WORK / 'selected-v13/public-benchmark.json'
archive_benchmark = REPO / 'training/evidence/braille-word-reader-v1/failed-selected-v13/public-benchmark.json'
expected_hash = 'b8b160a4983e63bd31f8e667d3bfe8cdc5f6a25bd3324b1ab5e5d505352eadf7'
assert sha(benchmark) == sha(archive_benchmark) == expected_hash
report = read(benchmark)
npz = WORK / 'selected-v13/public-word-outputs.npz'
with np.load(npz) as arrays:
    # Read the tiny reason array only, never the model-output tensor or a model.
    reasons = arrays['quality_reasons'].tolist()
assert len(reasons) == len(report['words']) == 607
quality_rows = []
for row, reason in zip(report['words'], reasons):
    path = WORK / 'data' / row['image']
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('RGB').resize((480, 192), Image.Resampling.BILINEAR), np.float32) / 255.
    assert quality(pixels) == reason
    if reason:
        assert row['rejection'] == reason and not row['predicted']
        quality_rows.append({'image': row['image'], 'image_sha256': sha(path), 'reason': reason})
assert len(quality_rows) == 29 and Counter(reasons) == Counter({'': 578, 'low_confidence': 29})
actual_reasons = dict(Counter(row['rejection'] for row in report['words'] if not row['predicted']))
expected_reasons = dict(actual_reasons)
expected_reasons['low_confidence'] -= 29
expected_reasons['no_grid'] += 29
assert expected_reasons == report['whole_words']['rejection_reasons']

def counts(rows):
    accepted = sum(bool(row['predicted']) for row in rows)
    correct = sum(row['predicted'] == row['expected'] for row in rows)
    return {'total': len(rows), 'accepted': accepted, 'correct': correct,
            'exact_word_accuracy': correct / len(rows), 'accepted_accuracy': correct / accepted if accepted else None}

numeric = counts(report['words'])
numeric['by_length'] = {str(length): counts([row for row in report['words'] if len(row['expected']) == length]) for length in range(1, 13)}
assert numeric == {key: value for key, value in report['whole_words'].items() if key != 'rejection_reasons'}
destination = REPO / 'training/evidence/braille-word-reader-v1/legacy-v2-oracle-607'
destination.mkdir(parents=True, exist_ok=True)
assert not (destination / 'archive.json').exists(), 'Retained diagnostic must remain immutable.'
files = []
for name in ('README.md', 'diagnose.py', 'retain_evidence.py', 'protocol.json', 'report.json', 'supplementary-summary.json', 'probabilities.npz', 'frozen-crops.json', 'cell-results.json'):
    source = HERE / name
    compressed = name in {'frozen-crops.json', 'cell-results.json'}
    output = destination / (name + '.gz' if compressed else name)
    if compressed:
        with source.open('rb') as original, output.open('wb') as target:
            with gzip.GzipFile(fileobj=target, mode='wb', filename='', mtime=0) as packed:
                shutil.copyfileobj(original, packed)
        assert hashlib.sha256(gzip.decompress(output.read_bytes())).hexdigest() == sha(source)
    else:
        shutil.copyfile(source, output)
        assert sha(output) == sha(source)
    files.append({'logical_file': name, 'file': output.name, 'sha256': sha(output), 'logical_sha256': sha(source),
                  'compression': 'gzip' if compressed else None, 'size_bytes': output.stat().st_size})
save(destination / 'archive.json', {'schemaVersion': 1, 'scope': 'Finished diagnostic provenance, not a scanner or model acceptance claim.',
     'files': files, 'model_and_source_photos': 'Existing preserved classifier asset and external hash-bound original Angelina photos; no model/data copies modified.',
     'original_working_directory': HERE.relative_to(REPO).as_posix()})
erratum_path = REPO / 'training/evidence/braille-word-reader-v1/failed-selected-v13-rejection-reason-erratum.json'
assert not erratum_path.exists()
erratum = {'schemaVersion': 1, 'scope': 'Reason-summary labeling only; frozen benchmark, outputs, rows, numerical counts and acceptance gates unchanged.',
           'original_report': {'file': archive_benchmark.relative_to(REPO).as_posix(), 'sha256': expected_hash},
           'model_sha256': report['model_sha256'], 'fixture_manifest_sha256': report['fixture_manifest_sha256'],
           'cached_outputs': {'path': npz.relative_to(REPO).as_posix(), 'sha256': sha(npz), 'only_field_read': 'quality_reasons'},
           'source_cause': 'word_benchmark used zeros_like(output) for pre-decoder quality rejection. summarize decoded these empty placeholders as no_grid, while final rows correctly retained low_confidence. Replacement future reporting counts reasons directly from the final rows and preserves all other summarize metrics.',
           'original_summary_reasons': report['whole_words']['rejection_reasons'], 'final_row_reasons': actual_reasons,
           'quality_reasons_reproduced_from_all_607_cached_rgb_images': True, 'quality_rejected_count': 29,
           'quality_rejected_rows': quality_rows, 'all_numeric_fields_and_per_length_counts_match_final_rows': True,
           'unchanged_numeric_metrics': numeric,
           'future_fix': {'file': 'training/scripts/word_benchmark.py', 'sha256': sha(REPO / 'training/scripts/word_benchmark.py')},
           'focused_regression': {'file': 'training/tests/test_word_benchmark.py', 'sha256': sha(REPO / 'training/tests/test_word_benchmark.py'), 'tests_passed': 2},
           'verification_source': {'file': (destination / 'retain_evidence.py').relative_to(REPO).as_posix(), 'sha256': sha(__file__)}}
save(erratum_path, erratum)
assert sha(benchmark) == sha(archive_benchmark) == expected_hash
print(json.dumps({'archive': str(destination), 'retained_bytes': sum(row['size_bytes'] for row in files),
                  'archive_sha256': sha(destination / 'archive.json'), 'erratum_sha256': sha(erratum_path),
                  'reasons': actual_reasons, 'quality_rejected': len(quality_rows), 'unchanged_counts': counts(report['words'])}, indent=2))
