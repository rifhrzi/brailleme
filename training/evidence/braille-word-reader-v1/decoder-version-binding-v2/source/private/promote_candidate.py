"""Promote only a frozen candidate whose actual external word gates pass."""
import argparse
import hashlib
import json
from pathlib import Path
from review_word_promotion import review_promotion

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--decoder-parity', type=Path, required=True)
parser.add_argument('--provenance', type=Path, required=True)
args = parser.parse_args()
root = Path.cwd()
args.provenance = args.provenance.resolve()
work = root / '.braille-training-work/word-scan-v1-20260905'
assets = root / 'assets/models/braille-word-reader-v1'
evidence = root / 'training/evidence/braille-word-reader-v1'
args.provenance.relative_to(evidence)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
benchmark = read(args.run / 'public-benchmark.json')
conversion = read(args.run / 'export-report.json')
selection = read(args.run / 'selected-validation-fullprecision.json')
decoder_version = read(assets / 'manifest.json')['decoderVersion']
assert isinstance(decoder_version, str) and decoder_version
assert selection['decoder_version'] == selection['oracle']['decoder_version'] == conversion['decoder_version'] == decoder_version
fixture = read(args.run / 'native-fixture.json')
provenance = read(args.provenance)
checkpoint_hash = sha(args.run / 'best-word.keras')
assert provenance['selectedCheckpointSha256'] == selection['oracle']['checkpoint_sha256'] == checkpoint_hash
assert conversion['source_checkpoint_sha256'] == checkpoint_hash
assert conversion['parity_fixture_sha256'] == selection['oracle']['fixture_sha256']
assert conversion['parity_samples'] == selection['words']['total'] == 405
selection_report = read(args.run / 'checkpoint-selection.json')
assert selection_report['decoder_version'] == decoder_version
assert all(candidate['decoder_version'] == decoder_version for candidate in selection_report['candidates'])
assert selection_report['selected_checkpoint_sha256'] == checkpoint_hash
assert selection_report['external_outputs_used'] is False
assert sha(root / provenance['selectionReport']['file']) == provenance['selectionReport']['sha256']
assert sha(args.run / 'checkpoint-selection.json') == provenance['selectionReport']['sha256']
for link in provenance['selectionSources']:
    assert sha(root / link['file']) == link['sha256']
additional_datasets = []
for supplemental in provenance.get('supplementalPreparations', []):
    assert supplemental['dataset'] == 'DSBI' and supplemental['revision'] == '6937eb8973efe5ff337e5c018485cb626303f5c2'
    assert sha(root / supplemental['sourceIndex']['file']) == supplemental['sourceIndex']['sha256']
    additional_datasets.append({'name': 'DSBI', 'repository': 'https://github.com/yeluo1994/DSBI',
                                'revision': supplemental['revision'], 'redistributionRightsConfirmed': False,
                                'sourceScope': '26 declared TRAIN sources only; 88 TEST image pixels not used'})
rows = benchmark['words']
assert len(rows) == 607
accepted = sum(bool(r['predicted']) for r in rows)
correct = sum(r['predicted'] == r['expected'] for r in rows)
assert benchmark['numerical_internal_gates_pass'] is True
assert correct / len(rows) >= .75 and correct / accepted >= .95
assert len(benchmark['negative_rows']) == 44 and not any(r['predicted'] for r in benchmark['negative_rows'])
assert len(benchmark['legacy_rows']) == 66
assert conversion['parity_pass'] is True
assert benchmark['model_sha256'] == conversion['artifact_sha256'] == sha(args.run / 'word_reader.tflite')
assert benchmark['thresholds'] == conversion['thresholds'] == selection['thresholds']
assert conversion['imageNormalization'] == selection['imageNormalization']
assert benchmark['fixture_manifest_sha256'] == sha(work / 'data/external-words.json')
parity = read(args.decoder_parity)
assert parity['model_sha256'] == conversion['artifact_sha256']
assert parity['benchmark_sha256'] == sha(args.run / 'public-benchmark.json')
assert parity['fixture_manifest_sha256'] == benchmark['fixture_manifest_sha256']
assert parity['thresholds'] == selection['thresholds']
assert parity['parity_passed'] is True and parity['first_mismatches'] == []
assert parity['total_pipeline_cases'] == len(parity['rows']) == 607
for key in ('quality_matches', 'pipeline_text_matches', 'pipeline_rejection_matches'):
    assert parity[key] == 607
for source in ('src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts',
               'src/ml/word-image-processing.ts', 'src/types/word-scan.ts'):
    assert sha(root / source) == parity['source_sha256'][source]
for index, (row, expected) in enumerate(zip(parity['rows'], rows, strict=True)):
    assert row['index'] == index and row['image'] == expected['image'] and row['expected'] == expected['expected']
    assert row['python_text'] == row['typescript_text'] == expected['predicted']
    assert row['python_reason'] == row['typescript_reason'] == expected['rejection']
assert sha(args.run / fixture['file']) == fixture['sha256']
for kind, count in [('repetition', 12), ('rejection', 14)]:
    diagnostics = read(args.run / f'{kind}-diagnostics.json')
    fixtures = read(evidence / f'{kind}-fixtures.json')
    assert diagnostics['model_sha256'] == conversion['artifact_sha256']
    assert diagnostics['thresholds'] == selection['thresholds']
    assert diagnostics['fixture_manifest_sha256'] == sha(evidence / f'{kind}-fixtures.json')
    assert diagnostics['total'] == len(diagnostics['rows']) == len(fixtures['records']) == count
    assert diagnostics['all_frozen_image_hashes_verified'] is True
    assert diagnostics['original_rgb_quality_checked_before_inference'] is True
    for name in ('word_diagnostics.py', 'word_benchmark.py', 'word_decode.py', 'word_data.py'):
        source = f'training/scripts/{name}'
        assert sha(root / source) == diagnostics['source_sha256'][source]
    for row, record in zip(diagnostics['rows'], fixtures['records']):
        assert all(row[key] == record[key] for key in ('image', 'image_sha256', 'expected'))
        assert sha(evidence / record['image']) == record['image_sha256']
        if row['predicted']:
            assert not row['rejection'] and row['detected_cell_count'] == len(row['predicted'])
        if kind == 'rejection':
            assert not row['predicted'] and row['rejection'] and row['correct'] is True

sources = {
    'publicBenchmark': (args.run / 'public-benchmark.json', evidence / 'public-benchmark.json'),
    'conversion': (args.run / 'export-report.json', evidence / 'export-report.json'),
    'validationSelection': (args.run / 'selected-validation-fullprecision.json', evidence / 'selected-validation-fullprecision.json'),
    'wholeWordFixtures': (work / 'data/external-words.json', evidence / 'external-words.json'),
    'decoderParity': (args.decoder_parity, evidence / 'decoder-parity.json'),
    'repetitionDiagnostics': (args.run / 'repetition-diagnostics.json', evidence / 'repetition-diagnostics.json'),
    'rejectionDiagnostics': (args.run / 'rejection-diagnostics.json', evidence / 'rejection-diagnostics.json'),
}
links = {}
proposed = {}
for key, (source, target) in sources.items():
    proposed[target] = source.read_bytes()
    links[key] = {'file': target.relative_to(root).as_posix(), 'sha256': sha(source)}
for key, file in {
    'boundaryReview': evidence / 'boundary-review.json',
    'nativePreprocessing': root / 'docs/evidence/2026-09-05-word-native-preprocessing.json',
    'nativeRuntime': evidence / 'native-worklet-performance.json',
    'repetitionFixtures': evidence / 'repetition-fixtures.json',
    'rejectionFixtures': evidence / 'rejection-fixtures.json',
}.items():
    links[key] = {'file': file.relative_to(root).as_posix(), 'sha256': sha(file)}
for name in ['checkpoint-selection.json', 'native-fixture.json', 'native-fixture-geometry.json']:
    source = args.run / name
    if source.is_file():
        proposed[evidence / name] = source.read_bytes()
assert Path(fixture['file']).name == fixture['file'] and fixture['file'].endswith('.png')
proposed[assets / 'word-reader.tflite'] = (args.run / 'word_reader.tflite').read_bytes()
proposed[assets / fixture['file']] = (args.run / fixture['file']).read_bytes()
manifest = read(assets / 'manifest.json')
assert manifest['decoderVersion'] == decoder_version, 'Manifest decoder version changed during preparation'
manifest['artifact'].update(status='unverified_candidate', sha256=sha(args.run / 'word_reader.tflite'), sizeBytes=(args.run / 'word_reader.tflite').stat().st_size)
manifest['thresholds'] = selection['thresholds']
manifest['imageNormalization'] = conversion['imageNormalization']
manifest['internalAcceptance'] = dict(passed=True, exactWordSuccessRate=correct/len(rows), acceptedWordAccuracy=correct/accepted, evaluatedWordCount=len(rows), acceptedWordCount=accepted, negativeImageCount=44, negativeFalseAccepts=0)
manifest['evidence'] = links
manifest['nativeSelfTest'] = fixture
manifest['physicalCameraValidated'] = False
manifest['provenance'] = {
    'datasets': [{'name': 'AngelinaDataset', 'repository': 'https://github.com/IlyaOvodov/AngelinaDataset', 'revision': 'e13038d483bb58e17046d640e078dc4768276e9b', 'redistributionRightsConfirmed': False}, *additional_datasets],
    'checkpointSha256': selection['oracle']['checkpoint_sha256'],
    'validationFixtureSha256': selection['oracle']['fixture_sha256'],
    'externalFixtureSha256': benchmark['fixture_manifest_sha256'],
    'retainedTrainingProvenance': {'file': args.provenance.relative_to(root).as_posix(), 'sha256': sha(args.provenance)},
    'retainedDataManifests': {'file': (evidence / 'data-manifests/index.json').relative_to(root).as_posix(), 'sha256': sha(evidence / 'data-manifests/index.json')},
    'selectionSources': provenance['selectionSources'],
    'selectionReport': 'training/evidence/braille-word-reader-v1/checkpoint-selection.json',
}
manifest['verification'] = {
    'tensorContractInspected': True, 'dotMaskMappingVerified': True, 'wordDecoderVerified': True,
    'preprocessingConfirmedAgainstTrainingCode': True, 'kerasTfliteConversionParityPassed': True,
    'nativePostCropFixturePassed': False, 'nativeCameraGoldenParityPassed': False,
    'independentPhysicalHoldoutPassed': False, 'negativeCorpusPassed': False,
    'deviceBenchmarkPassed': False, 'datasetRedistributionRightsConfirmed': False,
}
proposed[assets / 'manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode('utf-8')
module = root / 'src/ml/word-model-assets.ts'
original_module = module.read_bytes()
text = module.read_text(encoding='utf-8')
for declaration in ('WORD_MODEL_ASSET', 'WORD_NATIVE_FIXTURE_ASSET'):
    assert text.count(f'export const {declaration}: number | null = null;') == 1, f'Expected one pending literal for {declaration}'
text = text.replace('// Literal Metro assets are added only after the new model passes its benchmark.\n// A missing candidate must remain a normal unavailable state, not a broken import.', '// This candidate passed the frozen public benchmark; physical validation remains separate.')
text = text.replace('export const WORD_MODEL_ASSET: number | null = null;', "export const WORD_MODEL_ASSET: number | null = require('../../assets/models/braille-word-reader-v1/word-reader.tflite');")
text = text.replace('export const WORD_NATIVE_FIXTURE_ASSET: number | null = null;', f"export const WORD_NATIVE_FIXTURE_ASSET: number | null = require('../../assets/models/braille-word-reader-v1/{fixture['file']}');")
proposed[module] = text.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8')
review = review_promotion(root, proposed)
assert module.read_bytes() == original_module, 'Metro asset module changed during preflight'
temporaries = {target: target.with_name(target.name + '.promotion-tmp') for target in proposed}
assert not any(path.exists() for path in temporaries.values()), 'A previous unfinished promotion needs inspection'
for target, content in proposed.items():
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporaries[target].open('xb') as stream:
        stream.write(content)
for target, content in proposed.items():
    temporaries[target].replace(target)
    assert target.read_bytes() == content
print(json.dumps({'model_sha256': manifest['artifact']['sha256'], 'correct': correct, 'accepted': accepted, 'total': len(rows), 'fixture': fixture['expectedText'], 'prospective_verification': str(review)}, indent=2))
