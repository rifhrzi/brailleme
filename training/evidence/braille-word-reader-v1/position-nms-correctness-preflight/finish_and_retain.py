"""Freeze the reviewed private patch/proofs, without applying active source edits."""
from datetime import datetime,timezone
import difflib
import gzip
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
ARCHIVE=ROOT/'training/evidence/braille-word-reader-v1/position-nms-correctness-preflight'
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
spec=read(OUT/'prototype-spec.json');comparison=read(OUT/'held405-comparison/report.json')
assert not ARCHIVE.exists() and not (OUT/'integration.patch').exists()
patch=[]
for name in ('training/scripts/word_decode.py','src/ml/word-decoder.ts',
             'training/tests/test_word_decoder_positions.py','src/ml/__tests__/word-decoder-positions.test.ts'):
    original=OUT/'source-original'/name;private=OUT/'source-private'/name
    before=original.read_text().splitlines(keepends=True) if original.exists() else []
    after=private.read_text().splitlines(keepends=True)
    patch.extend(difflib.unified_diff(before,after,fromfile='a/'+name if before else '/dev/null',tofile='b/'+name))
(OUT/'integration.patch').write_text(''.join(patch),encoding='utf-8')
regression={'status':'passed','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
    'scope':'Recorded from the successful actual Python unittest and jest-expo subprocess exits in this task; no model execution.',
    'python_tests':5,'typescript_jest_tests':5,'python_exit_code':0,'jest_exit_code':0,
    'commands':['python source-private/training/tests/test_word_decoder_positions.py',
                'node node_modules/jest/bin/jest.js --config <prototype>/jest.config.cjs --runInBand --no-cache'],
    'source_sha256':{name:sha((OUT/'source-private'/name).read_bytes()) for name in (
        'training/scripts/word_decode.py','src/ml/word-decoder.ts','training/tests/test_word_decoder_positions.py',
        'src/ml/__tests__/word-decoder-positions.test.ts')},
    'jest_config_sha256':sha((OUT/'jest.config.cjs').read_bytes()),
    'synthetic_reference_cases':20,'python_typescript_coordinate_max_abs':0.,'cache_equivalence_checks':1600}
(OUT/'focused-regression-execution.json').write_text(json.dumps(regression,indent=2)+'\n')
notes='''# Position-only NMS correctness fix: private preflight

The original decoder suppresses one of two valid disjoint adjacent-grid cells before inspecting their boxes. Perfect AB or AA can therefore become accepted A. The private Python and TypeScript fixes remove only that local-grid suppression and preserve score-ordered positional IoU NMS, its deterministic ties, geometry/dot gates, quality, word limits and sorting.

The calibration cache now removes only scores below its declared minimum proposal threshold. All trials must use proposalScore at least that minimum. Geometry/dots and malformed rows remain exact. An old cache that already discarded nonmaxima cannot recover them. The saved selected-v15 backend-fullprecision.npz is full raw model output, independently bound by its95,425,411-byte SHA20f0a6c0224ec7f640ca0e46275914e912a2b1b440882f0b1e9e94e8bb72556b.

Twenty synthetic Python/TypeScript cases match exactly, including every coordinate. They cover equal/unequal adjacent AB/AA, real overlapping duplicates, an NMS A-B-C chain whose disjoint A/C survive, weak/uncertain/unsupported adjacent cells, clipped/outside-guide boxes, blank, and lengths1/12/13. The new cache matches raw decoding in1,600 fixed threshold trials. Two reviewable five-test regression files also pass actual unittest and jest-expo. TypeScript retains the existing worklet directives and adds no syntax or dependencies; this is source-level compatibility, not Android execution.

The single authorized cached405 replay used unchanged v15 outputs, original405 labels, cached30 quality flags and exactly .25/.25/.7 plus all original geometry gates. Correct words changed357/405→355/405; accepted362→359; accepted precision98.6188%→98.8858%. Natural changed130/135→129/135 with100%precision. Stress changed227/270→226/270, precision97.8448%→98.2609%. No exact word was gained. Cases20/176 LIQESTWO changed from correct acceptance to clipped;173 EGO changed from incorrect EO to clipped. The other four accepted wrong shortened words remain. Every changed case retained one extra box crossing the guide from above/left. This is a correctness fix with an observed two-word validation cost, not an accuracy improvement.

Across375 quality-passing blocks, score-qualified guide-intersecting proposals grow1,528 local maxima→8,675 proposals. Existing NMS reduces these to1,522→1,525 retained boxes. Most newly exposed neighboring proposals are removed positionally. The focal loss treats noncenter Gaussian sites as downweighted negatives; their geometry/dots are not directly supervised. They are not soft-positive cell targets. Additional poor predictions and Android latency remain risks; this single CPU timing is not a device benchmark. The separate native ideal405 audit found no adjacent-grid chosen GT pair, so405 cannot demonstrate recovery of the tiny-cell synthetic case.

Integration is deliberately pending selector completion. integration.patch contains the two decoder changes and two new regression files. Existing Python tests that pass make_target directly as a prediction need a TEST-only conversion retaining exact positive-center objectness>=.999 and zeroing other objectness. Gaussian shoulder values are loss weights, not ideal predicted objects; make_target, loss and training data must remain unchanged. A direct analytic repro gives too_far for the raw target and A for center-only ideal output under the new decoder. Review affected helpers in test_word_pipeline.py, test_word_benchmark.py, test_word_fixture_geometry.py and test_word_selection.py. Existing golden source fixtures must be reviewed case-by-case; do not blanket overwrite expected values.

Version consumers: WORD_DECODER_VERSION in src/ml/word-model-contract.ts feeds word-photo-camera.tsx and word-scan-screen.tsx events. Literal fixtures also occur in word-scan-screen.test.tsx and types/word-scan.test.ts. The DTO validates a nonempty version, not a fixed literal. Current model manifests do not have an explicit decoderVersion field; actual decoder/preprocessing source SHA already binds parity evidence. Root owns the proposed six-dot-word-v2 constant/literal update and new selection protocol; historical evidence remains immutable. No manifest-schema expansion is needed for this fix.

This archive contains sparse synthetic cases and compact reports, not raw model tensors or source photographs. The retained comparison establishes fixed-output effects only; it does not select a checkpoint, authorize a model promotion, or establish physical-camera readiness. Original and private sources are both retained by hash; duplicate bytes are stored once and referenced by the preflight map.
'''
(OUT/'README.md').write_text(notes,encoding='utf-8')
ARCHIVE.mkdir(parents=True);payloads=[];mapping={};dedup={}
files=[p for side in ('source-original','source-private') for p in (OUT/side).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [OUT/name for name in ('prepare.py','test_private.py','test_private.cjs','jest.config.cjs','finish_and_retain.py',
    'prototype-spec.json','private-edits.patch','integration.patch','cases.json','python-results.json',
    'python-test-report.json','typescript-test-report.json','focused-regression-execution.json','README.md','compare_held405.py')]
files += list((OUT/'held405-comparison').glob('*.json'))
for path in files:
    name=path.relative_to(OUT).as_posix();raw=path.read_bytes();rawsha=sha(raw)
    if rawsha in dedup:mapping[name]=dict(dedup[rawsha]);continue
    compressed=path.suffix=='.json' and len(raw)>100000
    data=gzip.compress(raw,compresslevel=9,mtime=0) if compressed else raw
    dest=ARCHIVE/(name+('.gz' if compressed else ''));dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    ref={'file':dest.relative_to(ROOT).as_posix(),'sha256':sha(data)}
    if compressed:
        assert gzip.decompress(data)==raw
        ref.update(encoding='gzip',rawSha256=rawsha,rawSizeBytes=len(raw))
    mapping[name]=ref;dedup[rawsha]=ref;payloads.append(dict(ref,sizeBytes=len(data),originalFile=path.relative_to(ROOT).as_posix()))
preflight={'schemaVersion':1,'createdAtUtc':datetime.now(timezone.utc).isoformat(),
    'scope':'Private position-only NMS correctness patch, synthetic equivalence tests and one authorized fixed-threshold raw405 comparison; active source was not changed by this preparation.',
    'files':mapping,'sourceOriginalSha256':spec['source_original_sha256'],'sourcePrivateSha256':spec['source_private_sha256'],
    'comparisonReport':mapping['held405-comparison/report.json'],'integrationPatch':mapping['integration.patch'],
    'omittedRawModelCache':{'originalWorkPath':str(OUT.parent/'selected-v15/backend-fullprecision.npz'),
        'sha256':'20f0a6c0224ec7f640ca0e46275914e912a2b1b440882f0b1e9e94e8bb72556b','sizeBytes':95425411,
        'shape':[405,48,120,11],'dtype':'float32','retainedInArchive':False},
    'notes':'Fixture hash, checkpoint hash, thresholds and raw/quality decisions are bound in the comparison protocol. No recalibration or607 inference occurred. Historical source/evidence remains unchanged.'}
p=ARCHIVE/'preflight.json';p.write_text(json.dumps(preflight,indent=2)+'\n');payloads.append({'file':p.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes()),'sizeBytes':p.stat().st_size})
for ref in payloads:assert sha((ROOT/ref['file']).read_bytes())==ref['sha256']
index={'files':sorted(payloads,key=lambda x:x['file']),'payloadCount':len(payloads),'payloadBytes':sum(x['sizeBytes'] for x in payloads)}
ix=ARCHIVE/'archive-index.json';ix.write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'status':'passed','preflight_sha256':sha(p.read_bytes()),'index_sha256':sha(ix.read_bytes()),'payloadCount':len(payloads),'payloadBytes':index['payloadBytes'],'patch_sha256':sha((OUT/'integration.patch').read_bytes())},indent=2))
