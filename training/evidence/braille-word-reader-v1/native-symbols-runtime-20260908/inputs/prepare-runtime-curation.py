"""Prepare a hash-bound runtime archive plan; copying requires a separate explicit command."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import stat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE = Path('C:/bm-symbols-apk-20260908/evidence')
TARGET = REPO/'training/evidence/braille-word-reader-v1/native-symbols-runtime-20260908'
RETAINED = REPO/'training/checkpoints/retained-models-20260908.json'
RETAINED_SHA = '29890d735d734184135c6e725a5065e46c67c827678bf66cb04534e9c53843ed'
spec = importlib.util.spec_from_file_location('runtime_curator', HERE/'curate-runtime.py')
curator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curator)

# Directories retain their original contents; curator indexes rather than copies
# packages and files over10MB. No parent stage/output tree is selected wholesale.
STAGE_INPUTS = [
    'apk-integrity-v6', 'build-20260908T173305Z', 'runtime-600-v6-complete',
    'actions-600-v6', 'timing-600-v6', 'native-pixel-resize-android-v1.json',
    'experimental-verifier-baseline-v1.json', 'experimental-verifier-baseline-v1.log',
    'runtime-host-v1.json', 'v6-tracker-live.png', 'v6-first-photo.png',
    'v6-first-photo.log', 'diagnostic-v6-live.log',
    'calibrated-feeds-v1', 'labelled-feeds-v1', 'camera-word.png',
    'camera-word-source.json', 'camera-word-large.png', 'camera-word-large-source.json',
    'emulator-20260908T171229943Z', 'emulator-feed-transition-v1.json',
    'calibrated-boot-v1', 'feed-preflight', 'ui-dump-no-idle-build-v3',
    'no-idle-v3-first.xml', 'no-idle-first-attempt.log',
    'actions-prepare-v3', 'actions-prepare-v4', 'actions-prepare-v5',
    'diagnostic-v3-crash.log', 'tracking-hook-before-diagnostic-v2.ts',
    'tracking-hook-diagnostic-v3.ts', 'camera-geometry-v4.txt',
    'runtime-600-v6', 'actions-cold-v6', 'system-ui-anr-v1', 'boot-anr-events-v1.log',
    'replaced-delivery-v2', 'png-crunch-diagnostic-v1.json',
]
WORK_INPUTS = [
    'apk-delivery.json', 'apk-delivery-history-v6-before-runtime-complete.json',
    'checks-1788888766290', 'helper-tests-final.json',
    'runtime-600-analysis-v6.json', 'runtime-600-independent-review-v6.json',
    'initial-preview-samples-v6.jsonl', 'analyze-initial-preview-v6.py',
    'analyze-runtime.py', 'test-analyze-runtime.py', 'runtime-protocol-v1.json',
    'timed-actions-v6.cjs', 'runtime-actions.py', 'test-runtime-actions.py',
    'runtime-actions-tests-v4', 'RUNTIME-ACTIONS.md', 'ui-dump-no-idle',
    'worklet-two-hop-reproduction-v1.json', 'reproduce-worklet-two-hop.mjs',
    'camerax-transform-analysis', 'camera-geometry-evidence-audit-v1.json',
    'v6-pixel-geometry-review.json', 'overlay-review-v6-tracking-masked-v2.json',
    'overlay-review-v6-first-photo-v1.json', 'v6-localization-roundtrip.json',
    'v6-sensor-projection-check.json', 'audit-camera-evidence.py',
    'audit-localization-roundtrip.cjs', 'summarize-v6-pixel-review.py',
    'review-overlay-pixels.py', 'review-overlay-pixels-green-only-v1.py',
    'test-review-overlay-pixels.py', 'prepare-calibrated-feeds.py',
    'prepare-runtime-cases.py', 'feed-mapping-v1.json', 'estimate-feed-mapping.py',
    'labelled-feed-review-v1.json', 'review-labelled-feeds.py', 'start-emulator.ps1',
    'stage-source.cjs', 'build-internal.ps1', 'run-checks.cjs', 'install-verified.ps1',
    'promote-verified-v6.ps1', 'curate-runtime.py', 'test-curate-runtime.py',
    'prepare-runtime-curation.py', 'test-prepare-runtime-curation.py',
    'orientation-actions.py', 'test-orientation-actions.py', 'RUNTIME-CURATION.md',
    'baseline/word-photo-camera.tsx', 'baseline/word-photo-camera.test.tsx',
    'baseline/word-scan-screen.tsx', 'baseline/word-scan-screen.test.tsx',
    'baseline/pre-existing.patch', 'baseline/protected-hashes.json',
    'final-preservation-audit-v6.json', 'v6-localization-source-audit.json',
    'approximate-v6-center-window-replay.json',
]
REPO_INPUTS = [
    'output/stage-source-binding-v6.json', 'output/stage-source-binding-v5.json',
    'output/stage-source-binding-v4.json', 'output/initial-preview-analysis-v6.json',
    'scripts/word-android-smoke/collect_runtime.py',
    'scripts/word-android-smoke/test_collect_runtime.py',
    'scripts/word-android-smoke/smoke.py', 'scripts/word-android-smoke/test_smoke.py',
    'scripts/word-android-smoke/verify_apk.py', 'scripts/word-android-smoke/test_verify_apk.py',
]
MODEL_SUFFIXES = {'.tflite', '.keras', '.h5', '.onnx', '.npz', '.npy'}


def scan_inputs(inputs):
    names, source_rows, files = set(), [], []
    for supplied in inputs:
        if not supplied.is_absolute() or '..' in supplied.parts:
            raise curator.EvidenceError('Every input must be absolute without parent traversal')
        source, info = curator._existing(supplied)
        if source.name.casefold() in names:
            raise curator.EvidenceError('Duplicate input basename: ' + source.name)
        if source.name == 'emulator-runtime.json':
            raise curator.EvidenceError('The75MB reflection dump is excluded from this compact plan')
        names.add(source.name.casefold())
        records = curator._scan(source) if stat.S_ISDIR(info.st_mode) else [(source, Path(), info)]
        source_files = []
        for path, relative, entry in records:
            if stat.S_ISDIR(entry.st_mode):
                continue
            if path.suffix.lower() in MODEL_SUFFIXES or path.name == 'emulator-runtime.json':
                raise curator.EvidenceError('Model duplicates/large reflection dump are not selected: ' + str(path))
            reason = ('file_exceeds_10_mb' if entry.st_size > curator.MAX_FILE_BYTES else
                      'excluded_application_package' if path.suffix.lower() in curator.PACKAGE_SUFFIXES else None)
            record = {'source': str(path), 'input_basename': source.name,
                      'relative': relative.as_posix(), 'bytes': entry.st_size,
                      'sha256': curator._hash(path), 'skip_reason': reason}
            files.append(record)
            source_files.append(record)
        source_rows.append({'source': str(source), 'basename': source.name,
                            'file_count': len(source_files),
                            'estimated_copy_bytes': sum(row['bytes'] for row in source_files if row['skip_reason'] is None),
                            'indexed_only_bytes': sum(row['bytes'] for row in source_files if row['skip_reason'] is not None)})
    return source_rows, files


def make_plan(inputs):
    sources, files = scan_inputs(inputs)
    if curator._hash(RETAINED) != RETAINED_SHA:
        raise curator.EvidenceError('Existing retained-models index changed; stop and review')
    if TARGET.exists():
        raise curator.EvidenceError('Archive already exists; this plan never overwrites evidence')
    return {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_not_copied', 'output': str(TARGET), 'inputs': sources, 'files': files,
            'retained_model_index': {'file': str(RETAINED), 'sha256': RETAINED_SHA, 'modified': False},
            'summary': {'input_count': len(sources), 'file_count': len(files),
                        'estimated_copy_bytes': sum(row['bytes'] for row in files if row['skip_reason'] is None),
                        'indexed_only_bytes': sum(row['bytes'] for row in files if row['skip_reason'] is not None)},
            'pending_additions': ['Completed background/font/cold UI evidence and frozen single-feed runs are appended explicitly after root finishes QA.',
                                  'Regenerate a new plan after final delivery metadata changes; copy rejects drift from the selected hashes.'],
            'limitations': ['Sizes exclude the new curator index.json and this plan file.',
                            'No artifact copy, test acceptance, camera accuracy, release or physical validation is implied.',
                            'Do not select the75MB emulator-runtime.json, model weights/tensors, or entire parent trees.']}


def verify_plan(plan):
    if plan.get('schema_version') != 1 or plan.get('status') != 'prepared_not_copied' or plan.get('output') != str(TARGET):
        raise curator.EvidenceError('Unexpected plan schema/status/destination')
    current_sources, current_files = scan_inputs([Path(row['source']) for row in plan['inputs']])
    if current_sources != plan['inputs'] or current_files != plan['files']:
        raise curator.EvidenceError('Source tree/hash changed after planning; create a new plan')
    if curator._hash(RETAINED) != RETAINED_SHA:
        raise curator.EvidenceError('Existing retained-models index changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Read/hash only; write a new plan and proposed companion')
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--add-input', type=Path, action='append', default=[])
    copy = commands.add_parser('copy', help='Only run after root explicitly authorizes final archive creation')
    copy.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'copy':
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        verify_plan(plan)
        # Existing curator handles containment, reparse points, file changes,
        # package/size exclusions, byte-exact copies and atomic complete index.
        report = curator.curate([Path(row['source']) for row in plan['inputs']] + [args.plan.resolve()], TARGET)
        print(json.dumps({'output': str(TARGET), **report['summary'], 'acceptanceInferred': False}))
        return
    plan_path = args.plan.resolve()
    companion = plan_path.with_name(plan_path.stem + '-retention-proposed.json')
    if plan_path.exists() or companion.exists():
        raise curator.EvidenceError('Use new plan/companion paths; never overwrite previous plans')
    inputs = [STAGE/name for name in STAGE_INPUTS] + [HERE/name for name in WORK_INPUTS] + [REPO/name for name in REPO_INPUTS] + args.add_input
    plan = make_plan(inputs)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2); stream.write('\n')
    proposed = {'schemaVersion': 1, 'status': 'proposed_archive_not_copied',
                'purpose': 'Additive retention of runtime evidence; not a deletion or model replacement authorization.',
                'modelRetentionIndex': {'file': 'training/checkpoints/retained-models-20260908.json', 'sha256': RETAINED_SHA,
                                       'relationship': 'Existing index remains unchanged; preserve the union of its files and the final runtime companion.'},
                'runtimeEvidence': {'directory': TARGET.relative_to(REPO).as_posix(), 'preserveEntireDirectory': True,
                                    'indexSha256': None, 'status': 'creation_pending'},
                'plan': {'file': str(plan_path), 'sha256': curator._hash(plan_path)},
                'finalization': 'After authorized copy, bind actual index.json hash in a new final companion; keep this proposal and original model index unchanged.',
                'physicalCameraValidated': False, 'acceptanceInferred': False}
    with companion.open('x', encoding='utf-8') as stream:
        json.dump(proposed, stream, indent=2); stream.write('\n')
    print(json.dumps({'plan': str(plan_path), 'proposed_companion': str(companion), **plan['summary'], 'archive_created': False}))


if __name__ == '__main__':
    main()
