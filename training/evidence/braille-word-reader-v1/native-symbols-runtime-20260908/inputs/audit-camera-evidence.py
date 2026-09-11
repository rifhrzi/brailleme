"""Read-only source/evidence audit; writes only a new dated output report."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
KEY = 'src/ml/word-camera-geometry.ts'
STAGE = Path('C:/bm-symbols-apk-20260908/evidence')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(data.decode('utf-16') if data[:2] in (b'\xff\xfe', b'\xfe\xff') else data.decode('utf-8-sig'))


manifest_path = ROOT / 'assets/models/braille-word-reader-v1/manifest.json'
manifest = read(manifest_path)
reports = []
for name, entry in manifest['evidence'].items():
    path = ROOT / entry['file']
    reports.append(dict(name=name, **entry, actual_sha256=sha(path), contains_camera_geometry_path=KEY in path.read_text()))
assert all(row['sha256'] == row['actual_sha256'] for row in reports)
assert not any(row['contains_camera_geometry_path'] for row in reports)
reference = manifest['nativeSelfTest']['runtime_input_reference']['source_sha256']
assert KEY not in reference

retention_path = ROOT / 'training/checkpoints/retained-models-20260908.json'
retention_rows = []
def scan(value):
    if isinstance(value, dict):
        if 'file' in value and 'sha256' in value and value.get('pathScope') == 'repository_relative':
            retention_rows.append(value)
        for child in value.values():
            scan(child)
    elif isinstance(value, list):
        for child in value:
            scan(child)
scan(read(retention_path))
mismatches = [row['file'] for row in retention_rows if sha(ROOT / row['file']) != row['sha256']]
integrity_path = STAGE / 'apk-integrity-v4/report.json'
integrity = read(integrity_path)
checks_dir = ROOT / 'output/update-symbols-tracking-20260908/checks-1788887739700'
log_path = STAGE / 'actions-prepare-v4/20260908T172048267394Z-prepare-02-909348.log'
log = log_path.read_text()
report = {
    'schema_version': 1,
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only audit of current linked evidence and retained v4 runtime observations; no inference, ADB, source edits or acceptance claim.',
    'audited_module': {'path': KEY, 'sha256_at_audit': sha(ROOT / KEY), 'note': 'Concurrent geometry implementation may change this file after this audit.'},
    'manifest': {'path': str(manifest_path), 'sha256': sha(manifest_path)},
    'linked_model_evidence': reports,
    'native_startup_reference_source_keys': list(reference),
    'verifier': {'path': 'scripts/verify-experimental-word-model-assets.mjs', 'sha256': sha(ROOT / 'scripts/verify-experimental-word-model-assets.mjs'), 'camera_geometry_explicitly_required': False},
    'history': [
        {'path': 'training/evidence/braille-word-reader-v1/preview-photo-geometry-source-verification.json', 'binding': 'applicationFiles records an earlier source hash; historical source/compile proof, not active model evidence'},
        {'path': 'training/evidence/braille-word-reader-v1/native-camera-capture-diagnostic-v3/report.json', 'binding': 'frozen copied source and bundle; its archive verifier checks archived sources, not current app geometry. replay_sources bind only README and verify_archive.py'},
        {'path': 'training/evidence/braille-word-reader-v1/indonesian-symbols-baseline-20260908/baseline-freeze.json', 'binding': 'historical pre-change hash inventory; never relabel as current source'},
    ],
    'regeneration_if_only_camera_geometry_ts_changes': {
        'model_evidence_to_regenerate': [],
        'required_fresh_evidence': ['geometry unit tests and tracking/camera integration checks', 'lint/typecheck and final source coverage', 'new build and staged-source snapshot', 'new APK integrity and installed APK hash', 'actual main-app frame/preview/photo mapping and tracking smoke, then requested runtime/performance cases'],
        'historical_evidence_action': 'Preserve original reports and hashes. Write new camera runtime/source evidence in a new directory.',
        'limits': 'Existing nativeFixtureGeometry is image crop/decoder replay with mocked native pixels, not actual camera-to-frame mapping.',
    },
    'if_other_files_change': {
        'HybridWordPhotoGeometry.kt_or_WordPixelResize.java_or_WordPhotoGeometry.nitro.ts': 'Regenerate nativePixelParity with actual changed native implementation, then update linked report/file hash. TS/Python model replay remains unchanged if its own sources/model unchanged.',
        'word-image-processing.ts_or_word-tracking.ts': 'Regenerate 405/516 TS replay, nativeFixtureGeometry/runtime startup reference and nativePixelParity because these source closures include them. Review all source maps again.',
        'word-decoder.ts_or_word-symbols.ts_or_types/word-scan.ts': 'Regenerate source-bound semantic/405/516/native fixture/native pixel evidence; changed semantics additionally require aligned Python evaluation and decoder parity.',
    },
    'v4_source_checks': {
        'directory': str(checks_dir), 'checks_sha256': sha(checks_dir / 'checks.json'), 'checks': read(checks_dir / 'checks.json'),
        'suites_passed': 38, 'tests_passed': 656,
        'coverage_percent': {'statements': 91.18, 'branches': 86.06, 'functions': 90.03, 'lines': 92.26},
        'camera_geometry_coverage_percent': {'statements': 93.24, 'branches': 93.2, 'functions': 100, 'lines': 98.01},
        'coverage_log_sha256': sha(checks_dir / 'test-coverage.log'),
        'source_identity_limit': 'checks.json timestamps/log hashes do not independently bind every test source into the APK. staged-source/build/integrity are separate evidence.',
    },
    'v4_runtime': {
        'integrity_report': str(integrity_path), 'integrity_report_sha256': sha(integrity_path),
        'integrity_passed': integrity['integrity_passed'], 'integrity_checks_passed': sum(v is True for v in integrity['checks'].values()),
        'integrity_source_keys': list(integrity['source_sha256']),
        'integrity_source_scope': 'Model manifest, TFLite and startup PNG only; no app source-to-bundle assertion.',
        'build': read(STAGE / 'build-20260908T171709Z/build.json'),
        'install': read(STAGE / 'apk-integrity-v4/install.json'),
        'log': str(log_path), 'log_sha256': sha(log_path),
        'observed_lines': [line.strip() for line in log.splitlines() if 'Update Preview stream state to STREAMING' in line or 'PreviewView started!' in line or 'BRAILLE_WORD_TRACKING_' in line],
        'summary': 'Camera preview streams, but one admitted frame fails preview_outside_frame at frame-transform (480x640/up); the UI shows Pindai ulang without a result or enabled shutter.',
        'not_demonstrated': ['successful tracking', 'successful capture-to-result', 'ten-minute preview/memory stability', '24 FPS target', 'p95 capture-to-result <=2s', 'physical camera accuracy'],
    },
    'retention': {
        'historical_index': str(retention_path), 'sha256': sha(retention_path), 'repository_file_rows_verified': len(retention_rows), 'mismatches': mismatches,
        'action': 'Do not rewrite this dated index. Add a later companion index for final runtime archive/index hashes, APK delivery hash/path, source/build evidence and final documentation; retain prior indexes as history.',
    },
}
target = OUT / 'camera-geometry-evidence-audit-v1.json'
with target.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps({'report': str(target), 'sha256': sha(target), 'model_evidence_entries': len(reports), 'retention_rows': len(retention_rows)}))
