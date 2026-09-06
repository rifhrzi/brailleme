"""Capture PRIVATE prototype proof with before/after source bindings."""
import contextlib
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

ROOT=Path.cwd().resolve();BASE=Path(__file__).resolve().parent;RUN=BASE.parent/'run25'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
config=json.loads((RUN/'training-config.json').read_text())
def active_hashes():return {name:sha(ROOT/'training/scripts'/name) for name in config['source_sha256']}
before=active_hashes();assert before==config['source_sha256']
config_hash=sha(RUN/'training-config.json')
preserved={p.name:sha(p) for p in RUN.iterdir() if p.is_file() and ('protocol' in p.name or p.name.endswith('-plan.json'))}
source_files={p.relative_to(BASE).as_posix():sha(p) for p in BASE.rglob('*.py') if '__pycache__' not in p.parts}
assert not (BASE/'proof-report.json').exists()
started=now();log=io.StringIO()
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
    spec=importlib.util.spec_from_file_location('private_aspect_tests',BASE/'test_private_aspect.py')
    tests=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tests;spec.loader.exec_module(tests)
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
completed=now()
after=active_hashes()
assert before==after==config['source_sha256']
assert sha(RUN/'training-config.json')==config_hash
assert {name:sha(RUN/name) for name in preserved}==preserved
assert all(sha(BASE/name)==expected for name,expected in source_files.items())
assert not any(name in sys.modules for name in ('keras','torch','tensorflow','h5py'))
(BASE/'stdout.log').write_text(log.getvalue(),encoding='utf-8')
report=dict(schema_version=1,status='passed' if result.wasSuccessful() else 'failed',
    scope='PRIVATE CPU NumPy/Pillow prototype only. No active source/model/profile change, no new training or selection protocol, no model inference or recognition accuracy claim.',
    started_at_utc=started,completed_at_utc=completed,tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    log_sha256=sha(BASE/'stdout.log'),source_sha256=source_files,active_training_sources_before=before,active_training_sources_after=after,
    run25_config_sha256=config_hash,preserved_run25_protocols_sha256=preserved,
    original_sources_exactly_match_run25=all(sha(BASE/'source-original'/name)==expected for name,expected in before.items()),
    spec_sha256=sha(BASE/'prototype-spec.json'),patch_sha256=sha(BASE/'private-edits.patch'),
    marker_oracle=dict(cases=len(tests.METRICS['marker_centroid_errors_px']),maximum_coordinate_error_px=max(tests.METRICS['marker_centroid_errors_px']),
        scope='Synthetic Gaussian centers only; actual Pillow bilinear affine at horizontal factors0.55 and1.25, all six slots, separate exact box-edge and +0.5/-0.5 point-center conventions.'),
    guarantees_tested=['Default OFF actual copied Patches pixels/targets/order exact across63 masks; width RNG helper never called while OFF.',
        'Existing zoom/y/translation/rotation/lighting RNG call sequences and values remain exact across18 seeds; width modifies only x and only when original affine gate is selected.',
        'Pixel affine invoked once; DSBI point-state affine receives exactly the same x/y scales and translations.',
        'Per-record stream deterministic and independent of batch grouping/order, legacy generator and global NumPy RNG; repeated record/epoch retains factor.',
        'All63 occupied masks retain slot identities; full targets remain original masks; unknown slots have no positive target.',
        'Clipped known DSBI fragments keep all six target bits unknown, original-outside/unoccupied/later-lost points never regain survival.'],
    limitations=['Synthetic points and images only; no real TRAIN image copies, held405 or external607 use.',
        'Private Patches flag has no CLI or training-config integration; running the normal app/trainer cannot activate this prototype.',
        'Intended profile is current illumination-only. Combining with existing camera-only aspect factor is outside this proposal and would compound factors.',
        'No evidence yet that horizontal aspect augmentation improves accuracy or avoids a precision regression. Future integration needs independent review and a separate frozen experiment after run25 selection.'],
    backend_modules_imported=[],numpy_version=tests.np.__version__,pillow_version=tests.Image.__version__)
(BASE/'proof-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
index={p.relative_to(BASE).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(BASE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='index.json'}
(BASE/'index.json').write_text(json.dumps(dict(scope=report['scope'],files=index),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=report['status'],tests=result.testsRun,report_sha256=sha(BASE/'proof-report.json'),index_sha256=sha(BASE/'index.json'),max_marker_error_px=report['marker_oracle']['maximum_coordinate_error_px'])))
sys.exit(0 if result.wasSuccessful() else 1)
