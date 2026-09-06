"""Run the unchanged asset verifier on a separate prospective checkout."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_promotion(root, proposed, review_parent=Path('C:/')):
    root = root.resolve()
    for target, value in proposed.items():
        target.resolve().relative_to(root)
        if not isinstance(value, bytes):
            raise TypeError('Prospective files must contain exact bytes')
    stage = Path(tempfile.mkdtemp(prefix='bm-word-promotion-', dir=review_parent)).resolve()
    copied = {}
    directories = ('assets/models/braille-word-reader-v1', 'src', 'scripts', 'modules',
                   'training/evidence/braille-word-reader-v1', 'training/scripts', 'docs/evidence')
    files = [root / 'package.json']
    for directory in directories:
        files.extend(path for path in (root / directory).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts)
    # The verifier has one package import. Keep its exact installed implementation.
    files.extend(root / name for name in ('node_modules/typescript/package.json',
                                         'node_modules/typescript/lib/typescript.js'))
    for source in files:
        relative = source.relative_to(root)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[relative.as_posix()] = digest(target)
    for target, value in proposed.items():
        staged = stage / target.resolve().relative_to(root)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(value)
    result = subprocess.run(['node', str(stage / 'scripts/verify-word-model-assets.mjs')],
                            cwd=stage, capture_output=True, text=True, encoding='utf-8')
    log = stage / 'promotion-verifier.log'
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    changed = [file for file, expected in copied.items()
               if not (root / file).is_file() or digest(root / file) != expected]
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Exact prospective files checked with the unchanged current asset verifier before writing active evidence, model, manifest or Metro imports.',
        'exit_code': result.returncode,
        'verifier_sha256': copied['scripts/verify-word-model-assets.mjs'],
        'source_changed_during_review': changed,
        'source_files_sha256': copied,
        'proposed_files_sha256': {path.relative_to(root).as_posix(): hashlib.sha256(value).hexdigest()
                                   for path, value in proposed.items()},
        'log_sha256': digest(log),
        'status': 'passed' if result.returncode == 0 and not changed else 'failed',
    }
    (stage / 'promotion-review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if report['status'] != 'passed':
        raise ValueError(f'Prospective asset verification failed; active files untouched. Review: {stage}')
    return stage
