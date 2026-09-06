"""Retain the demonstrated decoder defect and separate ideal-geometry audit."""
import hashlib
import json
from pathlib import Path

work = Path(__file__).resolve().parent
source = work / 'complete-cell-contract-audit'
target = work.parents[1] / 'training/evidence/braille-word-reader-v1/decoder-adjacent-grid-diagnostic'
expected = {
    'report.json': '7f90ce5b7a06ab901c4007d69c5c7aaef11bea7d894405e5461ccf542302b601',
    'ideal405/report.json': '7a4c3a8235a5aec9ace5f52d1de15a9c78cd96ff0b699c2b08b3fa91f9500e36',
}
digest = lambda data: hashlib.sha256(data).hexdigest()
payload = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
assert len(payload) == 12
for name, sha in expected.items():
    assert digest(payload[name]) == sha, name
if target.exists():
    raise SystemExit('Refuse to replace existing diagnostic evidence')
payload['README.md'] = (
    '# Adjacent grid centers can shorten a word\n\n'
    'This source-level diagnostic reproduces a decoder defect, without model inference. '
    'Two fully visible, disjoint 4 by 12 pixel cells at adjacent stride-4 sites encode AB, '
    'but the old 3 by 3 local-maximum filter accepts A. Equal and unequal high scores both reproduce it. '
    'Moving the second cell two grid sites away restores AB. Ordinary coordinate and A-Z bit roundtrips pass.\n\n'
    'The separate ideal405 audit finds zero adjacent or same-site distinct annotated pairs '
    'in the frozen validation set. It reads saved quality flags and cache identity metadata, '
    'not model predictions or image pixels. Therefore this latent small-cell defect is not '
    'a demonstrated cause of the current validation errors or physical C/H/J results.\n\n'
    'These reports precede any fix. Geometry and dot outputs outside supervised centers may '
    'be unreliable; a replacement postprocessor requires tests and actual cached-output comparison. '
    'No model, thresholds, benchmark, application source, or active assets were changed by this audit.\n'
).encode()
payload['retain.py'] = Path(__file__).read_bytes()
entries = []
for name, data in sorted(payload.items()):
    destination = target / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    assert destination.read_bytes() == data
    entries.append({'path': name, 'bytes': len(data), 'sha256': digest(data)})
index = {'scope': 'source defect plus separate frozen405 ideal geometry; no model inference', 'files': entries}
raw = (json.dumps(index, indent=2) + '\n').encode()
(target / 'archive-index.json').write_bytes(raw)
print(json.dumps({'files': len(entries), 'bytes': sum(e['bytes'] for e in entries), 'index_sha256': digest(raw)}))
