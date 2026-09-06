"""Archive original run27 continuation proofs, preserving run26's historical identity."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

root = Path.cwd()
work = root / '.braille-training-work/word-scan-v1-20260905'
run = work / 'run27'
output = root / 'training/evidence/braille-word-reader-v1/horizontal-aspect-continuation-preflight'
helper = work / 'archive_provenance.py'
original_preflight = root / 'training/evidence/braille-word-reader-v1/horizontal-aspect-preflight/preflight.json'
original_index = original_preflight.parent / 'archive-index.json'
assert not output.exists(), 'Use a fresh archive; original evidence is immutable'
tree = ast.parse(helper.read_text(encoding='utf-8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
args = SimpleNamespace(output=output)
exec(compile(ast.Module(body=functions, type_ignores=[]), str(helper), 'exec'), globals())

def link(path):
    return {'file': path.relative_to(root).as_posix(), 'sha256': sha(path)}

def verify_links(value):
    if isinstance(value, dict):
        if isinstance(value.get('file'), str) and isinstance(value.get('sha256'), str):
            assert sha(root / value['file']) == value['sha256'], value['file']
        for child in value.values():
            verify_links(child)
    elif isinstance(value, list):
        for child in value:
            verify_links(child)

old_preflight_bytes, old_index_bytes = original_preflight.read_bytes(), original_index.read_bytes()
ancestor = json.loads(old_preflight_bytes)
verify_links(ancestor)
old_index = json.loads(old_index_bytes)
for entry in old_index['files']:
    path = original_index.parent / entry['file']
    assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
config_path = run / 'training-config.json'
config = json.loads(config_path.read_bytes())
continuation = archive_horizontal_aspect_continuation(config, run)
assert continuation is not None and archive_horizontal_aspect(config, run) is None
assert ancestor['configuration']['sha256'] == continuation['inheritedRunConfigurationSha256']
assert ancestor['horizontalAspectPreflight']['protocol']['sha256'] == continuation['inheritedHorizontalAspectProtocolSha256']
ancestor['run'] = 'run26'
for name in ('sampling-history.json', 'sampling-lineage-final.json', 'validation-history.json'):
    ancestor[name] = archive(work / 'run26' / name, Path('run26') / name)
point_path = work / 'data' / config['dsbi_point_metadata']
assert sha(point_path) == config['data_sha256'][config['dsbi_point_metadata']]
point_link = archive(point_path, Path('point-metadata/points.json'))
ancestor['dsbiPointMetadata'] = point_link
report = {
    'scope': 'Run27 weight-only continuation provenance. Original prelaunch guard and parent HDF5 audit are retained byte-for-byte; this archive is prepared after launch and adds no model, accuracy, native, APK or physical-camera evidence. Run26 transform proof retains its original v15 initializer.',
    'run': 'run27',
    'configuration': archive(config_path, Path('run27/training-config.json')),
    'horizontalAspectContinuation': continuation,
    'sources': [],
    'dsbiPointMetadata': point_link,
    'ancestors': [ancestor],
    'originalHorizontalAspectPreflight': link(original_preflight),
    'originalHorizontalAspectArchiveIndex': link(original_index),
    'supportingProofs': [],
}
for name, digest in config['source_sha256'].items():
    source = run / 'source' / name
    assert sha(source) == digest
    report['sources'].append(archive(source, Path('run27/source') / name))
for name in ('run27-plan.json', 'launch-command.json', 'dsbi-addition-protocol.json',
             'dsbi-visible-partial-protocol.json', 'horizontal-aspect-continuation-protocol.json'):
    report[name] = archive(run / name, Path('run27') / name)
for path in (helper, Path(__file__).resolve(),
             work / 'horizontal-aspect-continuation-verifier-validation.json',
             work / 'horizontal-aspect-continuation-verifier-tests.log'):
    report['supportingProofs'].append(archive(path, Path('supporting-source') / path.name))
validation = json.loads((work / 'horizontal-aspect-continuation-verifier-validation.json').read_bytes())
assert validation['status'] == 'passed' and validation['model_execution'] is False
verify_links(validation)
for entry in validation['sources']:
    source = root / entry['file']
    report['supportingProofs'].append(archive(source, Path('supporting-source') / source.name))
verify_links(report)
assert original_preflight.read_bytes() == old_preflight_bytes and original_index.read_bytes() == old_index_bytes
target = output / 'preflight.json'
target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
(output / 'README.md').write_text(
    '# Run27 continuation preflight evidence\n\n'
    'The [preflight record](preflight.json) retains the run27 continuation protocol, independent command/reset guard, parent checkpoint HDF5 audit and exact 22 training sources. Its `ancestors` entry references the immutable [original run26 preflight](../horizontal-aspect-preflight/preflight.json), with separately retained run26 sampling history and lineage. Historical audits are not relabeled as tests of the new parent.\n\n'
    'The parent is run26 epoch 10 (`6113867b21316669169391fa7216f9d1b9bf147a13c153889ff6de1c26e9a4ea`), which was eligible but unselected against the v15 control. The protocol intentionally resets Adam, epoch/order cursors and random streams; it repeats the same augmentation realizations. Local epochs 8–10 must satisfy their own sampling cycle.\n\n'
    'This archive was assembled after launch from the original prelaunch records. No tests, model inference, training, conversion or physical-camera checks were rerun to assemble it. The retained verifier unit-test report is a separate source-integrity proof; real retained-data replay is appended separately. No active model or deliverable APK is established.\n\n'
    '`archive-index.json` hashes every file in this new directory; links to the original run26 tree are validated by its unchanged archive index. No checkpoint binary is included.\n', encoding='utf-8')
files = [{'file': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}
         for path in sorted(output.rglob('*')) if path.is_file()]
index = output / 'archive-index.json'
index.write_text(json.dumps({'scope': report['scope'], 'externalReferences': [link(original_preflight), link(original_index)], 'files': files}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'report': str(target), 'report_sha256': sha(target), 'archive_sha256': sha(index),
                  'files': len(files), 'bytes': sum(entry['bytes'] for entry in files), 'helper_sha256': sha(helper),
                  'retainer_sha256': sha(Path(__file__).resolve())}))
