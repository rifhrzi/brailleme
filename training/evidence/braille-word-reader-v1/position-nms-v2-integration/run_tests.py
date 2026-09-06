"""Source-bound targeted checks of the actual integrated decoder, no inference."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
assert not (OUT / 'test-execution.json').exists()
OWNED = [
    'training/scripts/word_decode.py', 'src/ml/word-decoder.ts',
    'training/tests/word_test_predictions.py',
    'training/tests/test_word_decoder_positions.py',
    'training/tests/test_word_pipeline.py', 'training/tests/test_word_benchmark.py',
    'training/tests/test_word_selection.py', 'training/tests/test_word_fixture_geometry.py',
    'training/tests/test_word_visible_partial.py',
    'src/ml/__tests__/word-decoder.test.ts', 'src/ml/__tests__/word-decoder-positions.test.ts',
]
DEPENDENCIES = [
    'training/scripts/word_data.py', 'training/scripts/word_model.py',
    'training/scripts/word_select.py', 'training/scripts/word_benchmark.py',
    'training/scripts/word_fixture.py', 'training/scripts/word_train.py',
    'src/ml/word-model-contract.ts', 'src/ml/__tests__/fixtures/word-decoder-reference.json',
]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
before = {name: sha(ROOT / name) for name in OWNED + DEPENDENCIES}
for name in before:
    dest = OUT / 'source' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes((ROOT / name).read_bytes())
environment = dict(os.environ, KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cpu',
                   CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
modules = ['test_word_decoder_positions', 'test_word_pipeline', 'test_word_benchmark',
           'test_word_selection', 'test_word_fixture_geometry', 'test_word_visible_partial']
python_code = ('import sys, unittest; sys.path.insert(0, "training/tests"); '
               'suite=unittest.TestLoader().loadTestsFromNames(' + repr(modules) + '); '
               'result=unittest.TextTestRunner(verbosity=2).run(suite); '
               'sys.exit(0 if result.wasSuccessful() else 1)')
commands = {
    'python': [sys.executable, '-c', python_code],
    'jest': ['node', 'node_modules/jest/bin/jest.js', '--runInBand', '--no-cache',
             '--runTestsByPath', 'src/ml/__tests__/word-decoder.test.ts',
             'src/ml/__tests__/word-decoder-positions.test.ts'],
}
def execute(item):
    name, command = item
    started = datetime.now(timezone.utc).isoformat()
    with (OUT / (name + '.log')).open('wb') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    return name, {'command': command, 'started_utc': started,
                  'finished_utc': datetime.now(timezone.utc).isoformat(),
                  'exit_code': result.returncode, 'log': name + '.log',
                  'log_sha256': sha(OUT / (name + '.log'))}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    results = dict(executor.map(execute, commands.items()))
after = {name: sha(ROOT / name) for name in before}
report = {'schema_version': 1, 'scope': 'Actual source targeted Python/Jest checks; no ML inference',
          'source_sha256': before, 'source_unchanged_after_tests': after == before,
          'executions': results, 'script_sha256': sha(__file__)}
(OUT / 'test-execution.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
sys.exit(0 if after == before and all(row['exit_code'] == 0 for row in results.values()) else 1)
