const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Buffer } = require('node:buffer');
const { test } = require('node:test');
const { root, load } = require('../word-native-parity/geometry-reference.cjs');
const contract = load(path.join(root, 'src/ml/word-model-contract.ts'));
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const driver = path.join(root, 'scripts/word-decoder-parity/replay-id.cjs');

function prepared() {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'braille-replay-id-test-'));
  const input = Buffer.alloc(480 * 192 * 3);
  for (let i = 0; i < input.length; i++) input[i] = i % 2 ? 192 : 64;
  const output = Buffer.alloc(48 * 120 * 11 * 4);
  fs.writeFileSync(path.join(directory, 'inputs.rgb'), input);
  fs.writeFileSync(path.join(directory, 'outputs.f32le'), output);
  const reference = { decoderVersion: contract.WORD_DECODER_VERSION, symbolProtocol: contract.WORD_SYMBOL_PROTOCOL,
    model_sha256: '0'.repeat(64), thresholds: contract.DEFAULT_WORD_MODEL_THRESHOLDS,
    raw_outputs_sha256: sha(output), raw_inputs_sha256: sha(input), source_sha256: {
      'training/scripts/word_symbols.py': sha(fs.readFileSync(path.join(root, 'training/scripts/word_symbols.py'))),
    }, rows: [{ index: 0, image: 'authored-blank.png', expected: '', expected_reject: true, selection: [0, 0, 1, 1],
      python_decoder_text: '', python_decoder_reason: 'no_grid', python_quality_reason: '', python_text: '', python_reason: 'no_grid' }] };
  fs.writeFileSync(path.join(directory, 'python-reference.json'), JSON.stringify(reference));
  return { directory, reference };
}
function run(directory, output = 'report.json') {
  return spawnSync(process.execPath, [driver, directory, path.join(directory, output)], { encoding: 'utf8', cwd: root });
}

test('replays an independently authored blank tensor and refuses report overwrite', () => {
  const { directory } = prepared();
  const result = run(directory);
  assert.equal(result.status, 0, result.stderr);
  const reportPath = path.join(directory, 'report.json'), original = fs.readFileSync(reportPath);
  const report = JSON.parse(original);
  assert.equal(report.parity_passed, true);
  assert.equal(report.correct, 1);
  assert.equal(report.compared_cell_semantics_and_boxes_cases, 0);
  assert.equal(report.rows[0].cell_semantics_and_boxes_match, null);
  assert.ok(report.source_sha256['src/ml/word-symbols.ts']);
  assert.ok(report.source_sha256['src/ml/word-tracking.ts']);
  assert.notEqual(run(directory).status, 0);
  assert.deepEqual(fs.readFileSync(reportPath), original);
});

test('records a semantic mismatch and exits unsuccessfully', () => {
  const { directory, reference } = prepared();
  reference.rows[0].python_text = 'a';
  fs.writeFileSync(path.join(directory, 'python-reference.json'), JSON.stringify(reference));
  assert.equal(run(directory).status, 1);
  const report = JSON.parse(fs.readFileSync(path.join(directory, 'report.json')));
  assert.equal(report.parity_passed, false);
  assert.equal(report.mismatch_count, 1);
});

test('rejects stale Python source or truncated input before creating a proof', () => {
  for (const defect of ['source', 'input']) {
    const { directory, reference } = prepared();
    if (defect === 'source') reference.source_sha256['training/scripts/word_symbols.py'] = '0'.repeat(64);
    else fs.writeFileSync(path.join(directory, 'inputs.rgb'), Buffer.from([0]));
    fs.writeFileSync(path.join(directory, 'python-reference.json'), JSON.stringify(reference));
    assert.notEqual(run(directory).status, 0);
    assert.equal(fs.existsSync(path.join(directory, 'report.json')), false);
  }
});
