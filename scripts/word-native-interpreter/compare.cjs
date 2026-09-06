// Compare actual Android C API tensors and actual app decoder decisions with desktop TFLite.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { load, root } = require('../word-native-parity/geometry-reference.cjs');

const [bundle, native, destination] = process.argv.slice(2);
if (!bundle || !native || !destination) throw new Error('Usage: node compare.cjs <prepared-bundle> <native-output> <report.json>');
if (fs.existsSync(destination)) throw new Error('Comparison report already exists; preserve original evidence');
assert.equal(os.endianness(), 'LE');
const sha = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const manifest = read(path.join(bundle, 'manifest.json'));
const execution = read(path.join(native, 'execution.json'));
assert.equal(manifest.status, 'trained_candidate_ready_native_execution_pending');
assert.equal(manifest.cases.length, 6, 'This preflight requires the six frozen TRAIN/analytic inputs');
assert.equal(execution.status, 'native_process_completed_comparison_pending');
assert.equal(execution.returncode, 0);
assert.equal(execution.bundle_manifest_sha256, sha(path.join(bundle, 'manifest.json')));
assert.equal(execution.model_sha256, manifest.model_sha256);
for (const [base, rows] of [[bundle, manifest.files], [native, execution.files]]) {
  for (const row of rows) {
    const file = path.join(base, row.path);
    assert.equal(fs.statSync(file).size, row.bytes);
    assert.equal(sha(file), row.sha256, `Changed evidence ${row.path}`);
  }
}
assert.equal(sha(path.join(bundle, 'model.tflite')), manifest.model_sha256);
const events = fs.readFileSync(path.join(native, 'native-stdout.jsonl'), 'utf8').trim().split(/\r?\n/).map(JSON.parse);
assert.equal(events[0].kind, 'runtime');
assert.equal(events[0].options, 'c_api_defaults_no_setters');
assert.deepEqual(events[0].input_shape, [1, 192, 480, 3]);
assert.deepEqual(events[0].output_shape, [1, 48, 120, 11]);
assert.equal(events.at(-1).kind, 'complete');
assert.equal(events.at(-1).cases, manifest.cases.length);
assert.equal(events.at(-1).status, 'passed_inference_only');
const cases = events.filter((event) => event.kind === 'case');
assert.equal(cases.length, manifest.cases.length);
assert.equal(events.length, manifest.cases.length + 2, 'Unexpected native runtime/completion records');
const expectedOutputNames = manifest.cases.map((row, index) => `${String(index).padStart(4, '0')}.f32`);
assert.deepEqual(fs.readdirSync(path.join(native, 'outputs')).sort(), expectedOutputNames);
const { decodeWordOutput } = load(path.join(root, 'src/ml/word-decoder.ts'));
const { isValidWordModelThresholds } = load(path.join(root, 'src/ml/word-model-contract.ts'));
assert.ok(isValidWordModelThresholds(manifest.thresholds));
assert.equal(manifest.absolute_tolerance, 1e-4);
assert.equal(manifest.relative_tolerance, 1e-4);
const tensor = (file) => {
  const bytes = fs.readFileSync(file);
  assert.equal(bytes.length, 48 * 120 * 11 * 4);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
};
const decision = (result) => ({ status: result.status, text: result.text ?? '', reason: result.reason ?? '',
  masks: result.cells.map((cell) => cell.mask) });
const rows = manifest.cases.map((row, index) => {
  const event = cases[index];
  assert.equal(event.index, index);
  assert.equal(row.index, index);
  assert.equal(event.file, expectedOutputNames[index]);
  assert.equal(event.invoke_status, 0);
  assert.equal(event.input_bytes, 192 * 480 * 3 * 4);
  assert.equal(event.output_bytes, 48 * 120 * 11 * 4);
  assert.equal(event.all_finite, true);
  const input = path.join(bundle, row.input);
  const expectedPath = path.join(bundle, row.desktop_output);
  const actualPath = path.join(native, 'outputs', `${String(index).padStart(4, '0')}.f32`);
  assert.equal(sha(input), row.input_sha256);
  assert.equal(sha(expectedPath), row.desktop_output_sha256);
  assert.equal(execution.transferred_files_verified_on_device.find((file) => file.path === row.input).sha256, row.input_sha256);
  const expected = tensor(expectedPath), actual = tensor(actualPath);
  let maxAbsolute = 0, failedValues = 0;
  const maxByChannel = Array(11).fill(0);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]) && Number.isFinite(expected[i]));
    const difference = Math.abs(actual[i] - expected[i]);
    maxAbsolute = Math.max(maxAbsolute, difference);
    maxByChannel[i % 11] = Math.max(maxByChannel[i % 11], difference);
    if (difference > 1e-4 + 1e-4 * Math.abs(expected[i])) failedValues++;
  }
  const expectedDecision = decision(decodeWordOutput(expected, manifest.thresholds, row.selection));
  const actualDecision = decision(decodeWordOutput(actual, manifest.thresholds, row.selection));
  const decisionMatches = JSON.stringify(expectedDecision) === JSON.stringify(actualDecision);
  return { index, identity: row.identity, input_sha256: row.input_sha256,
    desktop_output_sha256: row.desktop_output_sha256, native_output_sha256: sha(actualPath),
    max_absolute_difference: maxAbsolute, max_absolute_difference_by_channel: maxByChannel,
    values_outside_tolerance: failedValues, tensor_parity_passed: failedValues === 0,
    desktop_decision: expectedDecision, native_decision: actualDecision,
    decoder_decision_matches: decisionMatches, invoke_ms: event.invoke_ms };
});
const sources = ['src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts', 'src/types/word-scan.ts',
  'src/types/live-scan.ts', 'scripts/word-native-parity/geometry-reference.cjs', 'scripts/word-native-interpreter/compare.cjs'];
const passed = rows.every((row) => row.tensor_parity_passed && row.decoder_decision_matches);
const report = { schema_version: 1, status: passed ? 'passed' : 'failed',
  scope: 'Actual Android LiteRT C API inference on six TRAIN/analytic inputs; actual TypeScript whole-image decoder replay in Node.',
  model_sha256: manifest.model_sha256, execution_sha256: sha(path.join(native, 'execution.json')),
  bundle_manifest_sha256: sha(path.join(bundle, 'manifest.json')),
  runtime: events[0], peak_rss_kib: events.at(-1).peak_rss_kib,
  observed_model_operators: manifest.observed_model_operators,
  absolute_tolerance: 1e-4, relative_tolerance: 1e-4, thresholds: manifest.thresholds,
  samples: rows.length, tensor_parity_passed: rows.filter((row) => row.tensor_parity_passed).length,
  decoder_decisions_matched: rows.filter((row) => row.decoder_decision_matches).length,
  source_sha256: Object.fromEntries(sources.map((name) => [name, sha(path.join(root, name))])), rows,
  limits: ['C API execution does not exercise Java Interpreter, Nitro/JS bridge, worklets, camera, loader or main APK.',
    'Six TRAIN/analytic inputs do not replace full405 candidate parity, whole-word benchmark, physical or accessibility evidence.',
    'Whole-image decoder decisions are compared without RGB8 rounding or image-quality precheck; this is not camera pipeline replay.',
    'The diagnostic cannot promote a model or change recognition thresholds.'] };
fs.mkdirSync(path.dirname(destination), { recursive: true });
fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
process.stdout.write(JSON.stringify({ status: report.status, samples: report.samples,
  tensor_parity_passed: report.tensor_parity_passed, decoder_decisions_matched: report.decoder_decisions_matched }, null, 2) + '\n');
if (!passed) process.exitCode = 1;
