// Executes the actual app decoder and image-quality helpers on cached model outputs.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Buffer } = require('node:buffer');
const { load, root } = require('../word-native-parity/geometry-reference.cjs');

const directory = process.argv[2];
const destination = process.argv[3];
if (!directory || !destination) throw new Error('Usage: node compare.cjs <prepared-directory> <report.json>');
assert.equal(os.endianness(), 'LE', 'This raw tensor harness expects a little-endian host');
const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
function fileSha(file) {
  const digest = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  const bytes = Buffer.alloc(1024 * 1024);
  try {
    let count;
    while ((count = fs.readSync(fd, bytes)) > 0) digest.update(bytes.subarray(0, count));
  } finally { fs.closeSync(fd); }
  return digest.digest('hex');
}
const referencePath = path.join(directory, 'python-reference.json');
const reference = JSON.parse(fs.readFileSync(referencePath, 'utf8'));
const { decodeWordOutput } = load(path.join(root, 'src/ml/word-decoder.ts'));
const { convertWordRawPixels, assessWordImageQuality } = load(path.join(root, 'src/ml/word-image-processing.ts'));
const { isValidWordModelThresholds } = load(path.join(root, 'src/ml/word-model-contract.ts'));
assert.ok(isValidWordModelThresholds(reference.thresholds));
const outputPath = path.join(directory, 'outputs.f32le');
const inputPath = path.join(directory, 'inputs.rgb');
assert.equal(fileSha(outputPath), reference.raw_outputs_sha256);
assert.equal(fileSha(inputPath), reference.raw_inputs_sha256);
for (const [file, expected] of Object.entries(reference.source_sha256)) assert.equal(fileSha(path.join(root, file)), expected, `Python source changed after replay: ${file}`);
const outputs = fs.openSync(outputPath, 'r');
const inputs = fs.openSync(inputPath, 'r');
const outputBytes = Buffer.alloc(48 * 120 * 11 * 4);
const inputBuffer = new ArrayBuffer(480 * 192 * 3);
const inputBytes = new Uint8Array(inputBuffer);
const results = [];
try {
  for (const row of reference.rows) {
    assert.equal(fs.readSync(outputs, outputBytes), outputBytes.length, 'Missing cached output bytes');
    assert.equal(fs.readSync(inputs, inputBytes), inputBytes.length, 'Missing cached input bytes');
    const tensor = new Float32Array(outputBytes.buffer, outputBytes.byteOffset, outputBytes.length / 4);
    const image = convertWordRawPixels({ width: 480, height: 192, pixelFormat: 'RGB', buffer: inputBuffer });
    const quality = assessWordImageQuality(image) || '';
    const [x, y, right, bottom] = row.selection;
    let decoded;
    try {
      decoded = decodeWordOutput(tensor, reference.thresholds, { x, y, width: right - x, height: bottom - y });
    } catch (error) { decoded = { status: 'error', reason: String(error), text: '' }; }
    const decoderText = decoded.status === 'success' ? decoded.text : '';
    const decoderReason = decoded.status === 'success' ? '' : decoded.reason;
    const text = quality ? '' : decoderText;
    const reason = quality || decoderReason;
    results.push({ index: row.index, image: row.image, expected: row.expected, typescript_text: text, python_text: row.python_text, typescript_reason: reason, python_reason: row.python_reason, quality_bypassed_decoder: !!quality, saved_output_is_zero_placeholder: row.saved_output_is_zero_placeholder, quality_match: quality === row.python_quality_reason, decoder_text_match: decoderText === row.python_decoder_text, decoder_reason_match: decoderReason === row.python_decoder_reason, pipeline_text_match: text === row.python_text, pipeline_reason_match: reason === row.python_reason });
  }
} finally { fs.closeSync(outputs); fs.closeSync(inputs); }
// New archives retain real tensors even when runtime image quality skips decoding.
// Decode those tensors here as extra parity cases; exclude only old zero placeholders.
const actualDecoderRows = results.filter((row) => !row.saved_output_is_zero_placeholder);
const mismatches = results.filter((row) => !row.quality_match || !row.pipeline_text_match || !row.pipeline_reason_match || (!row.saved_output_is_zero_placeholder && (!row.decoder_text_match || !row.decoder_reason_match)));
const trackedSources = ['src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts', 'src/ml/word-image-processing.ts', 'src/types/word-scan.ts', 'scripts/word-native-parity/geometry-reference.cjs', 'scripts/word-decoder-parity/compare.cjs'];
const report = { schema_version: 1, created_at_utc: new Date().toISOString(), scope: 'Actual TypeScript decoder and quality helpers on frozen public benchmark tensors/pixels in Node. No model rerun, native camera, or accuracy promotion.', model_sha256: reference.model_sha256, benchmark_sha256: reference.benchmark_sha256, fixture_manifest_sha256: reference.fixture_manifest_sha256, cached_npz_sha256: reference.cached_npz_sha256, python_reference_sha256: fileSha(referencePath), raw_outputs_sha256: reference.raw_outputs_sha256, raw_inputs_sha256: reference.raw_inputs_sha256, thresholds: reference.thresholds, source_sha256: { ...reference.source_sha256, ...Object.fromEntries(trackedSources.map((file) => [file, fileSha(path.join(root, file))])) }, archive_retains_raw_quality_rejects: reference.archive_retains_raw_quality_rejects === true, total_pipeline_cases: results.length, actual_model_output_decoder_cases: actualDecoderRows.length, quality_bypass_cases: results.filter((row) => row.quality_bypassed_decoder).length, zero_placeholder_cases: results.filter((row) => row.saved_output_is_zero_placeholder).length, quality_matches: results.filter((row) => row.quality_match).length, actual_decoder_text_matches: actualDecoderRows.filter((row) => row.decoder_text_match).length, actual_decoder_rejection_matches: actualDecoderRows.filter((row) => row.decoder_reason_match).length, pipeline_text_matches: results.filter((row) => row.pipeline_text_match).length, pipeline_rejection_matches: results.filter((row) => row.pipeline_reason_match).length, parity_passed: mismatches.length === 0, first_mismatches: mismatches.slice(0, 20), outcomes_sha256: sha(JSON.stringify(results)), limitations: ['Legacy archives replace precheck-rejected outputs with zeros; zero_placeholder_cases are excluded from actual model-output decoder counts. New raw-tensor archives include those extra decoder cases.', 'Agreement between decoders does not imply that their predicted words are correct.', 'Node replay does not establish physical-camera or native TFLite inference equivalence.'], rows: results };
fs.mkdirSync(path.dirname(destination), { recursive: true });
fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
const summary = { ...report }; delete summary.rows; delete summary.source_sha256;
process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
if (!report.parity_passed) process.exitCode = 1;
