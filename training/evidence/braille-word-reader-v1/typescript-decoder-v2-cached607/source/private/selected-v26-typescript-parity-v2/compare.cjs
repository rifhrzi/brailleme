// Executes the actual app decoder and image-quality helpers on cached model outputs.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Buffer } = require('node:buffer');
const { load, root } = require(path.join(process.cwd(), 'scripts/word-native-parity/geometry-reference.cjs'));

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
const protocolPath = path.join(root, reference.protocol_file);
assert.equal(fileSha(protocolPath), reference.protocol_sha256);
const protocol = JSON.parse(fs.readFileSync(protocolPath, 'utf8'));
for (const [file, expected] of Object.entries(protocol.source_sha256)) assert.equal(fileSha(path.join(root, file)), expected);
const boxTolerance = 1e-12;
let maxBoxError = 0, maxConfidenceError = 0;
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
    const pythonBoxes = row.python_decoder_cells.map((cell) => {
      const [left, top, right, bottom] = cell.box;
      const x = row.python_decoder_text ? left : Math.max(0, Math.min(1, left));
      const y = row.python_decoder_text ? top : Math.max(0, Math.min(1, top));
      return { x, y, width: (row.python_decoder_text ? right : Math.min(1, right)) - x,
        height: (row.python_decoder_text ? bottom : Math.min(1, bottom)) - y };
    }).filter((box) => box.width > 0 && box.height > 0);
    const boxes = decoded.boxes || [];
    let rowBoxError = 0;
    const boxesMatch = boxes.length === pythonBoxes.length && boxes.every((box, index) =>
      ['x', 'y', 'width', 'height'].every((key) => {
        const error = Math.abs(box[key] - pythonBoxes[index][key]);
        rowBoxError = Math.max(rowBoxError, error);
        return Number.isFinite(error) && error <= boxTolerance;
      }));
    maxBoxError = Math.max(maxBoxError, rowBoxError);
    const cells = decoded.cells || [];
    let rowConfidenceError = 0;
    const cellsMatch = row.python_decoder_text ? cells.length === row.python_decoder_cells.length && cells.every((cell, index) => {
      const expected = row.python_decoder_cells[index];
      const error = Math.abs(cell.confidence - Math.min(expected.score, expected.dot_confidence));
      rowConfidenceError = Math.max(rowConfidenceError, error);
      return cell.letter === row.python_decoder_text[index] && cell.mask === expected.mask && Number.isFinite(error) && error <= boxTolerance;
    }) : cells.length === 0;
    maxConfidenceError = Math.max(maxConfidenceError, rowConfidenceError);
    results.push({ decoder_boxes_match: boxesMatch, decoder_cell_order_match: cellsMatch,
      box_max_abs_error: rowBoxError, confidence_max_abs_error: rowConfidenceError,
      typescript_boxes: boxes, typescript_cells: cells, index: row.index, image: row.image, expected: row.expected, typescript_text: text, python_text: row.python_text, typescript_reason: reason, python_reason: row.python_reason, quality_bypassed_decoder: !!quality, saved_output_is_zero_placeholder: row.saved_output_is_zero_placeholder, quality_match: quality === row.python_quality_reason, decoder_text_match: decoderText === row.python_decoder_text, decoder_reason_match: decoderReason === row.python_decoder_reason, pipeline_text_match: text === row.python_text, pipeline_reason_match: reason === row.python_reason });
  }
} finally { fs.closeSync(outputs); fs.closeSync(inputs); }
// New archives retain real tensors even when runtime image quality skips decoding.
// Decode those tensors here as extra parity cases; exclude only old zero placeholders.
const actualDecoderRows = results.filter((row) => !row.saved_output_is_zero_placeholder);
const mismatches = results.filter((row) => !row.decoder_boxes_match || !row.decoder_cell_order_match || !row.quality_match || !row.pipeline_text_match || !row.pipeline_reason_match || (!row.saved_output_is_zero_placeholder && (!row.decoder_text_match || !row.decoder_reason_match)));
const trackedSources = ['src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts', 'src/ml/word-image-processing.ts', 'src/types/word-scan.ts', 'scripts/word-native-parity/geometry-reference.cjs', 'scripts/word-decoder-parity/compare.cjs'];
const report = { schema_version: 1, created_at_utc: new Date().toISOString(), scope: 'Actual TypeScript decoder and quality helpers on frozen public benchmark tensors/pixels in Node. No model rerun, native camera, or accuracy promotion.', model_sha256: reference.model_sha256, benchmark_sha256: reference.benchmark_sha256, fixture_manifest_sha256: reference.fixture_manifest_sha256, cached_npz_sha256: reference.cached_npz_sha256, python_reference_sha256: fileSha(referencePath), raw_outputs_sha256: reference.raw_outputs_sha256, raw_inputs_sha256: reference.raw_inputs_sha256, thresholds: reference.thresholds, source_sha256: { ...reference.source_sha256, ...Object.fromEntries(trackedSources.map((file) => [file, fileSha(path.join(root, file))])) }, archive_retains_raw_quality_rejects: reference.archive_retains_raw_quality_rejects === true, total_pipeline_cases: results.length, actual_model_output_decoder_cases: actualDecoderRows.length, quality_bypass_cases: results.filter((row) => row.quality_bypassed_decoder).length, zero_placeholder_cases: results.filter((row) => row.saved_output_is_zero_placeholder).length, quality_matches: results.filter((row) => row.quality_match).length, actual_decoder_text_matches: actualDecoderRows.filter((row) => row.decoder_text_match).length, actual_decoder_rejection_matches: actualDecoderRows.filter((row) => row.decoder_reason_match).length, pipeline_text_matches: results.filter((row) => row.pipeline_text_match).length, pipeline_rejection_matches: results.filter((row) => row.pipeline_reason_match).length, parity_passed: mismatches.length === 0, first_mismatches: mismatches.slice(0, 20), outcomes_sha256: sha(JSON.stringify(results)), limitations: ['Legacy archives replace precheck-rejected outputs with zeros; zero_placeholder_cases are excluded from actual model-output decoder counts. New raw-tensor archives include those extra decoder cases.', 'Agreement between decoders does not imply that their predicted words are correct.', 'Node replay does not establish physical-camera or native TFLite inference equivalence.'], rows: results };
for (const [file, expected] of Object.entries(protocol.source_sha256)) assert.equal(fileSha(path.join(root, file)), expected);
const peakBytes = process.resourceUsage().maxRSS * 1024;
assert.ok(peakBytes < 350 * 1024 * 1024, `Node exceeded memory budget: ${peakBytes}`);
Object.assign(report, { decoder_version: protocol.decoder_version, protocol_sha256: reference.protocol_sha256,
  decoder_box_matches: results.filter((row) => row.decoder_boxes_match).length,
  decoder_cell_order_matches: results.filter((row) => row.decoder_cell_order_match).length,
  box_absolute_tolerance: boxTolerance, box_max_abs_error: maxBoxError,
  confidence_max_abs_error: maxConfidenceError, peak_working_set_bytes: peakBytes,
  python_peak_working_set_bytes: reference.peak_working_set_bytes,
  source_hashes_verified_before_after: true });
fs.mkdirSync(path.dirname(destination), { recursive: true });
fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
const summary = { ...report }; delete summary.rows; delete summary.source_sha256;
process.stdout.write(JSON.stringify(summary, null, 2) + '\n');
if (!report.parity_passed) process.exitCode = 1;
