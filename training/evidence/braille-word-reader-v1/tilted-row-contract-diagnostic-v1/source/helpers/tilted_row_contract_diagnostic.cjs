// Execute the actual app decoder on synthetic ideal rotated-row tensors only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { load, root } = require(path.join(process.cwd(), 'scripts/word-native-parity/geometry-reference.cjs'));
const directory = path.join(root, '.braille-training-work/word-scan-v1-20260905/tilted-row-contract-diagnostic-v1');
const sha = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const reference = JSON.parse(fs.readFileSync(path.join(directory, 'python-reference.json'), 'utf8'));
for (const [name, expected] of Object.entries(reference.source_sha256)) assert.equal(sha(path.join(root, name)), expected);
assert.equal(sha(path.join(directory, 'protocol.json')), reference.protocol_sha256);
const { decodeWordOutput } = load(path.join(root, 'src/ml/word-decoder.ts'));
const { WORD_DECODER_VERSION, DEFAULT_WORD_MODEL_THRESHOLDS } = load(path.join(root, 'src/ml/word-model-contract.ts'));
assert.equal(WORD_DECODER_VERSION, reference.decoder_version);
assert.deepEqual(DEFAULT_WORD_MODEL_THRESHOLDS, reference.thresholds);
const rawFile = path.join(directory, 'outputs.f32le');
assert.equal(sha(rawFile), reference.raw_outputs_sha256);
const raw = fs.openSync(rawFile, 'r');
const buffer = Buffer.alloc(48 * 120 * 11 * 4);
const rows = [];
try {
  for (const row of reference.rows) {
    assert.equal(fs.readSync(raw, buffer), buffer.length);
    const result = decodeWordOutput(new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4), reference.thresholds);
    const text = result.status === 'success' ? result.text : '';
    const reason = result.status === 'success' ? '' : result.reason;
    assert.equal(text, row.actual_python_text);
    assert.equal(reason, row.actual_python_reason);
    assert.equal(result.boxes.length, row.length);
    rows.push({ length: row.length, angle_degrees: row.angle_degrees, expected_text: row.expected_text,
      predicted: text, rejection: reason, detected_boxes: result.boxes,
      center_y_span_pixels: row.center_y_span_pixels, row_span_limit_pixels: row.row_span_limit_pixels,
      aabb_width_pixels: row.aabb_width_pixels, aabb_height_pixels: row.aabb_height_pixels,
      neighbor_gap_ratio: row.neighbor_gap_ratio, all_size_gap_nms_edge_conditions_pass: true });
  }
} finally { fs.closeSync(raw); }
for (const [name, expected] of Object.entries(reference.source_sha256)) assert.equal(sha(path.join(root, name)), expected);
const report = { status: 'reproduced_decoder_tilt_tolerance', created_at_utc: new Date().toISOString(),
  scope: reference.scope, decoder_version: reference.decoder_version, protocol_sha256: reference.protocol_sha256,
  python_reference_sha256: sha(path.join(directory, 'python-reference.json')),
  source_sha256: reference.source_sha256, source_hashes_verified_before_after: true,
  actual_python_typescript_cases: rows.length, python_typescript_decisions_equal: true,
  tilted_single_row_rejections: rows.filter((row) => row.rejection === 'multiple_rows').length,
  interpretation: 'The row rule bounds global vertical center span, so sufficiently long straight rows rotated by five degrees are rejected as multiple_rows despite perfect confident nonoverlapping cells. Current UI explicitly requests a horizontal row; this quantifies decoder tilt tolerance and a possible physical usability limit, not automatically a defect requiring new semantics or evidence of the physical C/H/J cause or model benchmark accuracy.',
  limitations: ['No model, real image, camera, 405 validation or 607 benchmark was used.', 'Geometry is centered in the full guide; a tightly drawn camera guide can separately reject rotated cells as clipped.', 'No decoder change or threshold adjustment is proposed or applied by this diagnostic.'], rows };
fs.writeFileSync(path.join(directory, 'report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ cases: rows.length, python_typescript_decisions_equal: true,
  tilted_single_row_rejections: report.tilted_single_row_rejections, report_sha256: sha(path.join(directory, 'report.json')) }, null, 2));
