// Diagnostic ideal-annotation decoder feasibility, never model accuracy.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { Buffer } = require('node:buffer');
const { load, root } = require('../word-native-parity/geometry-reference.cjs');
const { decodeWordOutput } = load(path.join(root, 'src/ml/word-decoder.ts'));
const { convertWordRawPixels, assessWordImageQuality } = load(path.join(root, 'src/ml/word-image-processing.ts'));
const { DEFAULT_WORD_MODEL_THRESHOLDS } = load(path.join(root, 'src/ml/word-model-contract.ts'));
const thresholds = { ...DEFAULT_WORD_MODEL_THRESHOLDS, proposalScore: .35, acceptScore: .4, dotConfidence: .7 };
const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

function encodeIdeal(record) {
  const tensor = new Float32Array(48 * 120 * 11);
  const occupied = new Map(), complete = [], partial = [], collisions = [], neighbors = [];
  for (const [index, box] of record.context_boxes.entries()) {
    const [left, top, right, bottom] = box;
    assert.ok(box.every(Number.isFinite) && right > left && bottom > top);
    if (left < 0 || top < 0 || right > 480 || bottom > 192) {
      partial.push(index);
      continue;
    }
    complete.push(index);
    const centerX = (left + right) / 2 / 4, centerY = (top + bottom) / 2 / 4;
    const column = Math.floor(centerX), row = Math.floor(centerY);
    const offset = (row * 120 + column) * 11;
    assert.ok(column >= 0 && column < 120 && row >= 0 && row < 48);
    if (occupied.has(offset)) {
      collisions.push({ first: occupied.get(offset), second: index, row, column });
      continue; // A stride-4 slot cannot encode two cells. Report, never repair it.
    }
    for (const [otherOffset, otherIndex] of occupied) {
      const otherRow = Math.floor(otherOffset / 11 / 120), otherColumn = otherOffset / 11 % 120;
      if (Math.abs(row - otherRow) <= 1 && Math.abs(column - otherColumn) <= 1) {
        neighbors.push({ first: otherIndex, second: index, row, column });
      }
    }
    occupied.set(offset, index);
    const mask = record.context_masks[index];
    assert.ok(Number.isInteger(mask) && mask >= 0 && mask <= 63);
    tensor.set([1, centerX - column, centerY - row, (right - left) / 480, (bottom - top) / 192,
      ...Array.from({ length: 6 }, (_, dot) => (mask >> dot) & 1)], offset);
  }
  const [x, y, right, bottom] = record.selection;
  const intersects = ([left, top, edgeX, edgeY]) => edgeX / 480 > x && left / 480 < right && edgeY / 192 > y && top / 192 < bottom;
  return { tensor, complete, partial, collisions, neighbors,
    partialIntersectingSelection: partial.filter((index) => intersects(record.context_boxes[index])) };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(values.length / 2);
  return values.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function selectedGeometry(record) {
  const ordered = [...record.boxes].sort((a, b) => (a[0] + a[2]) - (b[0] + b[2]));
  const height = median(ordered.map((box) => box[3] - box[1]));
  const centers = ordered.map((box) => [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]);
  return { max_center_gap_over_median_height: Math.max(0, ...centers.slice(1).map((center, index) => (center[0] - centers[index][0]) / height)),
    row_span_over_median_height: (Math.max(...centers.map((center) => center[1])) - Math.min(...centers.map((center) => center[1]))) / height };
}

function summarize(rows) {
  const reasons = {}, quality = {};
  for (const row of rows) {
    if (row.decoder_reason) reasons[row.decoder_reason] = (reasons[row.decoder_reason] || 0) + 1;
    if (row.quality_reason) quality[row.quality_reason] = (quality[row.quality_reason] || 0) + 1;
  }
  return { total: rows.length, decoder_accepted: rows.filter((row) => row.decoder_text).length,
    decoder_exact: rows.filter((row) => row.decoder_text === row.expected).length,
    decoder_accepted_wrong: rows.filter((row) => row.decoder_text && row.decoder_text !== row.expected).length,
    decoder_rejection_reasons: reasons, quality_rejections: quality,
    pipeline_exact: rows.filter((row) => !row.quality_reason && row.decoder_text === row.expected).length,
    grid_collision_fixtures: rows.filter((row) => row.grid_collisions.length).length,
    grid_collision_cells: rows.reduce((sum, row) => sum + row.grid_collisions.length, 0),
    adjacent_grid_peak_fixtures: rows.filter((row) => row.adjacent_grid_peaks.length).length,
    complete_context_cells: rows.reduce((sum, row) => sum + row.complete_context_cells, 0),
    omitted_partial_context_cells: rows.reduce((sum, row) => sum + row.omitted_partial_context_cells, 0),
    partial_context_intersects_selection_fixtures: rows.filter((row) => row.partial_context_intersects_selection.length).length };
}

function main(directory, destination) {
  if (!directory || !destination) throw new Error('Usage: node geometry-ceiling.cjs <prepared-directory> <report.json>');
  const referencePath = path.join(directory, 'geometry-inputs.json');
  const reference = JSON.parse(fs.readFileSync(referencePath, 'utf8'));
  const inputBuffer = new ArrayBuffer(480 * 192 * 3), inputBytes = new Uint8Array(inputBuffer);
  const inputs = fs.openSync(path.join(directory, 'inputs.rgb'), 'r'), pixelHash = crypto.createHash('sha256');
  const rows = [];
  try {
    for (const record of reference.rows) {
      assert.equal(fs.readSync(inputs, inputBytes), inputBytes.length);
      pixelHash.update(inputBytes);
      const quality = assessWordImageQuality(convertWordRawPixels({ width: 480, height: 192, pixelFormat: 'RGB', buffer: inputBuffer })) || '';
      const ideal = encodeIdeal(record);
      const [x, y, right, bottom] = record.selection;
      const decoded = decodeWordOutput(ideal.tensor, thresholds, { x, y, width: right - x, height: bottom - y });
      rows.push({ group: record.group, index: record.index, image: record.image, source: record.source, expected: record.expected,
        decoder_text: decoded.status === 'success' ? decoded.text : '', decoder_reason: decoded.status === 'success' ? '' : decoded.reason,
        quality_reason: quality, expected_cell_count: record.boxes.length, decoded_box_count: decoded.boxes.length,
        complete_context_cells: ideal.complete.length, omitted_partial_context_cells: ideal.partial.length,
        grid_collisions: ideal.collisions, adjacent_grid_peaks: ideal.neighbors,
        partial_context_intersects_selection: ideal.partialIntersectingSelection, ...selectedGeometry(record) });
    }
    assert.equal(fs.readSync(inputs, Buffer.alloc(1)), 0, 'Unconsumed RGB fixtures');
  } finally { fs.closeSync(inputs); }
  assert.equal(pixelHash.digest('hex'), reference.raw_inputs_sha256);
  const groups = Object.fromEntries(Object.keys(reference.manifests).map((group) => {
    const selected = rows.filter((row) => row.group === group);
    return [group, { ...summarize(selected), by_length: Object.fromEntries(Array.from({ length: 12 }, (_, index) => index + 1)
      .filter((length) => selected.some((row) => row.expected.length === length))
      .map((length) => [length, summarize(selected.filter((row) => row.expected.length === length))])) }];
  }));
  const sources = ['src/ml/word-decoder.ts', 'src/ml/word-model-contract.ts', 'src/ml/word-image-processing.ts', 'src/types/word-scan.ts',
    'scripts/word-native-parity/geometry-reference.cjs', 'scripts/word-decoder-parity/geometry-ceiling.py', 'scripts/word-decoder-parity/geometry-ceiling.cjs'];
  const outcomes = JSON.stringify(rows) + '\n';
  fs.writeFileSync(path.join(directory, 'all-outcomes.json'), outcomes);
  const report = { schema_version: 1, scope: 'Ideal annotated geometry through actual TypeScript decoder and separate RGB quality checks. Geometry/contract feasibility only; NEVER model accuracy or fixture filtering.',
    simulated_annotation_cases: rows.length, native_execution_cases: 0, model_inference_cases: 0,
    thresholds, manifests: reference.manifests, preparation_sha256: sha(fs.readFileSync(referencePath)), raw_inputs_sha256: reference.raw_inputs_sha256,
    all_outcomes_sha256: sha(outcomes),
    source_sha256: Object.fromEntries(sources.map((file) => [file, sha(fs.readFileSync(path.join(root, file)))])),
    encoding: 'Every complete context annotation in JSON double precision; center/4 grid floor, exact objectness1/dots0or1, dx/dy/normalizedw/h rounded only when written to Float32 tensor. Collisions retain first annotation and are explicitly reported. Partial context cells are counted but omitted as requested; complete cells intersecting selection remain included.',
    limitations: ['Natural135 are also contained within stress405; counts across groups are not independent.',
      'Perfect predicted annotations are a diagnostic oracle, not evidence of recognition performance.',
      'A trained model can output boxes different from annotations; these results describe the specified ideal-annotation representation.',
      'Partial cells outside model context are not represented by this oracle; separate clipped-cell regressions remain necessary.'],
    groups, accepted_text_mismatches: rows.filter((row) => row.decoder_text && row.decoder_text !== row.expected),
    decoder_rejected_fixtures: rows.filter((row) => row.decoder_reason),
    grid_collision_fixtures: rows.filter((row) => row.grid_collisions.length),
    adjacent_grid_peak_fixtures: rows.filter((row) => row.adjacent_grid_peaks.length) };
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
  process.stdout.write(JSON.stringify({ groups: Object.fromEntries(Object.entries(groups).map(([name, value]) => [name, { ...value, by_length: undefined }])), accepted_text_mismatches: report.accepted_text_mismatches }, null, 2) + '\n');
}

module.exports = { encodeIdeal, selectedGeometry };
if (require.main === module) main(process.argv[2], process.argv[3]);
