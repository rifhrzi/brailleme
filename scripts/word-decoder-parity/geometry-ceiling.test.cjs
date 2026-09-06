const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');
const { encodeIdeal } = require('./geometry-ceiling.cjs');
const { load, root } = require('../word-native-parity/geometry-reference.cjs');
const { decodeWordOutput } = load(path.join(root, 'src/ml/word-decoder.ts'));
const { DEFAULT_WORD_MODEL_THRESHOLDS } = load(path.join(root, 'src/ml/word-model-contract.ts'));

test('ideal Float32 annotations preserve repeated letters and spatial order', () => {
  const ideal = encodeIdeal({ context_boxes: [[200, 65, 225, 115], [110, 65, 135, 115], [170, 65, 195, 115], [140, 65, 165, 115]],
    context_masks: [1, 13, 13, 1], selection: [0, 0, 1, 1] });
  assert.ok(ideal.tensor instanceof Float32Array);
  const result = decodeWordOutput(ideal.tensor, DEFAULT_WORD_MODEL_THRESHOLDS);
  assert.equal(result.text, 'MAMA');
  assert.deepEqual(result.cells.map((cell) => cell.mask), [13, 1, 13, 1]);
  assert.deepEqual(ideal.collisions, []);
});

test('complete cells intersecting selection remain encoded and reject clipping', () => {
  const ideal = encodeIdeal({ context_boxes: [[100, 50, 125, 100], [150, 20, 180, 70], [-8, 30, 15, 100]],
    context_masks: [1, 3, 9], selection: [.15, .2, .8, .7] });
  assert.deepEqual(ideal.complete, [0, 1]);
  assert.deepEqual(ideal.partial, [2]);
  const result = decodeWordOutput(ideal.tensor, DEFAULT_WORD_MODEL_THRESHOLDS, { x: .15, y: .2, width: .65, height: .5 });
  assert.equal(result.reason, 'clipped');
});

test('same-grid collisions are explicitly reported without overwriting the first cell', () => {
  const ideal = encodeIdeal({ context_boxes: [[100, 50, 120, 90], [101, 50, 121, 90]],
    context_masks: [1, 3], selection: [0, 0, 1, 1] });
  assert.equal(ideal.complete.length, 2);
  assert.equal(ideal.collisions.length, 1);
  const offset = (ideal.collisions[0].row * 120 + ideal.collisions[0].column) * 11;
  assert.equal(ideal.tensor[offset + 5], 1);
  assert.equal(ideal.tensor[offset + 6], 0);
});
