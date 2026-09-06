import { assertWordModelOutput, decodeWordOutput, wordBoxIou } from '@/ml/word-decoder';
import { DEFAULT_WORD_MODEL_THRESHOLDS, isValidWordModelThresholds, WORD_LATIN_MASKS, WORD_MODEL_OUTPUT_LENGTH } from '@/ml/word-model-contract';

import referenceFixtures from './fixtures/word-decoder-reference.json';

const thresholds = DEFAULT_WORD_MODEL_THRESHOLDS;

function addCell(output: Float32Array, x: number, mask: number, options: { y?: number; width?: number; height?: number; score?: number } = {}) {
  const y = options.y ?? 0.5;
  const column = Math.floor(x * 120);
  const row = Math.floor(y * 48);
  const offset = (row * 120 + column) * 11;
  output.set([options.score ?? 0.98, x * 120 - column, y * 48 - row, options.width ?? 0.05, options.height ?? 0.25], offset);
  for (let dot = 0; dot < 6; dot += 1) output[offset + 5 + dot] = mask & (1 << dot) ? 0.99 : 0.01;
  return offset;
}

function wordOutput(word: string) {
  const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
  // Insert out of order to ensure ordering comes from coordinates, not producer order.
  for (let index = word.length - 1; index >= 0; index -= 1) {
    addCell(output, 0.08 + index * 0.075, WORD_LATIN_MASKS[word.charCodeAt(index) - 65]!);
  }
  return output;
}

describe('six-dot word decoder', () => {
  it.each(['A', 'MAMA', 'KAKAK', 'ABCDEFGHIJKL'])('recognizes and orders %s without collapsing repeated letters', (word) => {
    const result = decodeWordOutput(wordOutput(word), thresholds);
    expect(result.status).toBe('success');
    if (result.status !== 'success') throw new Error(result.reason);
    expect(result.text).toBe(word);
    expect(result.cells).toHaveLength(word.length);
    expect(result.boxes).toEqual(result.cells.map(({ box }) => box));
    expect(result.confidence).toBeCloseTo(0.98);
  });

  it('maps every A-Z mask with dot1 as the least significant bit', () => {
    for (const letter of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') {
      expect(decodeWordOutput(wordOutput(letter), thresholds)).toMatchObject({ status: 'success', text: letter });
    }
  });

  it('rejects all 38 non-letter patterns instead of assigning a nearest letter', () => {
    for (let mask = 0; mask < 64; mask += 1) {
      if ((WORD_LATIN_MASKS as readonly number[]).includes(mask)) continue;
      const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
      addCell(output, 0.5, mask);
      expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: 'unsupported_pattern', cells: [] });
    }
  });

  it('rejects a weak middle cell rather than returning a shortened word', () => {
    const output = wordOutput('MAMA');
    addCell(output, 0.08 + 0.075, 1, { score: 0.4 });
    const result = decodeWordOutput(output, thresholds);
    expect(result).toMatchObject({ status: 'rejected', reason: 'low_confidence', cells: [] });
    expect(result.boxes).toHaveLength(4);
    expect(result).not.toHaveProperty('text');
  });

  it('rejects any uncertain dot, including an uncertain absent dot', () => {
    const output = wordOutput('A');
    const offset = addCell(output, 0.08, 1);
    output[offset + 10] = 0.2;
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: 'low_confidence' });
  });

  it('rejects no detection and more than twelve detections', () => {
    expect(decodeWordOutput(new Float32Array(WORD_MODEL_OUTPUT_LENGTH), thresholds)).toMatchObject({ reason: 'no_grid' });
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    for (let index = 0; index < 13; index += 1) addCell(output, 0.06 + index * 0.07, 1, { width: 0.04, height: 0.2 });
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: 'too_many_cells' });
  });

  it.each([
    ['too_far', 0.5, { width: 0.004 }],
    ['too_far', 0.5, { height: 0.01 }],
    ['clipped', 0.01, { width: 0.08 }],
    ['clipped', 0.99, { width: 0.08 }],
    ['clipped', 0.5, { y: 0.02 }],
    ['clipped', 0.5, { y: 0.98 }],
  ] as const)('rejects %s geometry', (reason, x, options) => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(output, x, 1, options);
    const result = decodeWordOutput(output, thresholds);
    expect(result).toMatchObject({ status: 'rejected', reason });
    expect(result.boxes.every((box) => box.x >= 0 && box.y >= 0 && box.x + box.width <= 1 && box.y + box.height <= 1)).toBe(true);
  });

  it('rejects multiple rows and a suspicious gap', () => {
    const rows = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(rows, 0.3, 1, { y: 0.3 });
    addCell(rows, 0.4, 1, { y: 0.7 });
    expect(decodeWordOutput(rows, thresholds)).toMatchObject({ reason: 'multiple_rows' });
    const gap = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(gap, 0.2, 1);
    addCell(gap, 0.8, 1);
    expect(decodeWordOutput(gap, thresholds)).toMatchObject({ reason: 'gap' });
  });

  it('rejects a missing interior cell even when the resulting gap is below 1.8 cell heights', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(output, 0.2, 13);
    // Expected next cell at 0.275 was missed; do not return the shortened MA.
    addCell(output, 0.35, 1);
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: 'gap', cells: [] });
  });

  it('selects a narrow word row from model context without importing neighboring rows', () => {
    const output = wordOutput('MAMA');
    addCell(output, 0.08, 63, { y: 0.12, height: 0.12 });
    addCell(output, 0.2, 0, { y: 0.88, height: 0.12, score: 0.3 });
    const selection = { x: 0, y: 0.34375, width: 1, height: 0.3125 };
    const result = decodeWordOutput(output, thresholds, selection);
    expect(result).toMatchObject({ status: 'success', text: 'MAMA' });
    expect(result.boxes).toHaveLength(4);
  });

  it('rejects a cell intersecting the visible guide boundary instead of silently removing it', () => {
    const output = wordOutput('MAMA');
    addCell(output, 0.5, 1, { y: 0.3, height: 0.12 });
    expect(decodeWordOutput(output, thresholds, { x: 0, y: 0.34375, width: 1, height: 0.3125 })).toMatchObject({ status: 'rejected', reason: 'clipped', cells: [] });
  });

  it('keeps a weak cell inside the visible guide and validates the selection bounds', () => {
    const output = wordOutput('MAMA');
    addCell(output, 0.155, 1, { score: 0.4 });
    expect(decodeWordOutput(output, thresholds, { x: 0, y: 0.34375, width: 1, height: 0.3125 })).toMatchObject({ status: 'rejected', reason: 'low_confidence' });
    expect(() => decodeWordOutput(output, thresholds, { x: 0, y: 0, width: 0, height: 1 })).toThrow('pilihan');
  });

  it('suppresses duplicate boxes by position without deleting a repeated letter elsewhere', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(output, 0.3, 1, { width: 0.08 });
    addCell(output, 0.32, 1, { width: 0.08, score: 0.7 });
    addCell(output, 0.4, 1, { width: 0.08 });
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'success', text: 'AA' });
  });

  it('resolves tied overlapping boxes deterministically through positional NMS', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(output, 0.5, 1);
    addCell(output, 0.5 + 1 / 120, 1);
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'success', text: 'A' });
  });

  it('rejects overlapping proposals that survive NMS instead of guessing their sequence', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(output, 0.3, 1, { width: 0.1 });
    addCell(output, 0.34, 1, { width: 0.1 });
    expect(decodeWordOutput(output, { ...thresholds, nmsIou: 0.9 })).toMatchObject({ reason: 'overlapping_cells' });
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -0.1, 1.01])('rejects malformed probability %s anywhere in the tensor', (value) => {
    const output = wordOutput('A');
    output[output.length - 1] = value;
    expect(() => decodeWordOutput(output, thresholds)).toThrow('probabilitas');
  });

  it('rejects invalid tensor length and invalid thresholds', () => {
    expect(() => assertWordModelOutput(new Float32Array(26))).toThrow('Dimensi');
    expect(() => decodeWordOutput(wordOutput('A'), { ...thresholds, proposalScore: 0.8 })).toThrow('Ambang');
    expect(wordBoxIou({ x: 0, y: 0, width: 0, height: 0 }, { x: 0, y: 0, width: 0, height: 0 })).toBe(0);
  });

  it.each([undefined, null, [], {}, { ...thresholds, dotConfidence: 0.5 }, { ...thresholds, nmsIou: 1 }, { ...thresholds, maxGapRatio: Number.NaN }])('rejects unsafe manifest thresholds %p', (value) => {
    expect(isValidWordModelThresholds(value)).toBe(false);
  });

  it.each(referenceFixtures.cases)('matches the independent Python reference for $name', (fixture) => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    for (const row of fixture.sparse) output.set(row.values, row.offset);
    const result = decodeWordOutput(output, fixture.thresholds, fixture.selection);
    expect(result.status).toBe(fixture.expected.status);
    if (result.status === 'success') {
      expect(result.text).toBe(fixture.expected.text);
      expect(result.cells.map(({ mask }) => mask)).toEqual(fixture.expected.masks);
      result.boxes.forEach((box, index) => {
        const reference = fixture.expected.boxes![index]!;
        for (const key of ['x', 'y', 'width', 'height'] as const) expect(box[key]).toBeCloseTo(reference[key], 6);
      });
    } else {
      expect(result.reason).toBe(fixture.expected.reason);
      expect(result).not.toHaveProperty('text');
    }
  });
});
