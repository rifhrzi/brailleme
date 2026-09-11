import { assertWordModelOutput, decodeWordOutput, detectWordProposals, wordBoxIou } from '@/ml/word-decoder';
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
    expect(result.text).toBe(word.toLowerCase());
    expect(result.cells).toHaveLength(word.length);
    expect(result.boxes).toEqual(result.cells.map(({ box }) => box));
    expect(result.confidence).toBeCloseTo(0.98);
  });

  it('maps every A-Z mask with dot1 as the least significant bit', () => {
    for (const letter of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') {
      expect(decodeWordOutput(wordOutput(letter), thresholds)).toMatchObject({ status: 'success', text: letter.toLowerCase() });
    }
  });

  it('rejects unsupported patterns and standalone indicators instead of guessing', () => {
    for (let mask = 0; mask < 64; mask += 1) {
      if ((WORD_LATIN_MASKS as readonly number[]).includes(mask) || [50, 2, 22, 18, 6, 36].includes(mask)) continue;
      const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
      addCell(output, 0.5, mask);
      expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: mask === 32 || mask === 60 ? 'invalid_sequence' : mask === 38 || mask === 54 ? 'ambiguous_symbol' : 'unsupported_pattern', cells: [] });
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

  it('rejects no detection and more than twelve emitted characters', () => {
    expect(decodeWordOutput(new Float32Array(WORD_MODEL_OUTPUT_LENGTH), thresholds)).toMatchObject({ reason: 'no_grid' });
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    for (let index = 0; index < 13; index += 1) addCell(output, 0.06 + index * 0.07, 1, { width: 0.04, height: 0.2 });
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: 'too_many_characters' });
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
    expect(result).toMatchObject({ status: 'success', text: 'mama' });
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
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'success', text: 'aa' });
  });

  it('resolves tied overlapping boxes deterministically through positional NMS', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(output, 0.5, 1);
    addCell(output, 0.5 + 1 / 120, 1);
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'success', text: 'a' });
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

  it.each(referenceFixtures.cases)('preserves historical geometric decisions for $name under the new symbol protocol', (fixture) => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    for (const row of fixture.sparse) output.set(row.values, row.offset);
    const result = decodeWordOutput(output, fixture.thresholds, fixture.selection);
    expect(result.status).toBe(fixture.expected.status);
    if (result.status === 'success') {
      // Retained v2 oracle is historical evidence: unmarked text now prints lowercase.
      expect(result.text).toBe(fixture.expected.text.toLowerCase());
      expect(result.cells.map(({ mask }) => mask)).toEqual(fixture.expected.masks);
      result.boxes.forEach((box, index) => {
        const reference = fixture.expected.boxes![index]!;
        for (const key of ['x', 'y', 'width', 'height'] as const) expect(box[key]).toBeCloseTo(reference[key], 6);
      });
    } else {
      expect(result.reason).toBe(fixture.expected.reason === 'too_many_cells' ? 'too_many_characters' : fixture.expected.reason);
      expect(result).not.toHaveProperty('text');
    }
  });

  it.each([
    [[32, 1, 3], 'Ab'], [[32, 32, 1, 3], 'AB'], [[60, 26, 26, 1], '001'],
  ] as const)('decodes indicator sequences %p without losing their boxes', (masks, text) => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    masks.forEach((mask, index) => addCell(output, 0.2 + index * 0.075, mask));
    const result = decodeWordOutput(output, thresholds);
    expect(result).toMatchObject({ status: 'success', text });
    expect(result.cells).toHaveLength(masks.length);
    expect(result.boxes).toHaveLength(masks.length);
    if (result.status === 'success') expect(result.cells.map(({ output }) => output).join('')).toBe(text);
  });

  it('includes weak indicator evidence in confidence/rejection while exposing proposals for tracking', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(output, 0.2, 32, { score: 0.4 });
    addCell(output, 0.275, 1);
    addCell(output, 0.35, 63);
    const proposals = detectWordProposals(output, thresholds);
    expect(proposals.map(({ mask }) => mask)).toEqual([32, 1, 63]);
    expect(proposals[0]?.confidence).toBeCloseTo(0.4);
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: 'low_confidence' });
    addCell(output, 0.2, 32, { score: 0.8 });
    addCell(output, 0.35, 3);
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'success', text: 'Ab' });
    const accepted = decodeWordOutput(output, thresholds);
    if (accepted.status === 'success') expect(accepted.confidence).toBeCloseTo(0.8);
  });

  it('accepts twelve capital letters in 24 cells and rejects physical-cell overflow', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    for (let index = 0; index < 24; index += 1) addCell(output, 0.045 + index * 0.037, index % 2 ? 1 : 32, { width: 0.022 });
    const result = decodeWordOutput(output, thresholds);
    expect(result).toMatchObject({ status: 'success', text: 'A'.repeat(12) });
    expect(result.boxes).toHaveLength(24);
    addCell(output, 0.045 + 24 * 0.037, 1, { width: 0.022 });
    expect(decodeWordOutput(output, thresholds)).toMatchObject({ status: 'rejected', reason: 'too_many_cells' });
  });
});
