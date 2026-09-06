import { decodeWordOutput } from '@/ml/word-decoder';
import { DEFAULT_WORD_MODEL_THRESHOLDS, WORD_MODEL_OUTPUT_LENGTH } from '@/ml/word-model-contract';

const thresholds = { ...DEFAULT_WORD_MODEL_THRESHOLDS, proposalScore: 0.25, acceptScore: 0.25, dotConfidence: 0.7 };

function output(cells: readonly (readonly [number, number, number, number])[]): Float32Array {
  const values = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
  for (const [centerX, width, mask, score] of cells) {
    const column = Math.floor(centerX / 4);
    values.set([score, centerX / 4 - column, 0, width / 480, 12 / 192,
      ...Array.from({ length: 6 }, (_, bit) => (mask >> bit) & 1)], (24 * 120 + column) * 11);
  }
  return values;
}

describe('word decoder positional suppression', () => {
  it.each([[1, 'AA'], [3, 'AB']] as const)('keeps adjacent cells with mask %s', (secondMask, text) => {
    for (const [firstScore, secondScore] of [[1, 1], [0.99, 0.98], [0.98, 0.99]]) {
      expect(decodeWordOutput(output([[100, 4, 1, firstScore!], [104.5, 4, secondMask, secondScore!]]), thresholds))
        .toMatchObject({ status: 'success', text });
    }
  });

  it('removes tied overlapping duplicates and retains nonoverlapping ends of an NMS chain', () => {
    expect(decodeWordOutput(output([[100, 12, 1, 0.99], [104, 12, 1, 0.99]]), thresholds))
      .toMatchObject({ status: 'success', text: 'A' });
    expect(decodeWordOutput(output([[100, 7, 1, 0.99], [104, 7, 3, 0.98], [108, 7, 9, 0.97]]), thresholds))
      .toMatchObject({ status: 'success', text: 'AC' });
  });

  it('rejects the whole word for a weak, uncertain or unsupported adjacent cell', () => {
    const values = output([[100, 4, 1, 0.99], [104.5, 4, 3, 0.3]]);
    expect(decodeWordOutput(values, { ...thresholds, acceptScore: 0.4 }))
      .toMatchObject({ status: 'rejected', reason: 'low_confidence', cells: [] });
    const second = (24 * 120 + 26) * 11;
    values[second] = 0.98;
    values[second + 6] = 0.5;
    expect(decodeWordOutput(values, thresholds)).toMatchObject({ status: 'rejected', reason: 'low_confidence', cells: [] });
    values.fill(0, second + 5, second + 11);
    expect(decodeWordOutput(values, thresholds)).toMatchObject({ status: 'rejected', reason: 'unsupported_pattern', cells: [] });
  });

  it('ignores an outside-guide neighbor while rejecting a box that crosses the guide', () => {
    const values = output([[100, 4, 1, 0.9], [104.5, 4, 3, 0.99]]);
    expect(decodeWordOutput(values, thresholds, { x: 0.2, y: 0.45, width: 0.013, height: 0.1 }))
      .toMatchObject({ status: 'success', text: 'A' });
    expect(decodeWordOutput(values, thresholds, { x: 0.2, y: 0.45, width: 0.02, height: 0.1 }))
      .toMatchObject({ status: 'rejected', reason: 'clipped', cells: [] });
  });
});
