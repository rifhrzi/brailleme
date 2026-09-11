import { WORD_DECODER_VERSION, WORD_SYMBOL_PROTOCOL } from '@/ml/word-model-contract';
import { parseWordMasks } from '@/ml/word-symbols';

import fixtures from '../../../training/fixtures/word-symbols-id-v1.json';
import punctuationFixtures from './fixtures/word-symbols-id-v4-reference.json';

describe('Indonesian Braille symbols shared with Python', () => {
  it('declares the deployed semantic contract', () => {
    expect(punctuationFixtures.protocol).toBe(WORD_SYMBOL_PROTOCOL);
    expect(punctuationFixtures.decoderVersion).toBe(WORD_DECODER_VERSION);
  });

  it.each(punctuationFixtures.cases)('$name', (fixture) => {
    expect(parseWordMasks(fixture.masks)).toEqual(fixture.result);
  });

  it.each(fixtures.alphabet)('maps lowercase and capital %s', (letter, mask) => {
    expect(parseWordMasks([Number(mask)])).toMatchObject({ status: 'success', text: String(letter) });
    expect(parseWordMasks([32, Number(mask)])).toMatchObject({ status: 'success', text: String(letter).toUpperCase() });
  });

  it('separates the twelve-character limit from physical indicator cells', () => {
    expect(parseWordMasks([32, 32, ...Array<number>(12).fill(1)])).toMatchObject({ status: 'success', text: 'A'.repeat(12) });
    expect(parseWordMasks(Array.from({ length: 12 }, () => [32, 1]).flat())).toMatchObject({ status: 'success', text: 'A'.repeat(12) });
    expect(parseWordMasks([60, ...Array<number>(12).fill(26)])).toMatchObject({ status: 'success', text: '0'.repeat(12) });
    expect(parseWordMasks(Array<number>(13).fill(1))).toEqual({ status: 'rejected', reason: 'too_many_characters' });
    expect(parseWordMasks(Array<number>(25).fill(1))).toEqual({ status: 'rejected', reason: 'too_many_cells' });
  });

  it.each([-1, 64, 1.2, Number.NaN, Number.POSITIVE_INFINITY])('rejects invalid mask %s', (mask) => {
    expect(parseWordMasks([mask])).toEqual({ status: 'rejected', reason: 'unsupported_pattern' });
  });
});
