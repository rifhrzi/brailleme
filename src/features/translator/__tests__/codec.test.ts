import {
  type BrailleItem,
  decodeCells,
  encodeText,
  MAX_BRAILLE_CELLS,
  MAX_TEXT_CHARACTERS,
} from '../codec';

// Literal fixtures keep mapping checks independent of the shared scanner table.
const alphabet = 'abcdefghijklmnopqrstuvwxyz';
const latinMasks = [1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13, 29, 21, 15, 31, 23, 14, 30, 37, 39, 58, 45, 61, 53];
const cells = (...masks: number[]): BrailleItem[] => masks.map((mask) => ({ kind: 'cell', mask }));
const space: BrailleItem = { kind: 'space' };
const newline: BrailleItem = { kind: 'newline' };

function expectRoundTrip(text: string): void {
  const encoded = encodeText(text);
  expect(encoded.issues).toEqual([]);
  expect(encoded.valid).toBe(true);
  expect(decodeCells(encoded.value)).toEqual({ value: text, valid: true, issues: [] });
}

describe('translator letter and six-dot mapping', () => {
  test.each(Array.from(alphabet, (letter, index) => [letter, latinMasks[index]!] as const))('%s uses mask %i in both directions', (letter, mask) => {
    expect(encodeText(letter)).toEqual({ value: cells(mask), valid: true, issues: [] });
    expect(decodeCells(cells(mask))).toEqual({ value: letter, valid: true, issues: [] });
    expect(encodeText(letter.toUpperCase())).toEqual({ value: cells(32, mask), valid: true, issues: [] });
    expect(decodeCells(cells(32, mask))).toEqual({ value: letter.toUpperCase(), valid: true, issues: [] });
  });

  test.each([
    [1, 'a', undefined],
    [2, ',', undefined],
    [4, '\uFFFD', 'unsupported_mask'],
    [8, '\uFFFD', 'unsupported_mask'],
    [16, '\uFFFD', 'unsupported_mask'],
    [32, '\uFFFD', 'incomplete_indicator'],
  ] as const)('physical dot mask %i is interpreted without reversing the two columns', (mask, value, code) => {
    const result = decodeCells(cells(mask));
    expect(result.value).toBe(value);
    expect(result.valid).toBe(code === undefined);
    if (code) expect(result.issues).toEqual([expect.objectContaining({ index: 0, code })]);
  });

  test('capital encoding stays per letter; double-capital input applies to one whole word', () => {
    expect(encodeText('ABC').value).toEqual(cells(32, 1, 32, 3, 32, 9));
    expect(decodeCells(cells(32, 32, 1, 3, 9))).toEqual({ value: 'ABC', valid: true, issues: [] });
    expect(decodeCells([...cells(32, 32, 1, 3, 9), space, ...cells(25)])).toEqual({ value: 'ABC d', valid: true, issues: [] });
    expect(decodeCells(cells(32, 32, 1, 2, 3))).toEqual({ value: 'A,b', valid: true, issues: [] });
    expectRoundTrip('iPhone eBay ABC zZz');
  });

  test('every two-letter mixed-case pair round-trips without scanner word limits', () => {
    for (const left of alphabet) {
      for (const right of alphabet) expectRoundTrip(left.toUpperCase() + right);
    }
    expectRoundTrip('a'.repeat(150));
  });
});

describe('unsigned integer mode', () => {
  test('one number indicator covers every digit and preserves leading zeroes', () => {
    expect(encodeText('0123456789').value).toEqual(cells(60, 26, 1, 3, 9, 25, 17, 11, 27, 19, 10));
    expectRoundTrip('000 0123456789 9876543210');
    expect(decodeCells(cells(26, 26, 1)).value).toBe('jja');
    for (let number = 0; number < 100; number += 1) expectRoundTrip(String(number).padStart(2, '0'));
  });

  test.each(['12:b', 'a:12', '12, 34', '12. 34', '12;34', '(12)', '12\nb', 'a,5'])('punctuation or whitespace ends number mode in %j', (text) => {
    expectRoundTrip(text);
  });

  test('each integer after punctuation receives its own indicator', () => {
    expect(encodeText('1:2').value).toEqual(cells(60, 1, 18, 60, 3));
    expect(decodeCells(cells(60, 1, 18, 3))).toEqual({ value: '1:b', valid: true, issues: [] });
  });

  test.each([
    ['12,5', 'decimal_number', 2],
    ['12.5', 'decimal_number', 2],
    ['.5', 'decimal_number', 0],
    [',5', 'decimal_number', 0],
    ['a .5', 'decimal_number', 2],
    ['-10', 'negative_number', 0],
    ['a1', 'mixed_alphanumeric', 1],
    ['1a', 'mixed_alphanumeric', 1],
  ] as const)('%j is invalid without dropping input', (text, code, index) => {
    const result = encodeText(text);
    expect(result.valid).toBe(false);
    expect(result.issues).toContainEqual(expect.objectContaining({ code, index }));
    expect(result.value.length).toBeGreaterThanOrEqual(text.length);
  });

  test('decimal issues from Braille point to the original punctuation item, after indicators', () => {
    const result = decodeCells(cells(60, 1, 3, 2, 60, 17));
    expect(result.value).toBe('12,5');
    expect(result.valid).toBe(false);
    expect(result.issues).toEqual([expect.objectContaining({ code: 'decimal_number', index: 3 })]);
  });

  test.each([
    [cells(1, 60, 3), 'a2', 'mixed_alphanumeric', 1],
    [cells(60, 1, 5), '1k', 'mixed_alphanumeric', 2],
    [cells(60, 1, 32, 3), '1B', 'mixed_alphanumeric', 2],
    [cells(60, 1, 60, 3), '12', 'invalid_indicator', 2],
  ] as const)('invalid manual number transitions remain explicit: %j', (items, text, code, index) => {
    const result = decodeCells(items);
    expect(result.value).toBe(text);
    expect(result.valid).toBe(false);
    expect(result.issues).toContainEqual(expect.objectContaining({ code, index }));
  });
});

describe('punctuation and document boundaries', () => {
  test.each([
    ['.', 50], [',', 2], ['?', 38], ['!', 22], [':', 18], [';', 6], ['-', 36],
  ] as const)('standalone %s uses literal mask %i', (text, mask) => {
    expect(encodeText(text)).toEqual({ value: cells(mask), issues: [], valid: true });
    expect(decodeCells(cells(mask))).toEqual({ value: text, issues: [], valid: true });
  });

  test('parentheses share a cell and alternate across words and lines', () => {
    expect(encodeText('()').value).toEqual(cells(54, 54));
    expect(decodeCells(cells(54, 1, 54, 54, 3, 54))).toEqual({ value: '(a)(b)', valid: true, issues: [] });
    expectRoundTrip('(Aku punya 12 buku)');
    expectRoundTrip('(Aku\npunya 12 buku) (BACA!)');
    expectRoundTrip('.,?!:;-()');
  });

  test.each([
    ['(a', 'unmatched_parenthesis', 0],
    ['a)', 'unmatched_parenthesis', 1],
    ['((a))', 'nested_parentheses', 1],
    [')(a', 'unmatched_parenthesis', 0],
  ] as const)('invalid typed parentheses %j retain all cells and indexed issue', (text, code, index) => {
    const result = encodeText(text);
    expect(result.value).toHaveLength(text.length);
    expect(result.valid).toBe(false);
    expect(result.issues).toContainEqual(expect.objectContaining({ code, index }));
  });

  test('odd manual parenthesis count is invalid at its original item', () => {
    const result = decodeCells([...cells(32, 1), space, ...cells(54, 3)]);
    expect(result.value).toBe('A (b');
    expect(result.issues).toEqual([expect.objectContaining({ code: 'unmatched_parenthesis', index: 3 })]);
  });

  test.each(['', ' ', '\n', '  \n\n ', 'Aku  007\nBACA!', '  Aku  007\n\nBACA!  '])('preserves every space and newline in %j', (text) => expectRoundTrip(text));
});

describe('incomplete or unsupported input', () => {
  test('zero is an unfinished cell, never an implicit space', () => {
    const result = decodeCells([...cells(1, 0), space, ...cells(3)]);
    expect(result.value).toBe('a\uFFFD b');
    expect(result.valid).toBe(false);
    expect(result.issues).toEqual([expect.objectContaining({ index: 1, code: 'incomplete_cell' })]);
  });

  test.each([-1, 64, 1.5, NaN, Infinity])('invalid six-dot mask %s has a visible replacement and source index', (mask) => {
    const result = decodeCells([...cells(1), newline, ...cells(mask, 3)]);
    expect(result.value).toBe('a\n\uFFFDb');
    expect(result.issues).toEqual([expect.objectContaining({ index: 2, code: 'invalid_mask' })]);
  });

  test('unsupported Unicode codepoints use one placeholder each with codepoint indices', () => {
    const result = encodeText('a😀é\t\rZ');
    expect(result.value).toEqual(cells(1, 0, 0, 0, 0, 32, 53));
    expect(result.valid).toBe(false);
    expect(result.issues.map(({ index, code }) => ({ index, code }))).toEqual([1, 2, 3, 4].map((index) => ({ index, code: 'unsupported_character' })));
    expect(decodeCells(result.value).value).toBe('a\uFFFD\uFFFD\uFFFD\uFFFDZ');
  });

  test.each([32, 60])('pending indicator %i remains invalid at EOF or any non-letter boundary', (indicator) => {
    for (const suffix of [[], [space], [newline], cells(50), cells(0), cells(63)]) {
      const result = decodeCells([...cells(1), space, ...cells(indicator), ...suffix]);
      expect(result.valid).toBe(false);
      expect(result.value.startsWith('a \uFFFD')).toBe(true);
      expect(result.issues).toContainEqual(expect.objectContaining({ index: 2, code: 'incomplete_indicator' }));
    }
  });

  test('a number indicator before k is incomplete and cannot silently disappear', () => {
    const result = decodeCells(cells(60, 5));
    expect(result.value).toBe('\uFFFDk');
    expect(result.issues).toContainEqual(expect.objectContaining({ index: 0, code: 'incomplete_indicator' }));
    expect(result.issues).toContainEqual(expect.objectContaining({ index: 1, code: 'mixed_alphanumeric' }));
  });

  test.each([
    [cells(32, 32), 'incomplete_indicator', 0],
    [cells(32, 32, 32, 1), 'invalid_indicator', 0],
    [cells(1, 32, 32, 3), 'invalid_indicator', 1],
    [cells(32, 32, 1, 32, 3), 'invalid_indicator', 3],
    [cells(60, 60, 1), 'invalid_indicator', 1],
    [cells(32, 60, 1), 'incomplete_indicator', 0],
  ] as const)('malformed indicator sequence %j stays invalid', (items, code, index) => {
    const result = decodeCells(items);
    expect(result.valid).toBe(false);
    expect(result.issues).toContainEqual(expect.objectContaining({ code, index }));
  });

  test('all 64 physical patterns are either recognized or explicitly replaced', () => {
    const supported = new Set([...latinMasks, 32, 60, 50, 2, 38, 22, 18, 6, 36, 54]);
    for (let mask = 0; mask < 64; mask += 1) {
      const result = decodeCells(cells(mask));
      if (!supported.has(mask)) {
        expect(result.value).toBe('\uFFFD');
        expect(result.valid).toBe(false);
        expect(result.issues).toEqual([expect.objectContaining({ index: 0, code: mask === 0 ? 'incomplete_cell' : 'unsupported_mask' })]);
      }
    }
  });

  test('every valid two-cell pattern is safely re-encodable; issues are unique and in input order', () => {
    for (let first = 0; first < 64; first += 1) {
      for (let second = 0; second < 64; second += 1) {
        const result = decodeCells(cells(first, second));
        const keys = result.issues.map(({ index, code }) => `${index}:${code}`);
        expect(new Set(keys).size).toBe(keys.length);
        expect(result.issues.map(({ index }) => index)).toEqual(result.issues.map(({ index }) => index).sort((left, right) => left - right));
        if (result.valid) expectRoundTrip(result.value);
      }
    }
  });
});

describe('limits and raw-input preservation', () => {
  test('200 uppercase characters fit exactly 400 items', () => {
    expect(MAX_TEXT_CHARACTERS).toBe(200);
    expect(MAX_BRAILLE_CELLS).toBe(400);
    const text = 'A'.repeat(200);
    expect(encodeText(text).value).toHaveLength(400);
    expectRoundTrip(text);
    const oversize = encodeText(text + 'A');
    expect(oversize.value).toHaveLength(402);
    expect(oversize.issues.map(({ index, code }) => ({ index, code }))).toEqual([
      { index: 200, code: 'text_limit' }, { index: 200, code: 'braille_limit' },
    ]);
  });

  test('decoding preserves excess items and reports the first excess item, including newline', () => {
    const items = encodeText('A'.repeat(200)).value;
    const result = decodeCells([...items, newline]);
    expect(result.value).toBe('A'.repeat(200) + '\n');
    expect(result.issues).toEqual([
      expect.objectContaining({ index: 400, code: 'braille_limit' }),
      expect.objectContaining({ index: 400, code: 'text_limit' }),
    ]);
    const spaces = decodeCells(Array.from({ length: 401 }, () => space));
    expect(spaces.value).toBe(' '.repeat(401));
    expect(spaces.issues).toContainEqual(expect.objectContaining({ index: 400, code: 'braille_limit' }));
  });

  test('text length and Braille item length are independent limits', () => {
    const result = decodeCells(cells(...Array<number>(400).fill(1)));
    expect(result.value).toBe('a'.repeat(400));
    expect(result.issues).toEqual([expect.objectContaining({ index: 200, code: 'text_limit' })]);
  });

  test('text limits count Unicode codepoints, not UTF-16 code units', () => {
    const exactly200 = encodeText('a'.repeat(199) + '😀');
    expect(exactly200.value).toHaveLength(200);
    expect(exactly200.issues).toEqual([expect.objectContaining({ index: 199, code: 'unsupported_character' })]);
    const over200 = encodeText('a'.repeat(200) + '😀');
    expect(over200.value).toHaveLength(201);
    expect(over200.issues).toContainEqual(expect.objectContaining({ index: 200, code: 'text_limit' }));
  });

  test('large unsupported paste keeps every codepoint without quadratic issue deduplication', () => {
    const result = encodeText('😀'.repeat(20_000));
    expect(result.value).toHaveLength(20_000);
    expect(result.value.every((item) => item.kind === 'cell' && item.mask === 0)).toBe(true);
    expect(result.issues.filter(({ code }) => code === 'unsupported_character')).toHaveLength(20_000);
    expect(result.issues).toContainEqual(expect.objectContaining({ index: 200, code: 'text_limit' }));
    expect(result.issues).toContainEqual(expect.objectContaining({ index: 400, code: 'braille_limit' }));
  });

  test('decoding does not mutate editor documents or their items', () => {
    const items = Object.freeze([...cells(32, 1), space, ...cells(60, 26), newline].map((item) => Object.freeze(item)));
    const before = JSON.stringify(items);
    expect(decodeCells(items)).toEqual({ value: 'A 0\n', issues: [], valid: true });
    expect(JSON.stringify(items)).toBe(before);
  });
});
