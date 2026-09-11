import { WORD_LATIN_MASKS } from '@/ml/word-model-contract';
import { WORD_CAPITAL_MASK, WORD_NUMBER_MASK, WORD_PUNCTUATION_MASKS } from '@/ml/word-symbols';

export type BrailleItem = { kind: 'cell'; mask: number } | { kind: 'space' } | { kind: 'newline' };
export type TranslationIssue = { index: number; code: string; message: string };
export type TranslationResult<T> = { value: T; issues: TranslationIssue[]; valid: boolean };
export const MAX_TEXT_CHARACTERS = 200;
/** Document items, including spaces and newlines, count toward the editor limit. */
export const MAX_BRAILLE_CELLS = 400;

const REPLACEMENT = '\uFFFD';
const punctuation = Object.entries(WORD_PUNCTUATION_MASKS);

function addIssue(issues: TranslationIssue[], index: number, code: string, message: string): void {
  issues.push({ index, code, message });
}

function finish<T>(value: T, issues: TranslationIssue[]): TranslationResult<T> {
  const seen = new Set<string>();
  const uniqueIssues = issues.filter(({ index, code }) => {
    const key = `${index}:${code}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  uniqueIssues.sort((left, right) => left.index - right.index);
  return { value, issues: uniqueIssues, valid: uniqueIssues.length === 0 };
}

function isLetter(char: string | undefined): boolean {
  return char !== undefined && /^[A-Za-z]$/.test(char);
}

function isDigit(char: string | undefined): boolean {
  return char !== undefined && /^[0-9]$/.test(char);
}

/** Positions refer to the ORIGINAL input: codepoints when encoding, items when decoding. */
function checkTextRules(chars: readonly string[], origins: readonly number[], issues: TranslationIssue[]): void {
  const parentheses: number[] = [];
  if (chars.length > MAX_TEXT_CHARACTERS) addIssue(issues, origins[MAX_TEXT_CHARACTERS]!, 'text_limit', `Teks maksimal ${MAX_TEXT_CHARACTERS} karakter. Masukan tetap dipertahankan.`);
  for (let index = 0; index < chars.length; index += 1) {
    const char = chars[index]!;
    const previous = chars[index - 1], next = chars[index + 1], origin = origins[index]!;
    if ((isLetter(previous) && isDigit(char)) || (isDigit(previous) && isLetter(char))) {
      addIssue(issues, origin, 'mixed_alphanumeric', 'Pisahkan huruf dan angka dengan spasi, baris baru, atau tanda baca.');
    }
    if (char === '-' && isDigit(next)) addIssue(issues, origin, 'negative_number', 'Bilangan negatif belum didukung. Gunakan bilangan bulat tanpa tanda.');
    if ((char === '.' || char === ',') && isDigit(next) && !isLetter(previous)) {
      addIssue(issues, origin, 'decimal_number', 'Angka desimal atau pemisah ribuan belum didukung. Pisahkan bilangan dengan spasi.');
    }
    if (char === '(') {
      if (parentheses.length) addIssue(issues, origin, 'nested_parentheses', 'Tanda kurung bertingkat belum didukung.');
      parentheses.push(origin);
    } else if (char === ')') {
      if (parentheses.length) parentheses.pop();
      else addIssue(issues, origin, 'unmatched_parenthesis', 'Tanda kurung tutup belum memiliki pasangan pembuka.');
    }
  }
  for (const index of parentheses) addIssue(issues, index, 'unmatched_parenthesis', 'Tanda kurung buka belum memiliki pasangan penutup.');
}

/**
 * Deterministic digital transcription, independent of scanner token/geometry limits.
 * Every uppercase letter gets its own capital indicator. Each integer gets one
 * number indicator; punctuation, spaces and newlines end number mode. Unsupported
 * codepoints become editable mask-0 placeholders rather than disappearing.
 * LF and ASCII spaces are preserved exactly; other whitespace is unsupported.
 * Invalid/oversize input is retained in the returned value, never truncated.
 */
export function encodeText(text: string): TranslationResult<BrailleItem[]> {
  const chars = Array.from(text), issues: TranslationIssue[] = [];
  const value: BrailleItem[] = [], itemOrigins: number[] = [];
  let numeric = false;
  const append = (item: BrailleItem, origin: number) => { value.push(item); itemOrigins.push(origin); };
  for (let index = 0; index < chars.length; index += 1) {
    const char = chars[index]!;
    if (isDigit(char)) {
      if (!numeric) append({ kind: 'cell', mask: WORD_NUMBER_MASK }, index);
      append({ kind: 'cell', mask: WORD_LATIN_MASKS[(Number(char) + 9) % 10]! }, index);
      numeric = true;
      continue;
    }
    numeric = false;
    if (isLetter(char)) {
      if (char >= 'A' && char <= 'Z') append({ kind: 'cell', mask: WORD_CAPITAL_MASK }, index);
      append({ kind: 'cell', mask: WORD_LATIN_MASKS[char.toLowerCase().charCodeAt(0) - 97]! }, index);
    } else if (char === ' ') append({ kind: 'space' }, index);
    else if (char === '\n') append({ kind: 'newline' }, index);
    else {
      const mark = punctuation.find(([symbol]) => symbol === char);
      if (mark) append({ kind: 'cell', mask: mark[1] }, index);
      else {
        append({ kind: 'cell', mask: 0 }, index);
        addIssue(issues, index, 'unsupported_character', 'Karakter ini belum didukung. Ganti dengan huruf, angka, atau tanda baca yang tersedia.');
      }
    }
  }
  checkTextRules(chars, chars.map((_, index) => index), issues);
  if (value.length > MAX_BRAILLE_CELLS) addIssue(issues, itemOrigins[MAX_BRAILLE_CELLS]!, 'braille_limit', `Braille maksimal ${MAX_BRAILLE_CELLS} sel, termasuk spasi dan baris baru. Masukan tetap dipertahankan.`);
  return finish(value, issues);
}

/**
 * Interpret committed six-dot cells in the translator's nine-punctuation scope.
 * Unlike the scanner, a question-mark cell is a question mark in this scope.
 * Identical parenthesis cells alternate open/close; a lone opening stays invalid.
 * Mask 0 is an unfinished cell, never an implicit space. Unknown cells produce
 * replacement characters and indexed issues. Indicator cells emit no text when valid.
 */
export function decodeCells(items: readonly BrailleItem[]): TranslationResult<string> {
  const issues: TranslationIssue[] = [], chars: string[] = [], origins: number[] = [];
  let mode: 'letters' | 'numbers' | undefined;
  let capitalWord = false;
  let capitalPending: number | undefined;
  let numberPending: number | undefined;
  let parenthesisOpen = false;
  const append = (char: string, origin: number) => { chars.push(char); origins.push(origin); };
  const flushIndicators = () => {
    if (capitalPending !== undefined) {
      addIssue(issues, capitalPending, 'incomplete_indicator', 'Tanda kapital harus diikuti huruf.');
      append(REPLACEMENT, capitalPending);
      capitalPending = undefined;
    }
    if (numberPending !== undefined) {
      addIssue(issues, numberPending, 'incomplete_indicator', 'Tanda angka harus diikuti pola a–j untuk digit 1–9 atau 0.');
      append(REPLACEMENT, numberPending);
      numberPending = undefined;
    }
  };
  const boundary = () => {
    flushIndicators();
    mode = undefined;
    capitalWord = false;
  };
  if (items.length > MAX_BRAILLE_CELLS) addIssue(issues, MAX_BRAILLE_CELLS, 'braille_limit', `Braille maksimal ${MAX_BRAILLE_CELLS} sel, termasuk spasi dan baris baru. Masukan tetap dipertahankan.`);
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index]!;
    if (item.kind === 'space' || item.kind === 'newline') {
      boundary();
      append(item.kind === 'space' ? ' ' : '\n', index);
      continue;
    }
    const mask = item.mask;
    if (!Number.isInteger(mask) || mask < 0 || mask > 63 || mask === 0) {
      boundary();
      addIssue(issues, index, mask === 0 ? 'incomplete_cell' : 'invalid_mask', mask === 0 ? 'Sel belum memiliki titik. Isi sel atau gunakan tombol Spasi.' : 'Pola sel harus memakai enam titik Braille yang valid.');
      append(REPLACEMENT, index);
      continue;
    }
    if (mask === WORD_CAPITAL_MASK) {
      let count = 1;
      while (items[index + count]?.kind === 'cell' && (items[index + count] as { kind: 'cell'; mask: number }).mask === WORD_CAPITAL_MASK) count += 1;
      if (mode === 'numbers') addIssue(issues, index, 'mixed_alphanumeric', 'Pisahkan huruf dan angka dengan spasi, baris baru, atau tanda baca.');
      flushIndicators();
      if (count > 2) {
        addIssue(issues, index, 'invalid_indicator', 'Gunakan satu tanda kapital per huruf atau dua tanda di awal kata.');
        append(REPLACEMENT, index);
        mode = undefined;
        capitalWord = false;
      } else {
        if (capitalWord || (count === 2 && mode === 'letters')) addIssue(issues, index, 'invalid_indicator', 'Dua tanda kapital hanya boleh berada di awal kata.');
        capitalWord = count === 2;
        capitalPending = index;
        mode = 'letters';
      }
      index += count - 1;
      continue;
    }
    if (mask === WORD_NUMBER_MASK) {
      if (mode === 'letters') addIssue(issues, index, 'mixed_alphanumeric', 'Pisahkan huruf dan angka dengan spasi, baris baru, atau tanda baca.');
      if (mode === 'numbers') addIssue(issues, index, 'invalid_indicator', 'Gunakan satu tanda angka di awal bilangan.');
      flushIndicators();
      capitalWord = false;
      mode = 'numbers';
      numberPending = index;
      continue;
    }
    const letter = WORD_LATIN_MASKS.findIndex((candidate) => candidate === mask);
    if (letter >= 0) {
      if (mode === 'numbers' && letter < 10) {
        append(String((letter + 1) % 10), index);
        numberPending = undefined;
      } else {
        if (mode === 'numbers') {
          addIssue(issues, index, 'mixed_alphanumeric', 'Pisahkan huruf dan angka dengan spasi, baris baru, atau tanda baca.');
          flushIndicators();
        }
        append(String.fromCharCode((capitalWord || capitalPending !== undefined ? 65 : 97) + letter), index);
        capitalPending = undefined;
        mode = 'letters';
      }
      continue;
    }
    const mark = punctuation.find(([, candidate]) => candidate === mask);
    boundary();
    if (mark) {
      if (mask === WORD_PUNCTUATION_MASKS['(']) {
        append(parenthesisOpen ? ')' : '(', index);
        parenthesisOpen = !parenthesisOpen;
      } else append(mark[0], index);
    } else {
      addIssue(issues, index, 'unsupported_mask', 'Pola titik ini belum termasuk huruf, angka, atau tanda baca yang didukung.');
      append(REPLACEMENT, index);
    }
  }
  flushIndicators();
  checkTextRules(chars, origins, issues);
  return finish(chars.join(''), issues);
}
