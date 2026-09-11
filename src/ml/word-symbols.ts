import { WORD_LATIN_MASKS, WORD_MAX_CELLS, WORD_MAX_CHARACTERS } from '@/ml/word-model-contract';

export const WORD_CAPITAL_MASK = 32;
export const WORD_NUMBER_MASK = 60;

export type WordSymbolRole = 'letter' | 'digit' | 'capital_indicator' | 'number_indicator' | 'punctuation';
export type WordSymbolRejection = 'unsupported_pattern' | 'invalid_sequence' | 'ambiguous_symbol' | 'too_many_cells' | 'too_many_characters';
export interface WordSymbol {
  readonly mask: number;
  readonly role: WordSymbolRole;
  /** Indicators occupy a physical cell but emit no printed character. */
  readonly output: string;
}
export type WordSymbolResult =
  | { readonly status: 'success'; readonly text: string; readonly symbols: readonly WordSymbol[] }
  | { readonly status: 'rejected'; readonly reason: WordSymbolRejection };

/** Indonesian uncontracted token; no implicit capitalization or number guessing. */
function parseBasicMasks(masks: readonly number[]): WordSymbolResult {
  'worklet';
  if (masks.length === 0) return { status: 'rejected', reason: 'invalid_sequence' };
  if (masks.length > WORD_MAX_CELLS) return { status: 'rejected', reason: 'too_many_cells' };
  const symbols: WordSymbol[] = [];
  let text = '';
  const numeric = masks[0] === WORD_NUMBER_MASK;
  const capitals = masks[0] === WORD_CAPITAL_MASK && masks[1] === WORD_CAPITAL_MASK;
  let capitalNext = false;
  for (let index = 0; index < masks.length; index += 1) {
    const mask = masks[index]!;
    if (!Number.isInteger(mask) || mask < 0 || mask > 63) return { status: 'rejected', reason: 'unsupported_pattern' };
    if (mask === WORD_NUMBER_MASK) {
      if (index !== 0) return { status: 'rejected', reason: 'invalid_sequence' };
      symbols.push({ mask, role: 'number_indicator', output: '' });
      continue;
    }
    if (mask === WORD_CAPITAL_MASK) {
      if (numeric || (capitals ? index > 1 : capitalNext)) return { status: 'rejected', reason: 'invalid_sequence' };
      if (!capitals) capitalNext = true;
      symbols.push({ mask, role: 'capital_indicator', output: '' });
      continue;
    }
    const letter = WORD_LATIN_MASKS.findIndex((candidate) => candidate === mask);
    if (letter < 0) return { status: 'rejected', reason: 'unsupported_pattern' };
    if (numeric && letter >= 10) return { status: 'rejected', reason: 'invalid_sequence' };
    const output = numeric ? String((letter + 1) % 10) : String.fromCharCode((capitals || capitalNext ? 65 : 97) + letter);
    symbols.push({ mask, role: numeric ? 'digit' : 'letter', output });
    text += output;
    capitalNext = false;
    if (text.length > WORD_MAX_CHARACTERS) return { status: 'rejected', reason: 'too_many_characters' };
  }
  if (!text || capitalNext) return { status: 'rejected', reason: 'invalid_sequence' };
  return { status: 'success', text, symbols };
}

export const WORD_PUNCTUATION_MASKS = { '.': 50, ',': 2, '?': 38, '!': 22, ':': 18, ';': 6, '-': 36, '(': 54, ')': 54 } as const;

function punctuation(mask: number, output: string): WordSymbol {
  'worklet';
  return { mask, role: 'punctuation', output };
}

function finalPunctuation(mask: number | undefined): string | undefined {
  'worklet';
  switch (mask) {
    case 50: return '.';
    case 2: return ',';
    case 38: return '?';
    case 22: return '!';
    case 18: return ':';
    case 6: return ';';
    default: return undefined;
  }
}

/** Exact frozen word_symbols_id_v4 grammar. No inferred signs, decimals or spaces. */
export function parseWordMasks(masks: readonly number[]): WordSymbolResult {
  'worklet';
  if (!masks.length) return { status: 'rejected', reason: 'invalid_sequence' };
  if (masks.length > WORD_MAX_CELLS) return { status: 'rejected', reason: 'too_many_cells' };
  if (masks.some((mask) => !Number.isInteger(mask) || mask < 0 || mask > 63)) return { status: 'rejected', reason: 'unsupported_pattern' };
  if (masks.length === 1) {
    const mask = masks[0]!;
    if (mask === 38 || mask === 54) return { status: 'rejected', reason: 'ambiguous_symbol' };
    const standalone = mask === 36 ? '-' : finalPunctuation(mask);
    if (standalone) return { status: 'success', text: standalone, symbols: [punctuation(mask, standalone)] };
  }
  let cells = [...masks];
  const before: WordSymbol[] = [];
  const after: WordSymbol[] = [];
  const terminal = finalPunctuation(cells[cells.length - 1]);
  if (terminal) after.push(punctuation(cells.pop()!, terminal));
  if (cells.includes(54)) {
    if (cells.length < 3 || cells.filter((mask) => mask === 54).length !== 2 || cells[0] !== 54 || cells[cells.length - 1] !== 54) {
      return { status: 'rejected', reason: 'ambiguous_symbol' };
    }
    cells = cells.slice(1, -1);
    before.push(punctuation(54, '('));
    after.unshift(punctuation(54, ')'));
    const innerTerminal = finalPunctuation(cells[cells.length - 1]);
    if (innerTerminal) {
      if (terminal) return { status: 'rejected', reason: 'invalid_sequence' };
      after.unshift(punctuation(cells.pop()!, innerTerminal));
    }
  }
  if (!cells.length) return { status: 'rejected', reason: 'invalid_sequence' };
  const segments: number[][] = [[]];
  for (const mask of cells) {
    if (mask === 36) segments.push([]);
    else segments[segments.length - 1]!.push(mask);
  }
  if (segments.some((segment) => !segment.length)) return { status: 'rejected', reason: 'invalid_sequence' };
  const symbols = [...before];
  for (let index = 0; index < segments.length; index += 1) {
    const parsed = parseBasicMasks(segments[index]!);
    if (parsed.status === 'rejected') return parsed;
    if (segments.length > 1 && !/^[A-Za-z]+$/.test(parsed.text)) return { status: 'rejected', reason: 'invalid_sequence' };
    if (index) symbols.push(punctuation(36, '-'));
    symbols.push(...parsed.symbols);
  }
  symbols.push(...after);
  const text = symbols.map((symbol) => symbol.output).join('');
  if (text.length > WORD_MAX_CHARACTERS) return { status: 'rejected', reason: 'too_many_characters' };
  return { status: 'success', text, symbols };
}
