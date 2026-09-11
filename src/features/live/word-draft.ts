export const MAX_WORD_LENGTH = 12;

export type BrailleLetter =
  | 'A'
  | 'B'
  | 'C'
  | 'D'
  | 'E'
  | 'F'
  | 'G'
  | 'H'
  | 'I'
  | 'J'
  | 'K'
  | 'L'
  | 'M'
  | 'N'
  | 'O'
  | 'P'
  | 'Q'
  | 'R'
  | 'S'
  | 'T'
  | 'U'
  | 'V'
  | 'W'
  | 'X'
  | 'Y'
  | 'Z';

export type WordCharacter = BrailleLetter | Lowercase<BrailleLetter> | '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
  | '.' | ',' | '?' | '!' | ':' | ';' | '-' | '(' | ')';

const WORD_TOKEN = '(?:[A-Za-z]+(?:-[A-Za-z]+)*|[0-9]+)';
// One optional final mark, inside OR outside a balanced pair of parentheses.
// A bare question mark or parenthesis is ambiguous in the physical grammar.
const WORD_TEXT_PATTERN = new RegExp(
  `^(?:[.,!:;-]|${WORD_TOKEN}[.,?!:;]?|\\(${WORD_TOKEN}[.,?!:;]?\\)|\\(${WORD_TOKEN}\\)[.,?!:;])$`,
);

export type WordDraftState = {
  readonly letters: readonly WordCharacter[];
  readonly consumedEvidenceKeys: readonly string[];
  readonly revision: number;
};

export type WordDraftAction =
  | {
      readonly type: 'commit_letter';
      readonly evidenceKey: string;
      readonly letter: string;
    }
  | { readonly type: 'remove_last_letter' }
  | { readonly type: 'replace_text'; readonly text: string; readonly evidenceKey: string }
  | { readonly type: 'clear_word' };

export const INITIAL_WORD_DRAFT_STATE: WordDraftState = Object.freeze({
  letters: Object.freeze([]),
  consumedEvidenceKeys: Object.freeze([]),
  revision: 0,
});

export function isBrailleLetter(value: unknown): value is BrailleLetter {
  return typeof value === 'string' && /^[A-Z]$/.test(value);
}

export function isWordCharacter(value: unknown): value is WordCharacter {
  return typeof value === 'string' && value.length === 1 && /^[A-Za-z0-9.,?!:;()-]$/.test(value);
}

export function isWordText(value: unknown, allowEmpty = false): value is string {
  return typeof value === 'string' && ((allowEmpty && value === '') ||
    (value.length > 0 && value.length <= MAX_WORD_LENGTH &&
      [...value].every(isWordCharacter) && WORD_TEXT_PATTERN.test(value)));
}

export function getWordText(state: WordDraftState): string {
  return state.letters.join('');
}

export function wordDraftReducer(
  state: WordDraftState,
  action: WordDraftAction,
): WordDraftState {
  switch (action.type) {
    case 'replace_text': {
      const evidenceKey = action.evidenceKey.trim();
      if (!isWordText(action.text, true) || (action.text && (!evidenceKey || state.consumedEvidenceKeys.includes(evidenceKey)))) return state;
      if (!action.text && state.letters.length === 0) return state;
      return {
        letters: [...action.text] as WordCharacter[],
        consumedEvidenceKeys: evidenceKey ? [...state.consumedEvidenceKeys, evidenceKey] : state.consumedEvidenceKeys,
        revision: state.revision + 1,
      };
    }
    case 'commit_letter': {
      const evidenceKey = action.evidenceKey.trim();
      if (
        !evidenceKey ||
        !isBrailleLetter(action.letter) ||
        state.letters.some((letter) => !/^[A-Za-z]$/.test(letter)) ||
        state.letters.length >= MAX_WORD_LENGTH ||
        state.consumedEvidenceKeys.includes(evidenceKey)
      ) {
        return state;
      }

      return {
        letters: [...state.letters, action.letter],
        consumedEvidenceKeys: [...state.consumedEvidenceKeys, evidenceKey],
        revision: state.revision + 1,
      };
    }
    case 'remove_last_letter':
      if (state.letters.length === 0) {
        return state;
      }
      return {
        ...state,
        letters: state.letters.slice(0, -1),
        revision: state.revision + 1,
      };
    case 'clear_word':
      if (state.letters.length === 0) {
        return state;
      }
      return {
        ...state,
        letters: [],
        revision: state.revision + 1,
      };
  }
}
