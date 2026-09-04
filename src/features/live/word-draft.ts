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

export type WordDraftState = {
  readonly letters: readonly BrailleLetter[];
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
  | { readonly type: 'clear_word' };

export const INITIAL_WORD_DRAFT_STATE: WordDraftState = Object.freeze({
  letters: Object.freeze([]),
  consumedEvidenceKeys: Object.freeze([]),
  revision: 0,
});

export function isBrailleLetter(value: unknown): value is BrailleLetter {
  return typeof value === 'string' && /^[A-Z]$/.test(value);
}

export function getWordText(state: WordDraftState): string {
  return state.letters.join('');
}

export function wordDraftReducer(
  state: WordDraftState,
  action: WordDraftAction,
): WordDraftState {
  switch (action.type) {
    case 'commit_letter': {
      const evidenceKey = action.evidenceKey.trim();
      if (
        !evidenceKey ||
        !isBrailleLetter(action.letter) ||
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
