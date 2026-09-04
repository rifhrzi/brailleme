import {
  INITIAL_WORD_DRAFT_STATE,
  MAX_WORD_LENGTH,
  getWordText,
  isBrailleLetter,
  wordDraftReducer,
} from '@/features/live/word-draft';

function commit(
  state = INITIAL_WORD_DRAFT_STATE,
  letter = 'A',
  evidenceKey = `evidence-${state.revision}`,
) {
  return wordDraftReducer(state, {
    type: 'commit_letter',
    evidenceKey,
    letter,
  });
}

describe('wordDraftReducer', () => {
  it('menyusun huruf berulang dari evidence epoch berbeda', () => {
    const first = commit(INITIAL_WORD_DRAFT_STATE, 'A', 'session:0:A');
    const second = commit(first, 'A', 'session:1:A');

    expect(getWordText(second)).toBe('AA');
    expect(second.revision).toBe(2);
  });

  it('membuat commit dari evidence yang sama idempoten', () => {
    const first = commit(INITIAL_WORD_DRAFT_STATE, 'B', 'session:0:B');
    const duplicate = commit(first, 'B', 'session:0:B');

    expect(duplicate).toBe(first);
    expect(getWordText(duplicate)).toBe('B');
  });

  it('menolak karakter bukan A-Z dan spasi', () => {
    expect(isBrailleLetter('A')).toBe(true);
    expect(isBrailleLetter('a')).toBe(false);
    expect(isBrailleLetter('?')).toBe(false);
    expect(isBrailleLetter(' ')).toBe(false);
    expect(commit(INITIAL_WORD_DRAFT_STATE, '?')).toBe(
      INITIAL_WORD_DRAFT_STATE,
    );
  });

  it('membatasi draft pada dua belas huruf', () => {
    let state = INITIAL_WORD_DRAFT_STATE;
    for (let index = 0; index < MAX_WORD_LENGTH; index += 1) {
      state = commit(state, 'A', `epoch-${index}`);
    }

    const overflow = commit(state, 'B', 'overflow');
    expect(getWordText(state)).toHaveLength(MAX_WORD_LENGTH);
    expect(overflow).toBe(state);
  });

  it('menghapus huruf terakhir tanpa mengizinkan evidence lama dipakai ulang', () => {
    const first = commit(INITIAL_WORD_DRAFT_STATE, 'C', 'session:0:C');
    const removed = wordDraftReducer(first, { type: 'remove_last_letter' });
    const staleCommit = commit(removed, 'C', 'session:0:C');

    expect(getWordText(removed)).toBe('');
    expect(removed.revision).toBe(2);
    expect(staleCommit).toBe(removed);
  });

  it('menghapus kata tetapi mempertahankan ledger evidence', () => {
    const first = commit(INITIAL_WORD_DRAFT_STATE, 'D', 'session:0:D');
    const cleared = wordDraftReducer(first, { type: 'clear_word' });

    expect(getWordText(cleared)).toBe('');
    expect(cleared.consumedEvidenceKeys).toEqual(['session:0:D']);
    expect(cleared.revision).toBe(2);
  });
});
