import {
  decodeCells,
  encodeText,
  MAX_BRAILLE_CELLS,
  type BrailleItem,
} from './codec';

export type TranslatorMode = 'text-to-braille' | 'braille-to-text';

export type TranslatorDraft = {
  mode: TranslatorMode;
  text: string;
  items: readonly BrailleItem[];
  selectedIndex: number | null;
  notice: string | null;
  preserveOnReturn: boolean;
};

export type TranslatorAction =
  | { type: 'text'; text: string }
  | { type: 'mode'; mode: TranslatorMode }
  | { type: 'select'; index: number }
  | { type: 'toggle-dot'; dot: number }
  | { type: 'insert'; item: BrailleItem }
  | { type: 'delete' }
  | { type: 'clear' };

export function createTranslatorDraft(): TranslatorDraft {
  return {
    mode: 'text-to-braille',
    text: '',
    items: [],
    selectedIndex: null,
    notice: null,
    preserveOnReturn: false,
  };
}

export function translatorDraftReducer(
  state: TranslatorDraft,
  action: TranslatorAction,
): TranslatorDraft {
  switch (action.type) {
    case 'text':
      return { ...state, text: action.text, notice: null, preserveOnReturn: false };
    case 'mode': {
      if (action.mode === state.mode) return state;
      // An unsuccessful transfer must also survive a round trip through the
      // other mode. Only editing that destination enables a new transfer.
      if (state.preserveOnReturn) {
        return { ...state, mode: action.mode, notice: null, preserveOnReturn: false };
      }
      if (action.mode === 'braille-to-text') {
        const translated = encodeText(state.text);
        return translated.valid
          ? {
              ...state,
              mode: action.mode,
              items: translated.value,
              selectedIndex: translated.value.length ? 0 : null,
              notice: null,
              preserveOnReturn: false,
            }
          : {
              ...state,
              mode: action.mode,
              preserveOnReturn: true,
              notice:
                'Teks belum dapat diterjemahkan. Teks tetap tersimpan; draf Braille sebelumnya tidak berubah.',
            };
      }
      const translated = decodeCells(state.items);
      return translated.valid
        ? { ...state, mode: action.mode, text: translated.value, notice: null, preserveOnReturn: false }
        : {
            ...state,
            mode: action.mode,
            preserveOnReturn: true,
            notice:
              'Braille belum dapat diterjemahkan. Susunan titik tetap tersimpan; draf teks sebelumnya tidak berubah.',
          };
    }
    case 'select':
      return Number.isInteger(action.index) && state.items[action.index]
        ? { ...state, selectedIndex: action.index, notice: null }
        : state;
    case 'toggle-dot': {
      const selected = state.selectedIndex;
      const item = selected === null ? undefined : state.items[selected];
      if (
        state.mode !== 'braille-to-text' ||
        selected === null ||
        item?.kind !== 'cell' ||
        !Number.isInteger(action.dot) ||
        action.dot < 1 ||
        action.dot > 6
      ) return state;
      return {
        ...state,
        notice: null,
        preserveOnReturn: false,
        items: state.items.map((existing, index) =>
          index === selected
            ? { kind: 'cell', mask: item.mask ^ (1 << (action.dot - 1)) }
            : existing,
        ),
      };
    }
    case 'insert': {
      if (state.mode !== 'braille-to-text') return state;
      if (state.items.length >= MAX_BRAILLE_CELLS) {
        return {
          ...state,
          notice: 'Batas 400 sel tercapai. Hapus sebuah sel, spasi, atau baris baru sebelum menambah.',
        };
      }
      const index = state.selectedIndex === null
        ? state.items.length
        : state.selectedIndex + 1;
      return {
        ...state,
        items: [...state.items.slice(0, index), { ...action.item }, ...state.items.slice(index)],
        selectedIndex: index,
        notice: null,
        preserveOnReturn: false,
      };
    }
    case 'delete': {
      if (state.mode !== 'braille-to-text' || state.selectedIndex === null) return state;
      const items = state.items.filter((_, index) => index !== state.selectedIndex);
      return {
        ...state,
        items,
        selectedIndex: items.length ? Math.min(state.selectedIndex, items.length - 1) : null,
        notice: null,
        preserveOnReturn: false,
      };
    }
    case 'clear':
      return state.mode === 'text-to-braille'
        ? { ...state, text: '', notice: null, preserveOnReturn: false }
        : { ...state, items: [], selectedIndex: null, notice: null, preserveOnReturn: false };
  }
}

// Session memory only: leaving a tab never writes translator input to storage.
let draft = createTranslatorDraft();
const listeners = new Set<() => void>();

export function getTranslatorDraft() {
  return draft;
}

export function subscribeTranslatorDraft(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function updateTranslatorDraft(action: TranslatorAction) {
  const next = translatorDraftReducer(draft, action);
  if (next === draft) return;
  draft = next;
  listeners.forEach((listener) => listener());
}
