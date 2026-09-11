import { encodeText, type BrailleItem } from '../codec';
import { createTranslatorDraft, translatorDraftReducer as reduce, type TranslatorDraft } from '../draft';

function brailleDraft(items: readonly BrailleItem[]): TranslatorDraft {
  return { ...createTranslatorDraft(), mode: 'braille-to-text', items, selectedIndex: items.length ? 0 : null };
}

describe('Draf penerjemah', () => {
  it('memindahkan hasil valid dengan kapital, nol awal, spasi, dan baris baru', () => {
    const text = 'Baca? 001\nABC!';
    let state = reduce(createTranslatorDraft(), { type: 'text', text });
    state = reduce(state, { type: 'mode', mode: 'braille-to-text' });
    expect(state.items).toEqual(encodeText(text).value);
    state = reduce(state, { type: 'mode', mode: 'text-to-braille' });
    expect(state.text).toBe(text);
  });

  it('menyimpan teks tidak valid dan tujuan ketika pergi lalu kembali tanpa edit', () => {
    const initial = { ...createTranslatorDraft(), text: 'A1', items: [{ kind: 'cell', mask: 3 }] as BrailleItem[] };
    const away = reduce(initial, { type: 'mode', mode: 'braille-to-text' });
    expect(away).toMatchObject({ text: 'A1', items: initial.items, preserveOnReturn: true });
    const selectedOnly = reduce(away, { type: 'select', index: 0 });
    const returned = reduce(selectedOnly, { type: 'mode', mode: 'text-to-braille' });
    expect(returned.text).toBe('A1');
    expect(returned.items).toBe(initial.items);
  });

  it('menyimpan sel belum lengkap saat pulang dari draf teks yang valid', () => {
    const initial = { ...brailleDraft([{ kind: 'cell', mask: 0 }]), text: 'lama' };
    const away = reduce(initial, { type: 'mode', mode: 'text-to-braille' });
    expect(away.text).toBe('lama');
    const returned = reduce(away, { type: 'mode', mode: 'braille-to-text' });
    expect(returned.items).toBe(initial.items);
    expect(returned.items).toEqual([{ kind: 'cell', mask: 0 }]);
  });

  it('memindahkan hasil tujuan yang benar-benar diedit setelah transfer gagal', () => {
    let state = reduce({ ...createTranslatorDraft(), text: 'A1' }, { type: 'mode', mode: 'braille-to-text' });
    state = reduce(state, { type: 'insert', item: { kind: 'cell', mask: 0 } });
    state = reduce(state, { type: 'toggle-dot', dot: 1 });
    state = reduce(state, { type: 'mode', mode: 'text-to-braille' });
    expect(state.text).toBe('a');

    state = reduce({ ...brailleDraft([{ kind: 'cell', mask: 0 }]), text: 'lama' }, { type: 'mode', mode: 'text-to-braille' });
    state = reduce(state, { type: 'text', text: 'baru' });
    state = reduce(state, { type: 'mode', mode: 'braille-to-text' });
    expect(state.items).toEqual(encodeText('baru').value);
  });

  it('menyunting sel identik berdasarkan indeks tanpa mengubah draf lama', () => {
    const initial = brailleDraft([{ kind: 'cell', mask: 1 }, { kind: 'cell', mask: 1 }]);
    const selected = reduce(initial, { type: 'select', index: 1 });
    const edited = reduce(selected, { type: 'toggle-dot', dot: 6 });
    expect(edited.items).toEqual([{ kind: 'cell', mask: 1 }, { kind: 'cell', mask: 33 }]);
    expect(initial.items).toEqual([{ kind: 'cell', mask: 1 }, { kind: 'cell', mask: 1 }]);
  });

  it('menyisipkan setelah pilihan dan menghapus spasi maupun baris baru', () => {
    let state = brailleDraft([{ kind: 'cell', mask: 1 }, { kind: 'cell', mask: 3 }]);
    state = reduce(state, { type: 'insert', item: { kind: 'space' } });
    state = reduce(state, { type: 'insert', item: { kind: 'newline' } });
    expect(state.items).toEqual([{ kind: 'cell', mask: 1 }, { kind: 'space' }, { kind: 'newline' }, { kind: 'cell', mask: 3 }]);
    expect(state.selectedIndex).toBe(2);
    const beforeToggle = state;
    expect(reduce(state, { type: 'toggle-dot', dot: 1 })).toBe(beforeToggle);
    state = reduce(state, { type: 'delete' });
    state = reduce(state, { type: 'select', index: 1 });
    state = reduce(state, { type: 'delete' });
    expect(state.items).toEqual([{ kind: 'cell', mask: 1 }, { kind: 'cell', mask: 3 }]);
  });

  it('menghapus sel terakhir dan mengembalikan pilihan ke kosong', () => {
    const state = reduce(brailleDraft([{ kind: 'cell', mask: 0 }]), { type: 'delete' });
    expect(state).toMatchObject({ items: [], selectedIndex: null });
    expect(reduce(state, { type: 'delete' })).toBe(state);
  });

  it('menahan sel ke-401 dengan pesan, menghitung pemisah dalam batas', () => {
    const initial = brailleDraft(Array.from({ length: 400 }, (_, index) => ({ kind: index % 2 ? 'newline' : 'space' } as BrailleItem)));
    const blocked = reduce(initial, { type: 'insert', item: { kind: 'cell', mask: 0 } });
    expect(blocked.items).toBe(initial.items);
    expect(blocked.notice).toMatch(/400/);
  });

  it('tidak memotong hasil tempel panjang atau karakter di luar cakupan', () => {
    const text = 'é🙂' + 'a'.repeat(201);
    expect(reduce(createTranslatorDraft(), { type: 'text', text }).text).toBe(text);
  });

  it('menghapus mode aktif saja, lalu hasil kosong valid ikut dipindahkan', () => {
    const initial = { ...brailleDraft([{ kind: 'cell', mask: 1 }]), text: 'a' };
    const cleared = reduce(initial, { type: 'clear' });
    expect(cleared).toMatchObject({ text: 'a', items: [] });
    expect(reduce(cleared, { type: 'mode', mode: 'text-to-braille' }).text).toBe('');
    const clearedText = reduce({ ...initial, mode: 'text-to-braille' }, { type: 'clear' });
    expect(clearedText.items).toBe(initial.items);
  });

  it('mengabaikan pilihan dan nomor titik yang tidak ada', () => {
    const initial = brailleDraft([{ kind: 'cell', mask: 1 }]);
    for (const index of [-1, 3, 0.5, NaN]) expect(reduce(initial, { type: 'select', index })).toBe(initial);
    for (const dot of [0, 7, 1.5, NaN]) expect(reduce(initial, { type: 'toggle-dot', dot })).toBe(initial);
    expect(reduce(createTranslatorDraft(), { type: 'toggle-dot', dot: 1 })).toEqual(createTranslatorDraft());
  });
});
