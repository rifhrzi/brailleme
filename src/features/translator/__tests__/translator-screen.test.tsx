import { act, fireEvent, render, screen } from '@testing-library/react-native';
import * as ReactNative from 'react-native';
import { SafeAreaInsetsContext } from 'react-native-safe-area-context';

import { getTranslatorDraft, updateTranslatorDraft } from '../draft';
import { TranslatorScreen } from '../translator-screen';

function renderTranslator() {
  return render(
    <SafeAreaInsetsContext.Provider value={{ top: 24, right: 0, bottom: 0, left: 0 }}>
      <TranslatorScreen />
    </SafeAreaInsetsContext.Provider>,
  );
}

async function mode(label: 'Teks ke Braille' | 'Braille ke Teks') {
  await fireEvent.press(screen.getByRole('tab', { name: label }));
}

async function press(label: string) {
  await fireEvent.press(screen.getByRole('button', { name: label }));
}

describe('Penerjemah interaktif', () => {
  beforeEach(() => {
    updateTranslatorDraft({ type: 'clear' });
    updateTranslatorDraft({ type: 'mode', mode: 'braille-to-text' });
    updateTranslatorDraft({ type: 'clear' });
    updateTranslatorDraft({ type: 'mode', mode: 'text-to-braille' });
    updateTranslatorDraft({ type: 'clear' });
  });

  afterEach(() => jest.restoreAllMocks());

  it('mengubah pratinjau saat mengetik dan memindahkan teks valid dua arah', async () => {
    await renderTranslator();
    expect(screen.getByRole('header', { name: 'Penerjemah' })).toBeOnTheScreen();
    await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), 'a');
    expect(screen.getByTestId('braille-cell-0')).toHaveAccessibleName('Sel 1, titik 1');
    await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), 'Baca? 001');
    await mode('Braille ke Teks');
    expect(screen.getByTestId('translated-text')).toHaveTextContent('Baca? 001');
    expect(screen.getByRole('tab', { name: 'Braille ke Teks' }).props.accessibilityState).toMatchObject({ selected: true });
    await mode('Teks ke Braille');
    expect(screen.getByDisplayValue('Baca? 001')).toBeOnTheScreen();
  });

  it('memilih sel identik kedua dan hanya mengubah enam titik sel itu', async () => {
    await renderTranslator();
    await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), 'aa');
    await mode('Braille ke Teks');
    await fireEvent.press(screen.getByTestId('braille-cell-1'));
    expect(screen.getByTestId('braille-cell-1').props.accessibilityState).toMatchObject({ selected: true });
    await fireEvent.press(screen.getByRole('checkbox', { name: 'Titik 2, tidak aktif' }));
    expect(screen.getByTestId('translated-text')).toHaveTextContent('ab');
    expect(screen.getByRole('checkbox', { name: 'Titik 2, aktif' })).toBeChecked();
    expect(screen.getByTestId('braille-cell-0')).toHaveAccessibleName('Sel 1, titik 1');
    expect(screen.getByTestId('braille-cell-1')).toHaveAccessibleName('Sel 2, titik 1, 2');
  });

  it('membedakan sel kosong dari spasi, menyisipkan setelah pilihan dan menghapusnya', async () => {
    await renderTranslator();
    await mode('Braille ke Teks');
    await press('Tambah sel');
    expect(screen.getByTestId('braille-cell-0')).toHaveAccessibleName('Sel 1, kosong, belum lengkap');
    expect(screen.getByTestId('translation-issues')).toBeOnTheScreen();
    await fireEvent.press(screen.getByTestId('translator-dot-1'));
    expect(screen.getByTestId('translated-text')).toHaveTextContent('a');
    await press('Spasi');
    expect(screen.getByTestId('braille-cell-1')).toHaveAccessibleName('Sel 2, spasi');
    expect(screen.getByTestId('translator-dot-1')).toBeDisabled();
    await press('Baris baru');
    expect(getTranslatorDraft().items.map((item) => item.kind)).toEqual(['cell', 'space', 'newline']);
    await fireEvent.press(screen.getByTestId('braille-cell-0'));
    await press('Tambah sel');
    expect(getTranslatorDraft().items.map((item) => item.kind)).toEqual(['cell', 'cell', 'space', 'newline']);
    await press('Hapus sel');
    expect(getTranslatorDraft().items.map((item) => item.kind)).toEqual(['cell', 'space', 'newline']);
    await press('Hapus semua');
    expect(getTranslatorDraft().items).toEqual([]);
    expect(screen.getByRole('button', { name: 'Hapus sel' })).toBeDisabled();
  });

  it('menjaga teks tidak valid saat berpindah mode lalu kembali', async () => {
    await renderTranslator();
    await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), 'A1');
    expect(screen.getByTestId('translation-issues')).toBeOnTheScreen();
    await mode('Braille ke Teks');
    expect(screen.getByText(/Teks tetap tersimpan/)).toBeOnTheScreen();
    await mode('Teks ke Braille');
    expect(screen.getByDisplayValue('A1')).toBeOnTheScreen();
    expect(screen.getByTestId('translation-issues')).toBeOnTheScreen();
  });

  it('menjaga titik belum lengkap saat pulang dari mode teks, kemudian dapat memperbaikinya', async () => {
    await renderTranslator();
    await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), 'a');
    await mode('Braille ke Teks');
    await fireEvent.press(screen.getByTestId('translator-dot-1'));
    await mode('Teks ke Braille');
    expect(screen.getByDisplayValue('a')).toBeOnTheScreen();
    await mode('Braille ke Teks');
    expect(screen.getByTestId('braille-cell-0')).toHaveAccessibleName('Sel 1, kosong, belum lengkap');
    await fireEvent.press(screen.getByTestId('translator-dot-1'));
    expect(screen.queryByTestId('translation-issues')).toBeNull();
    expect(screen.getByTestId('translated-text')).toHaveTextContent('a');
  });

  it('menyimpan draf di memori setelah layar dilepas dan dibuka kembali', async () => {
    const first = await renderTranslator();
    await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), 'Baca?');
    await mode('Braille ke Teks');
    await fireEvent.press(screen.getByTestId('braille-cell-2'));
    await first.unmount();
    await renderTranslator();
    expect(screen.getByRole('tab', { name: 'Braille ke Teks' }).props.accessibilityState).toMatchObject({ selected: true });
    expect(screen.getByTestId('braille-cell-2').props.accessibilityState).toMatchObject({ selected: true });
    expect(screen.getByTestId('translated-text')).toHaveTextContent('Baca?');
    expect(globalThis.localStorage.setItem).not.toHaveBeenCalled();
  });

  it('menampilkan posisi masalah dan jumlah masukan tanpa memotong tempel berlebih', async () => {
    await renderTranslator();
    const text = 'a'.repeat(200) + '🙂';
    await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), text);
    expect(screen.getByDisplayValue(text)).toBeOnTheScreen();
    expect(screen.getByText('201 / 200 karakter')).toBeOnTheScreen();
    expect(screen.getAllByText(/Karakter 201:/).length).toBeGreaterThan(0);
    expect(screen.getByText('Perbaiki teks untuk melihat hasil Braille.')).toBeOnTheScreen();
    expect(screen.getByLabelText('Teks yang diterjemahkan').props.maxLength).toBeUndefined();
  });

  it('menghitung 400 sel dan menolak tambahan tanpa mengganti sel yang ada', async () => {
    updateTranslatorDraft({ type: 'mode', mode: 'braille-to-text' });
    for (let index = 0; index < 400; index++) updateTranslatorDraft({ type: 'insert', item: { kind: 'cell', mask: 1 } });
    await renderTranslator();
    const previous = getTranslatorDraft().items;
    await press('Tambah sel');
    expect(getTranslatorDraft().items).toBe(previous);
    expect(screen.getByTestId('braille-count')).toHaveTextContent(/400 \/ 400 sel/);
    expect(screen.getByText(/Batas 400 sel tercapai/)).toBeOnTheScreen();
  });

  it('mengganti teks lengkap dari papan ketik tanpa menggandakan huruf atau spasi', async () => {
    await renderTranslator();
    for (const text of ['A', 'Ak', 'Aku', 'Aku  ', 'Aku  007', 'Aku  007\nBACA!']) {
      await fireEvent.changeText(screen.getByLabelText('Teks yang diterjemahkan'), text);
      expect(getTranslatorDraft().text).toBe(text);
    }
    await mode('Braille ke Teks');
    expect(screen.getByTestId('translated-text').props.children).toBe('Aku  007\nBACA!');
  });

  it('mengecilkan viewport sampai tepi papan ketik dengan inset atas dan memulihkannya setelah ditutup', async () => {
    await renderTranslator();
    const viewport = screen.getByTestId('translator-keyboard-viewport');
    await fireEvent(viewport, 'layout', {
      nativeEvent: { layout: { x: 0, y: 0, width: 360, height: 760 } },
      persist: jest.fn(),
    });
    const event = {
      duration: 0,
      easing: 'keyboard',
      endCoordinates: { screenX: 0, screenY: 500, width: 360, height: 300 },
    };
    try {
      await act(() => {
        ReactNative.DeviceEventEmitter.emit('keyboardWillShow', event);
        ReactNative.DeviceEventEmitter.emit('keyboardDidShow', event);
      });
      const adjustedStyle = ReactNative.StyleSheet.flatten(screen.getByTestId('translator-keyboard-viewport').props.style);
      expect(adjustedStyle).toMatchObject(process.env.EXPO_OS === 'ios'
        ? { paddingBottom: 284, flex: 1 }
        : { height: 476, flex: 0 });
    } finally {
      await act(() => {
        ReactNative.DeviceEventEmitter.emit('keyboardWillHide', event);
        ReactNative.DeviceEventEmitter.emit('keyboardDidHide', event);
      });
    }
    expect(ReactNative.StyleSheet.flatten(screen.getByTestId('translator-keyboard-viewport').props.style).height).toBeUndefined();
    expect(ReactNative.StyleSheet.flatten(screen.getByTestId('translator-keyboard-viewport').props.style).flex).toBe(1);
  });

  it.each([{ width: 320, height: 640, scale: 1, fontScale: 2 }, { width: 760, height: 1024, scale: 1, fontScale: 1 }])(
    'menjaga kontrol besar dan pengguliran saat ukuran $width dengan skala huruf $fontScale', async (dimensions) => {
      const original = ReactNative.Dimensions.get('window');
      await act(() => ReactNative.Dimensions.set({ window: dimensions }));
      try {
        await renderTranslator();
      expect(ReactNative.StyleSheet.flatten(screen.getByTestId('translator-mode-switch').props.style).flexDirection)
        .toBe(dimensions.width < 380 || dimensions.fontScale >= 1.5 ? 'column' : 'row');
      const scroll = screen.getByTestId('translator-screen');
      expect(scroll.props.keyboardShouldPersistTaps).toBe('handled');
      expect(scroll.props.automaticallyAdjustKeyboardInsets).toBe(true);
      expect(screen.getByLabelText('Teks yang diterjemahkan').props.multiline).toBe(true);
      expect(screen.getByLabelText('Teks yang diterjemahkan').props.allowFontScaling).toBe(true);
      await mode('Braille ke Teks');
      await press('Tambah sel');
      const dots = screen.getAllByRole('checkbox');
      expect(dots.map((dot) => dot.props.accessibilityLabel)).toEqual([1, 4, 2, 5, 3, 6].map((dot) => `Titik ${dot}, tidak aktif`));
      for (const dot of dots) {
        const style = ReactNative.StyleSheet.flatten(dot.props.style);
        expect(style.minWidth).toBeGreaterThanOrEqual(56);
        expect(style.minHeight).toBeGreaterThanOrEqual(56);
        expect(style.height).toBeUndefined();
        expect(dot).not.toBeChecked();
      }
      expect(screen.getByRole('button', { name: 'Hapus semua' })).toBeOnTheScreen();
      expect(screen.queryByText(/experimental|debug|benchmark|model/i)).toBeNull();
      } finally {
        await act(() => ReactNative.Dimensions.set({ window: original }));
      }
    },
  );
});
