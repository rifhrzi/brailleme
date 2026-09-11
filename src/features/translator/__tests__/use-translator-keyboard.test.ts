import { act, renderHook } from '@testing-library/react-native';
import { Keyboard, type KeyboardEvent, type ScrollView, type TextInput } from 'react-native';

import { useTranslatorKeyboard } from '../use-translator-keyboard';

describe('Papan ketik penerjemah', () => {
  afterEach(() => jest.restoreAllMocks());

  it('menggulir input saat papan ketik tampil dan isi bertambah, lalu berhenti saat blur atau dilepas', async () => {
    let onKeyboard: ((event: KeyboardEvent) => void) | undefined;
    const remove = jest.fn();
    jest.spyOn(Keyboard, 'addListener').mockImplementation((_event, handler) => {
      onKeyboard = handler;
      return { remove } as unknown as ReturnType<typeof Keyboard.addListener>;
    });
    const visible = jest.spyOn(Keyboard, 'isVisible').mockReturnValue(false);
    const scroll = jest.fn();
    const input = {} as TextInput;
    const { result, unmount } = await renderHook(() => useTranslatorKeyboard(24));
    result.current.scrollRef.current = { scrollResponderScrollNativeHandleToKeyboard: scroll } as unknown as ScrollView;
    result.current.inputRef.current = input;

    result.current.onFocus();
    expect(scroll).not.toHaveBeenCalled();
    visible.mockReturnValue(true);
    await act(() => onKeyboard?.({} as KeyboardEvent));
    expect(scroll).toHaveBeenLastCalledWith(input, 40, true);
    result.current.keepInputVisible();
    expect(scroll).toHaveBeenCalledTimes(2);

    result.current.onBlur();
    result.current.keepInputVisible();
    await act(() => onKeyboard?.({} as KeyboardEvent));
    expect(scroll).toHaveBeenCalledTimes(2);

    result.current.onFocus();
    const retained = result.current.keepInputVisible;
    await unmount();
    retained();
    expect(scroll).toHaveBeenCalledTimes(3);
    expect(remove).toHaveBeenCalledTimes(1);
  });

  it('menjaga fokus saat inset berubah dan tidak mengakses input yang dilepas', async () => {
    jest.spyOn(Keyboard, 'addListener').mockReturnValue({ remove: jest.fn() } as unknown as ReturnType<typeof Keyboard.addListener>);
    jest.spyOn(Keyboard, 'isVisible').mockReturnValue(true);
    const scroll = jest.fn();
    const input = {} as TextInput;
    const { result, rerender } = await renderHook<ReturnType<typeof useTranslatorKeyboard>, { top: number }>(({ top }) => useTranslatorKeyboard(top), { initialProps: { top: 24 } });
    result.current.scrollRef.current = { scrollResponderScrollNativeHandleToKeyboard: scroll } as unknown as ScrollView;
    result.current.inputRef.current = input;
    result.current.onFocus();
    await rerender({ top: 36 });
    result.current.keepInputVisible();
    expect(scroll).toHaveBeenLastCalledWith(input, 52, true);
    result.current.inputRef.current = null;
    result.current.keepInputVisible();
    expect(scroll).toHaveBeenCalledTimes(2);
  });
});
