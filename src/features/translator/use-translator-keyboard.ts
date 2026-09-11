import { useCallback, useEffect, useRef } from 'react';
import { Keyboard, type ScrollView, type TextInput } from 'react-native';

import { spacing } from '@/theme';

export function useTranslatorKeyboard(topInset: number) {
  const scrollRef = useRef<ScrollView>(null);
  const inputRef = useRef<TextInput>(null);
  const focused = useRef(false);

  const keepInputVisible = useCallback(() => {
    if (!focused.current || !Keyboard.isVisible() || !inputRef.current) return;
    // The native responder measures the input after layout. Its keyboard
    // coordinates are screen-relative, so include the tab's top safe area.
    scrollRef.current?.scrollResponderScrollNativeHandleToKeyboard(
      inputRef.current,
      spacing.md + topInset,
      true,
    );
  }, [topInset]);

  useEffect(() => {
    const subscription = Keyboard.addListener('keyboardDidShow', keepInputVisible);
    return () => subscription.remove();
  }, [keepInputVisible]);

  useEffect(() => () => { focused.current = false; }, []);

  return {
    scrollRef,
    inputRef,
    keepInputVisible,
    onFocus: () => {
      focused.current = true;
      keepInputVisible();
    },
    onBlur: () => { focused.current = false; },
  };
}
