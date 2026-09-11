import type { PropsWithChildren, Ref } from 'react';
import {
  ScrollView,
  type ScrollViewProps,
  StyleSheet,
  useWindowDimensions,
} from 'react-native';

import { colors, spacing } from '@/theme';

export type ScreenProps = PropsWithChildren<
  Omit<ScrollViewProps, 'contentInsetAdjustmentBehavior'> & {
    scrollRef?: Ref<ScrollView>;
  }
>;

export function Screen({
  children,
  contentContainerStyle,
  keyboardShouldPersistTaps = 'handled',
  scrollRef,
  style,
  ...scrollViewProps
}: ScreenProps) {
  const { width } = useWindowDimensions();
  const horizontalPadding = width >= 768 ? 48 : 20;

  return (
    <ScrollView
      {...scrollViewProps}
      ref={scrollRef}
      contentContainerStyle={[
        styles.content,
        { paddingHorizontal: horizontalPadding },
        contentContainerStyle,
      ]}
      contentInsetAdjustmentBehavior="automatic"
      keyboardShouldPersistTaps={keyboardShouldPersistTaps}
      style={[styles.container, style]}
    >
      {children}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.canvas,
  },
  content: {
    flexGrow: 1,
    width: '100%',
    maxWidth: 760,
    alignSelf: 'center',
    gap: spacing.lg,
    paddingBottom: spacing.xxl,
    paddingTop: spacing.lg,
  },
});
