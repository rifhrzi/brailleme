import type { PropsWithChildren } from 'react';
import { StyleSheet, View } from 'react-native';

import { colors } from '@/theme';

export type PrimaryPageName = 'home' | 'scan' | 'translator' | 'settings';

export type PrimaryPageProps = PropsWithChildren<{
  page: PrimaryPageName;
}>;

export function PrimaryPage({ children, page }: PrimaryPageProps) {
  return (
    <View style={styles.page} testID={`primary-page-${page}`}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  page: {
    flex: 1,
    backgroundColor: colors.canvas,
  },
});
