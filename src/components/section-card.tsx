import type { PropsWithChildren } from 'react';
import {
  StyleSheet,
  Text,
  View,
  type StyleProp,
  type ViewProps,
  type ViewStyle,
} from 'react-native';

import { colors, radii, spacing, typography } from '@/theme';

export type SectionCardProps = PropsWithChildren<{
  title: string;
  description?: string | undefined;
  variant?: 'plain' | 'tonal' | undefined;
  style?: StyleProp<ViewStyle>;
  testID?: ViewProps['testID'];
}>;

export function SectionCard({
  children,
  description,
  style,
  testID,
  title,
  variant = 'tonal',
}: SectionCardProps) {
  return (
    <View
      accessible={false}
      style={[
        styles.card,
        variant === 'plain' ? styles.plain : styles.tonal,
        style,
      ]}
      testID={testID}
    >
      <View accessible={false} style={styles.headingGroup}>
        <Text
          accessibilityRole="header"
          allowFontScaling
          selectable
          style={styles.title}
        >
          {title}
        </Text>
        {description ? (
          <Text allowFontScaling selectable style={styles.description}>
            {description}
          </Text>
        ) : null}
      </View>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    gap: spacing.md,
    borderColor: colors.outlineVariant,
    borderCurve: 'continuous',
    borderRadius: radii.lg,
    borderWidth: 1,
    padding: spacing.lg,
  },
  tonal: {
    backgroundColor: colors.surfaceContainerLowest,
  },
  plain: {
    backgroundColor: 'transparent',
    borderWidth: 0,
    paddingHorizontal: 0,
  },
  headingGroup: {
    gap: spacing.xs,
  },
  title: {
    ...typography.heading,
    color: colors.text,
    fontSize: 20,
    lineHeight: 28,
  },
  description: {
    ...typography.body,
    color: colors.textSecondary,
  },
});
