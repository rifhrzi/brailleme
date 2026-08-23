import type { PropsWithChildren } from 'react';
import {
  StyleSheet,
  Text,
  View,
  type StyleProp,
  type ViewStyle,
} from 'react-native';

import {
  colors,
  minimumTouchTarget,
  spacing,
  typography,
} from '@/theme';

export type SettingRowProps = PropsWithChildren<{
  label: string;
  description?: string | undefined;
  value?: string | undefined;
  style?: StyleProp<ViewStyle>;
  testID?: string;
}>;

export function SettingRow({
  children,
  description,
  label,
  style,
  testID,
  value,
}: SettingRowProps) {
  const hasControl = children !== undefined && children !== null && children !== false;
  const readOnlyLabel = [label, value, description].filter(Boolean).join('. ');

  return (
    <View
      accessibilityLabel={hasControl ? undefined : readOnlyLabel}
      accessibilityRole={hasControl ? undefined : 'text'}
      accessible={!hasControl}
      style={[styles.row, style]}
      testID={testID}
    >
      <View
        accessibilityLabel={hasControl ? readOnlyLabel : undefined}
        accessibilityRole={hasControl ? 'text' : undefined}
        accessible={hasControl}
        style={styles.copy}
      >
        <Text
          accessible={false}
          allowFontScaling
          importantForAccessibility="no-hide-descendants"
          style={styles.label}
        >
          {label}
        </Text>
        {value ? (
          <Text
            accessible={false}
            allowFontScaling
            importantForAccessibility="no-hide-descendants"
            style={styles.value}
          >
            {value}
          </Text>
        ) : null}
        {description ? (
          <Text
            accessible={false}
            allowFontScaling
            importantForAccessibility="no-hide-descendants"
            style={styles.description}
          >
            {description}
          </Text>
        ) : null}
      </View>
      {hasControl ? (
        <View accessible={false} style={styles.control}>
          {children}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    minHeight: minimumTouchTarget,
    alignItems: 'stretch',
    gap: spacing.md,
    paddingVertical: spacing.md,
  },
  copy: {
    gap: spacing.xs,
    width: '100%',
  },
  control: {
    minHeight: minimumTouchTarget,
    minWidth: minimumTouchTarget,
    alignItems: 'stretch',
    justifyContent: 'center',
    width: '100%',
  },
  label: {
    ...typography.label,
    color: colors.text,
  },
  value: {
    ...typography.body,
    color: colors.text,
  },
  description: {
    ...typography.supporting,
    color: colors.textSecondary,
  },
});
