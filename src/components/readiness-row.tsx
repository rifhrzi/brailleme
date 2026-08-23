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
  radii,
  spacing,
  typography,
} from '@/theme';

export type ReadinessTone = 'success' | 'warning' | 'error' | 'neutral';

export type ReadinessRowProps = {
  label: string;
  value: string;
  detail?: string | undefined;
  tone?: ReadinessTone | undefined;
  style?: StyleProp<ViewStyle>;
  testID?: string;
};

export function ReadinessRow({
  detail,
  label,
  style,
  testID,
  tone = 'neutral',
  value,
}: ReadinessRowProps) {
  const spokenLabel = detail
    ? `${label}. ${value}. ${detail}`
    : `${label}. ${value}`;

  return (
    <View
      accessibilityLabel={spokenLabel}
      accessibilityRole="text"
      accessible
      style={[styles.row, style]}
      testID={testID}
    >
      <Text
        accessible={false}
        allowFontScaling
        importantForAccessibility="no-hide-descendants"
        style={styles.label}
      >
        {label}
      </Text>
      <View
        accessible={false}
        style={[styles.badge, { backgroundColor: toneSurfaces[tone] }]}
      >
        <Text
          accessible={false}
          allowFontScaling
          importantForAccessibility="no-hide-descendants"
          style={[styles.value, { color: toneColors[tone] }]}
        >
          {value}
        </Text>
      </View>
      {detail ? (
        <Text
          accessible={false}
          allowFontScaling
          importantForAccessibility="no-hide-descendants"
          style={styles.detail}
        >
          {detail}
        </Text>
      ) : null}
    </View>
  );
}

const toneColors: Record<ReadinessTone, string> = {
  success: colors.success,
  warning: colors.warning,
  error: colors.error,
  neutral: colors.textSecondary,
};

const toneSurfaces: Record<ReadinessTone, string> = {
  success: colors.successContainer,
  warning: colors.warningContainer,
  error: colors.errorContainer,
  neutral: colors.surfaceContainerHighest,
};

const styles = StyleSheet.create({
  row: {
    minHeight: minimumTouchTarget,
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
  },
  label: {
    ...typography.body,
    flexBasis: 180,
    flexGrow: 1,
    flexShrink: 1,
    color: colors.text,
    fontWeight: '600',
  },
  badge: {
    maxWidth: '100%',
    flexShrink: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.sm,
    borderRadius: radii.pill,
  },
  value: {
    ...typography.labelCaps,
    flexShrink: 1,
    fontWeight: '700',
  },
  detail: {
    ...typography.supporting,
    width: '100%',
    color: colors.textSecondary,
  },
});
