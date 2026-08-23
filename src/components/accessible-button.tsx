import { useState } from 'react';
import {
  Pressable,
  StyleSheet,
  Text,
  type PressableProps,
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

export type AccessibleButtonVariant =
  | 'action'
  | 'primary'
  | 'navigation'
  | 'secondary'
  | 'danger';

export type AccessibleButtonProps = Omit<
  PressableProps,
  | 'accessibilityLabel'
  | 'accessibilityRole'
  | 'accessibilityState'
  | 'accessible'
  | 'children'
  | 'disabled'
  | 'style'
> & {
  label: string;
  accessibilityLabel?: string | undefined;
  variant?: AccessibleButtonVariant | undefined;
  disabled?: boolean | undefined;
  style?: StyleProp<ViewStyle>;
};

export function AccessibleButton({
  accessibilityLabel,
  disabled = false,
  label,
  onBlur,
  onFocus,
  style,
  variant = 'primary',
  ...pressableProps
}: AccessibleButtonProps) {
  const [focused, setFocused] = useState(false);

  return (
    <Pressable
      {...pressableProps}
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityRole="button"
      accessibilityState={{ disabled }}
      accessible
      disabled={disabled}
      onBlur={(event) => {
        setFocused(false);
        onBlur?.(event);
      }}
      onFocus={(event) => {
        setFocused(true);
        onFocus?.(event);
      }}
      style={({ pressed }) => [
        styles.button,
        buttonStyles[variant],
        variant === 'navigation' ? styles.navigation : undefined,
        disabled ? styles.disabled : undefined,
        focused && !disabled ? styles.focused : undefined,
        pressed && !disabled ? styles.pressed : undefined,
        style,
      ]}
    >
      <Text
        accessible={false}
        allowFontScaling
        style={[
          styles.label,
          labelStyles[variant],
          disabled ? styles.disabledLabel : undefined,
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: minimumTouchTarget,
    minWidth: minimumTouchTarget,
    maxWidth: '100%',
    flexShrink: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderCurve: 'continuous',
    borderRadius: radii.default,
    borderWidth: 2,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  label: {
    ...typography.label,
    flexShrink: 1,
    textAlign: 'center',
  },
  navigation: {
    alignItems: 'flex-start',
  },
  pressed: {
    opacity: 0.82,
    transform: [{ scale: 0.99 }],
  },
  focused: {
    borderColor: colors.text,
    borderWidth: 3,
  },
  disabled: {
    backgroundColor: colors.surfaceContainerHighest,
    borderColor: colors.surfaceContainerHighest,
  },
  disabledLabel: {
    color: colors.textSecondary,
  },
});

const buttonStyles = StyleSheet.create({
  action: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  primary: {
    backgroundColor: colors.inverseSurface,
    borderColor: colors.inverseSurface,
  },
  navigation: {
    backgroundColor: colors.surfaceContainerLowest,
    borderColor: colors.outline,
  },
  secondary: {
    backgroundColor: colors.surfaceContainerLowest,
    borderColor: colors.outline,
  },
  danger: {
    backgroundColor: colors.errorContainer,
    borderColor: colors.error,
  },
});

const labelStyles = StyleSheet.create({
  action: {
    color: colors.onPrimary,
  },
  primary: {
    color: colors.onPrimary,
  },
  navigation: {
    color: colors.text,
    textAlign: 'left',
  },
  secondary: {
    color: colors.text,
  },
  danger: {
    color: colors.onErrorContainer,
  },
});
