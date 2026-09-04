import { fireEvent, render, screen } from '@testing-library/react-native';

import { AccessibleButton } from '@/components/accessible-button';
import { colors, minimumTouchTarget } from '@/theme';

describe('AccessibleButton', () => {
  it('mempertahankan target sentuh, border kontras, dan fokus yang terlihat', async () => {
    const onBlur = jest.fn();
    const onFocus = jest.fn();

    await render(
      <AccessibleButton
        label="Bantuan"
        onBlur={onBlur}
        onFocus={onFocus}
        onPress={jest.fn()}
        variant="navigation"
      />,
    );
    const button = screen.getByRole('button', { name: 'Bantuan' });

    expect(button).toHaveStyle({
      borderColor: colors.outline,
      minHeight: minimumTouchTarget,
      minWidth: minimumTouchTarget,
    });

    await fireEvent(button, 'focus');
    expect(onFocus).toHaveBeenCalledTimes(1);
    expect(button).toHaveStyle({
      borderColor: colors.text,
      borderWidth: 3,
    });

    await fireEvent(button, 'blur');
    expect(onBlur).toHaveBeenCalledTimes(1);
    expect(button).toHaveStyle({ borderColor: colors.outline });
  });

  it('menggabungkan state pemanggil dengan state disabled yang sebenarnya', async () => {
    await render(
      <AccessibleButton
        accessibilityState={{ busy: true, disabled: false, expanded: true }}
        disabled
        label="Memuat"
        onPress={jest.fn()}
      />,
    );

    expect(
      screen.getByRole('button', { name: 'Memuat' }).props.accessibilityState,
    ).toEqual({
      busy: true,
      disabled: true,
      expanded: true,
    });
  });
});
