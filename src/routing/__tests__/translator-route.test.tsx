import { render, screen } from '@testing-library/react-native';

import TranslatorRoute from '../../../app/(tabs)/translator';
import { resetAppPreferences, updateAppPreferences } from '@/storage';

jest.mock('@/features/translator/translator-screen', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  const { Text } = jest.requireActual<typeof import('react-native')>('react-native');
  return { TranslatorScreen: () => React.createElement(Text, null, 'Penerjemah siap') };
});

describe('route penerjemah', () => {
  beforeEach(() => resetAppPreferences());

  it('mengarahkan deep link pertama ke pengenalan', async () => {
    await render(<TranslatorRoute />);
    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({ text: '/onboarding' });
    expect(screen.queryByText('Penerjemah siap')).toBeNull();
  });

  it('membuka penerjemah setelah pengenalan tanpa aktivasi kamera', async () => {
    updateAppPreferences({ onboardingCompleted: true });
    await render(<TranslatorRoute />);
    expect(screen.getByTestId('primary-page-translator')).toBeOnTheScreen();
    expect(screen.getByText('Penerjemah siap')).toBeOnTheScreen();
    expect(screen.queryByTestId('live-activation-gate')).toBeNull();
  });
});
