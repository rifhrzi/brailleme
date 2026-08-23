import {
  render,
  screen,
  userEvent,
} from '@testing-library/react-native';

import HelpScreen from '@/features/help/help-screen';
import HomeScreen from '@/features/home/home-screen';
import OnboardingScreen from '@/features/onboarding/onboarding-screen';
import {
  ALPHA_PREFERENCES_STORAGE_KEY,
  getAlphaPreferences,
  resetAlphaPreferences,
  updateAlphaPreferences,
} from '@/storage';

type RouterMock = {
  back: jest.Mock;
  canGoBack: jest.Mock;
  push: jest.Mock;
  replace: jest.Mock;
};

type ExpoRouterMock = {
  __router: RouterMock;
  __setLocalSearchParams(params: Record<string, string | undefined>): void;
};

function getExpoRouterMock() {
  return jest.requireMock<ExpoRouterMock>('expo-router');
}

describe('alur onboarding', () => {
  beforeEach(() => {
    resetAlphaPreferences();
  });

  it('mengalihkan first-run dari Beranda ke onboarding', async () => {
    await render(<HomeScreen />);

    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({
      text: '/onboarding',
    });
    expect(screen.queryByTestId('home-screen')).toBeNull();
  });

  it('menyelesaikan onboarding dan mempersistensikan pilihan', async () => {
    const user = userEvent.setup();
    const { __router } = getExpoRouterMock();

    await render(<OnboardingScreen />);
    await user.press(
      screen.getByRole('button', { name: 'Buka Beranda' }),
    );

    expect(getAlphaPreferences().onboardingCompleted).toBe(true);
    expect(__router.replace).toHaveBeenCalledWith('/');
    expect(
      JSON.parse(
        globalThis.localStorage.getItem(ALPHA_PREFERENCES_STORAGE_KEY) ?? '{}',
      ),
    ).toMatchObject({ onboardingCompleted: true });
  });

  it('membuka kembali onboarding dari Bantuan dalam mode review', async () => {
    const user = userEvent.setup();
    const { __router } = getExpoRouterMock();

    updateAlphaPreferences({ onboardingCompleted: true });
    await render(<HelpScreen />);
    await user.press(
      screen.getByRole('button', { name: 'Buka kembali pengenalan' }),
    );

    expect(__router.push).toHaveBeenCalledWith({
      pathname: '/onboarding',
      params: { review: '1' },
    });
  });

  it('menutup mode review dengan kembali', async () => {
    const user = userEvent.setup();
    const routerModule = getExpoRouterMock();
    routerModule.__setLocalSearchParams({ review: '1' });

    updateAlphaPreferences({ onboardingCompleted: true });
    await render(<OnboardingScreen />);

    await user.press(screen.getByRole('button', { name: 'Selesai' }));

    expect(routerModule.__router.back).toHaveBeenCalledTimes(1);
    expect(routerModule.__router.replace).not.toHaveBeenCalled();
  });

  it('menjaga onboarding bebas dari istilah implementasi', async () => {
    await render(<OnboardingScreen />);

    for (const implementationTerm of [
      /technical alpha/i,
      /frontend/i,
      /milestone/i,
      /adapter/i,
      /id-latin-6dot/i,
    ]) {
      expect(screen.queryByText(implementationTerm)).toBeNull();
    }
  });

  it('menjaga Bantuan bebas dari istilah implementasi', async () => {
    await render(<HelpScreen />);

    expect(
      screen.getByRole('header', { name: 'Mulai dari Pindai Satu Sel' }),
    ).toBeOnTheScreen();
    expect(screen.queryByText(/proofreader/i)).toBeNull();
    for (const implementationTerm of [
      /technical alpha/i,
      /frontend/i,
      /milestone/i,
      /adapter/i,
      /id-latin-6dot/i,
    ]) {
      expect(screen.queryByText(implementationTerm)).toBeNull();
    }
  });

  it('menampilkan hanya tiga tujuan utama Beranda setelah onboarding', async () => {
    updateAlphaPreferences({ onboardingCompleted: true });
    await render(<HomeScreen />);

    expect(screen.getByText(/Periksa Braille/i)).toBeOnTheScreen();
    expect(
      await screen.findByRole('button', { name: 'Pindai satu sel' }),
    ).toBeOnTheScreen();
    expect(
      screen.getByRole('header', { name: 'Pemindai satu sel dikonfigurasi' }),
    ).toBeOnTheScreen();
    expect(
      screen.getByText(/Kamera dan model diperiksa ketika layar pemindai dibuka/i),
    ).toBeOnTheScreen();
    expect(screen.getByRole('button', { name: 'Pengaturan' })).toBeOnTheScreen();
    expect(screen.getByRole('button', { name: 'Bantuan' })).toBeOnTheScreen();
    expect(screen.queryByRole('button', { name: /Uji model/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /Dokumen/i })).toBeNull();
    expect(screen.queryByText('Kesiapan offline')).toBeNull();
    expect(screen.queryByText('Model pemindai')).toBeNull();
    expect(screen.queryByText(/technical alpha/i)).toBeNull();
    expect(screen.queryByText(/frontend/i)).toBeNull();
  });
});
