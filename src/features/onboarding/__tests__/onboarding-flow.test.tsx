import {
  render,
  screen,
  userEvent,
} from '@testing-library/react-native';

import HelpScreen from '@/features/help/help-screen';
import HomeScreen from '@/features/home/home-screen';
import OnboardingScreen from '@/features/onboarding/onboarding-screen';
import {
  APP_PREFERENCES_STORAGE_KEY,
  getAppPreferences,
  resetAppPreferences,
  updateAppPreferences,
} from '@/storage';

jest.mock('@/ml/word-model-assets', () => ({
  WORD_MODEL_BUNDLED: true,
  WORD_MODEL_RELEASE_VERIFIED: false,
}));

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
    resetAppPreferences();
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

    expect(getAppPreferences().onboardingCompleted).toBe(true);
    expect(__router.replace).toHaveBeenCalledWith('/');
    expect(
      JSON.parse(
        globalThis.localStorage.getItem(APP_PREFERENCES_STORAGE_KEY) ?? '{}',
      ),
    ).toMatchObject({ onboardingCompleted: true });
  });

  it('membatasi lebar aksi footer pada layar besar', async () => {
    await render(<OnboardingScreen />);

    expect(screen.getByTestId('onboarding-footer-content')).toHaveStyle({
      width: '100%',
      maxWidth: 760,
      alignSelf: 'center',
    });
  });

  it('membuka kembali onboarding dari Bantuan dalam mode review', async () => {
    const user = userEvent.setup();
    const { __router } = getExpoRouterMock();

    updateAppPreferences({ onboardingCompleted: true });
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

    updateAppPreferences({ onboardingCompleted: true });
    await render(<OnboardingScreen />);

    await user.press(screen.getByRole('button', { name: 'Selesai' }));

    expect(routerModule.__router.back).toHaveBeenCalledTimes(1);
    expect(routerModule.__router.replace).not.toHaveBeenCalled();
  });

  it('menjaga onboarding bebas dari istilah implementasi', async () => {
    await render(<OnboardingScreen />);

    for (const implementationTerm of [
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
      screen.getByRole('header', { name: 'Pindai satu blok kata' }),
    ).toBeOnTheScreen();
    expect(screen.queryByText(/proofreader/i)).toBeNull();
    for (const implementationTerm of [
      /frontend/i,
      /milestone/i,
      /adapter/i,
      /id-latin-6dot/i,
    ]) {
      expect(screen.queryByText(implementationTerm)).toBeNull();
    }
  });

  it('menampilkan hanya tiga tujuan utama Beranda setelah onboarding', async () => {
    updateAppPreferences({ onboardingCompleted: true });
    await render(<HomeScreen />);

    expect(screen.getByText(/Pindai satu kata Braille/i)).toBeOnTheScreen();
    expect(
      await screen.findByRole('button', { name: 'Buka pemindai kata' }),
    ).toBeOnTheScreen();
    expect(screen.getByText('Siap memindai kata')).toBeOnTheScreen();
    expect(
      screen.getByText(/Tempatkan satu kata di dalam bingkai/i),
    ).toBeOnTheScreen();
    expect(screen.getByRole('button', { name: 'Bantuan' })).toBeOnTheScreen();
    expect(screen.getByRole('button', { name: 'Privasi' })).toBeOnTheScreen();
    expect(screen.queryByRole('button', { name: /Uji model/i })).toBeNull();
    expect(screen.queryByRole('button', { name: /Dokumen/i })).toBeNull();
    expect(screen.queryByText('Kesiapan offline')).toBeNull();
    expect(screen.queryByText('Model pemindai')).toBeNull();
    expect(screen.queryByText(/frontend/i)).toBeNull();
  });
});
