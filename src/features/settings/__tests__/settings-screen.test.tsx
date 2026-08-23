import {
  fireEvent,
  render,
  screen,
  userEvent,
} from '@testing-library/react-native';
import { Alert, AppState } from 'react-native';

import SettingsScreen from '@/features/settings/settings-screen';
import { ttsPort } from '@/services';
import {
  ALPHA_PREFERENCES_STORAGE_KEY,
  getAlphaPreferences,
  resetAlphaPreferences,
  updateAlphaPreferences,
} from '@/storage';

const runtimeGlobals = globalThis as typeof globalThis & { __DEV__: boolean };
const originalDev = runtimeGlobals.__DEV__;

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, reject, resolve };
}

describe('Pengaturan', () => {
  beforeEach(() => {
    runtimeGlobals.__DEV__ = originalDev;
    resetAlphaPreferences();
  });

  afterAll(() => {
    runtimeGlobals.__DEV__ = originalDev;
  });

  it('menampilkan default yang bersih dan semantik kontrol', async () => {
    await render(<SettingsScreen />);

    expect(
      screen.getByText(/Atur cara BrailleMe membacakan hasil/i),
    ).toBeOnTheScreen();
    expect(
      screen.getByTestId('speech-mode'),
    ).toHaveAccessibleName(
      'Mode membaca. Tekan untuk membaca. Hasil hanya dibaca setelah pengguna menekan Bacakan.',
    );

    expect(
      screen.getByRole('adjustable', { name: 'Kecepatan suara' }),
    ).toHaveAccessibilityValue({ min: 0.5, max: 2, now: 1, text: '1.0 kali' });
    expect(
      screen.getByRole('adjustable', { name: 'Volume suara' }),
    ).toHaveAccessibilityValue({ min: 0, max: 100, now: 100, text: '100 persen' });
    expect(
      screen.getByRole('switch', { name: 'Bisukan suara' }),
    ).not.toBeChecked();
    expect(
      screen.getByRole('switch', { name: 'Umpan balik haptik' }),
    ).toBeChecked();
    expect(
      screen.getByRole('switch', { name: 'Aktifkan menu diagnostik' }),
    ).not.toBeChecked();
    expect(
      screen.queryByRole('button', { name: 'Buka menu diagnostik' }),
    ).toBeNull();

    expect(screen.queryByRole('adjustable', { name: /Profil/i })).toBeNull();
    expect(screen.queryByText('id-latin-6dot-alpha-v0')).toBeNull();
    expect(screen.queryByText(/technical alpha/i)).toBeNull();
    expect(screen.queryByText(/frontend/i)).toBeNull();
    expect(screen.queryByText(/model pemindai/i)).toBeNull();
    expect(screen.queryByText(/auto[- ]?speak/i)).toBeNull();
    expect(screen.queryByText(/baca otomatis/i)).toBeNull();

    for (const forbiddenLabel of [
      'Mode Dokumen',
      'Riwayat',
      'Simpan hasil',
      'Bagikan',
      'Ekspor',
    ]) {
      expect(screen.queryByText(forbiddenLabel, { exact: true })).toBeNull();
      expect(
        screen.queryByRole('button', { name: forbiddenLabel }),
      ).toBeNull();
    }
  });

  it('mengaktifkan menu diagnostik, mempersistensikannya, dan membuka route debug', async () => {
    const user = userEvent.setup();
    const routerModule = jest.requireMock<{
      __router: { push: jest.Mock };
    }>('expo-router');
    await render(<SettingsScreen />);

    await fireEvent(
      screen.getByRole('switch', { name: 'Aktifkan menu diagnostik' }),
      'valueChange',
      true,
    );

    expect(getAlphaPreferences().debugMenuEnabled).toBe(true);
    expect(
      JSON.parse(
        globalThis.localStorage.getItem(ALPHA_PREFERENCES_STORAGE_KEY) ?? '{}',
      ),
    ).toMatchObject({ debugMenuEnabled: true });

    await user.press(
      screen.getByRole('button', { name: 'Buka menu diagnostik' }),
    );
    expect(routerModule.__router.push).toHaveBeenCalledWith('/debug');
  });

  it('menyembunyikan menu diagnostik pada build produksi', async () => {
    runtimeGlobals.__DEV__ = false;
    updateAlphaPreferences({ debugMenuEnabled: true });

    await render(<SettingsScreen />);

    expect(screen.queryByText('Diagnostik')).toBeNull();
    expect(
      screen.queryByRole('switch', { name: 'Aktifkan menu diagnostik' }),
    ).toBeNull();
    expect(
      screen.queryByRole('button', { name: 'Buka menu diagnostik' }),
    ).toBeNull();
  });

  it('mengubah rate, volume, mute, dan haptik lalu mempersistensikannya', async () => {
    const user = userEvent.setup();
    await render(<SettingsScreen />);

    await user.press(
      screen.getByRole('button', { name: 'Tambah kecepatan' }),
    );
    await user.press(
      screen.getByRole('button', { name: 'Kurangi volume' }),
    );
    await fireEvent(
      screen.getByRole('switch', { name: 'Bisukan suara' }),
      'valueChange',
      true,
    );
    await fireEvent(
      screen.getByRole('switch', { name: 'Umpan balik haptik' }),
      'valueChange',
      false,
    );

    expect(
      screen.getByRole('adjustable', { name: 'Kecepatan suara' }),
    ).toHaveAccessibilityValue({ now: 1.1, text: '1.1 kali' });
    expect(
      screen.getByRole('adjustable', { name: 'Volume suara' }),
    ).toHaveAccessibilityValue({ now: 90, text: '90 persen' });
    expect(
      screen.getByRole('switch', { name: 'Bisukan suara' }),
    ).toBeChecked();
    expect(
      screen.getByRole('switch', { name: 'Umpan balik haptik' }),
    ).not.toBeChecked();

    expect(getAlphaPreferences()).toMatchObject({
      speechRate: 1.1,
      speechVolume: 0.9,
      muted: true,
      hapticsEnabled: false,
    });
    expect(
      JSON.parse(
        globalThis.localStorage.getItem(ALPHA_PREFERENCES_STORAGE_KEY) ?? '{}',
      ),
    ).toMatchObject({
      speechRate: 1.1,
      speechVolume: 0.9,
      muted: true,
      hapticsEnabled: false,
    });
  });

  it('membulatkan nilai slider ke langkah yang ditetapkan', async () => {
    await render(<SettingsScreen />);

    await fireEvent(
      screen.getByRole('adjustable', { name: 'Kecepatan suara' }),
      'slidingComplete',
      1.26,
    );
    await fireEvent(
      screen.getByRole('adjustable', { name: 'Volume suara' }),
      'slidingComplete',
      0.54,
    );

    expect(getAlphaPreferences()).toMatchObject({
      speechRate: 1.3,
      speechVolume: 0.5,
    });
  });

  it('mengembalikan semua preferensi ke default', async () => {
    const user = userEvent.setup();
    const alertSpy = jest.spyOn(Alert, 'alert').mockImplementation(
      (_title, _message, buttons) => {
        buttons?.find((button) => button.style === 'destructive')?.onPress?.();
      },
    );
    updateAlphaPreferences({
      onboardingCompleted: true,
      speechRate: 1.8,
      speechVolume: 0.3,
      muted: true,
      hapticsEnabled: false,
      debugMenuEnabled: true,
    });
    await render(<SettingsScreen />);

    await user.press(
      screen.getByRole('button', { name: 'Reset preferensi' }),
    );

    expect(getAlphaPreferences()).toEqual({
      schemaVersion: 2,
      onboardingCompleted: false,
      profile: 'id-latin-6dot-alpha-v0',
      speechMode: 'press',
      speechRate: 1,
      speechVolume: 1,
      muted: false,
      hapticsEnabled: true,
      debugMenuEnabled: false,
    });
    expect(
      screen.getByRole('alert', {
        name: 'Preferensi dikembalikan ke nilai awal.',
      }),
    ).toBeOnTheScreen();
    expect(alertSpy).toHaveBeenCalledWith(
      'Reset semua preferensi?',
      expect.any(String),
      expect.any(Array),
    );
  });

  it('membuka pengaturan TTS Android melalui adapter', async () => {
    const user = userEvent.setup();
    const intentLauncher = jest.requireMock<{
      ActivityAction: { TTS_SETTINGS: string };
      startActivityAsync: jest.Mock;
    }>('expo-intent-launcher');

    await render(<SettingsScreen />);
    await user.press(
      screen.getByRole('button', { name: 'Buka pengaturan suara Android' }),
    );

    expect(intentLauncher.startActivityAsync).toHaveBeenCalledWith(
      intentLauncher.ActivityAction.TTS_SETTINGS,
    );
    expect(
      screen.getByRole('alert', {
        name: 'Pengaturan suara Android dibuka.',
      }),
    ).toBeOnTheScreen();
  });

  it('mempertahankan hasil refresh TTS terbaru saat request lama selesai belakangan', async () => {
    type Availability = Awaited<ReturnType<typeof ttsPort.getAvailability>>;
    const firstRequest = deferred<Availability>();
    const secondRequest = deferred<Availability>();
    let appStateListener: ((state: 'active') => void) | undefined;
    const originalAddEventListener = AppState.addEventListener;
    AppState.addEventListener = jest.fn((_type, listener) => {
      appStateListener = listener as (state: 'active') => void;
      return { remove: jest.fn() };
    });
    const availabilitySpy = jest
      .spyOn(ttsPort, 'getAvailability')
      .mockReturnValueOnce(firstRequest.promise)
      .mockReturnValueOnce(secondRequest.promise);
    const invalidateSpy = jest.spyOn(ttsPort, 'invalidateAvailabilityCache');

    try {
      await render(<SettingsScreen />);
      appStateListener?.('active');

      secondRequest.resolve({
        status: 'available',
        localVoiceAvailable: true,
        message: 'Suara lokal tersedia.',
      });
      expect(
        await screen.findByRole('text', {
          name: /Suara lokal\. Tersedia secara lokal\./,
        }),
      ).toBeOnTheScreen();

      firstRequest.resolve({
        status: 'unavailable',
        localVoiceAvailable: false,
        message: 'Suara lokal tidak tersedia.',
      });
      await Promise.resolve();

      expect(
        screen.getByRole('text', {
          name: /Suara lokal\. Tersedia secara lokal\./,
        }),
      ).toBeOnTheScreen();
      expect(
        screen.queryByRole('text', {
          name: /Suara lokal\. Belum tersedia\./,
        }),
      ).toBeNull();
      expect(availabilitySpy).toHaveBeenCalledTimes(2);
      expect(invalidateSpy).toHaveBeenCalledTimes(1);
    } finally {
      invalidateSpy.mockRestore();
      availabilitySpy.mockRestore();
      AppState.addEventListener = originalAddEventListener;
    }
  });
});
