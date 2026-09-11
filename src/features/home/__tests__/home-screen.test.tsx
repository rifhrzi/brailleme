import { render, screen } from '@testing-library/react-native';

import HomeScreen from '@/features/home/home-screen';
import type { ScanReadinessPort } from '@/services/ports';
import { resetAppPreferences, updateAppPreferences } from '@/storage';

jest.mock('@/services/scan-readiness-port', () => ({
  scanReadinessPort: {
    getAvailability: jest.fn(),
  },
}));

function getAvailabilityMock() {
  return jest.requireMock<{
    scanReadinessPort: {
      getAvailability: jest.MockedFunction<
        ScanReadinessPort['getAvailability']
      >;
    };
  }>('@/services/scan-readiness-port').scanReadinessPort.getAvailability;
}

describe('Beranda readiness pemindai', () => {
  beforeEach(() => {
    resetAppPreferences();
    updateAppPreferences({ onboardingCompleted: true });
  });

  it.each([
    [
      'model production diblokir',
      'model_not_available' as const,
      'Pemindai belum tersedia pada build ini.',
    ],
    [
      'runtime scanner tidak tersedia',
      'scanner_not_available' as const,
      'Komponen pemindai belum tersedia pada perangkat ini.',
    ],
  ])('tidak menampilkan CTA siap saat %s', async (_, reason, message) => {
    getAvailabilityMock().mockResolvedValue({
      available: false,
      reason,
      message,
    });

    await render(<HomeScreen />);

    expect(await screen.findByText('Pemindai belum siap')).toBeOnTheScreen();
    expect(screen.getByText(/Coba buka kembali aplikasi atau periksa pembaruan aplikasi/)).toBeOnTheScreen();
    expect(screen.queryByText(message)).toBeNull();
    expect(
      screen.getByRole('button', { name: 'Lihat status pemindai' }),
    ).toBeOnTheScreen();
    expect(screen.queryByText('Siap memindai Braille')).toBeNull();
    expect(
      screen.queryByRole('button', { name: 'Buka pemindai kata' }),
    ).toBeNull();
    expect(screen.queryByText(/classifier|verifikasi|runtime|build/i)).toBeNull();
  });

  it('menampilkan status dan CTA siap ketika seluruh gate pemindai lulus', async () => {
    getAvailabilityMock().mockResolvedValue({ available: true });

    await render(<HomeScreen />);

    expect(await screen.findByText('Siap memindai Braille')).toBeOnTheScreen();
    expect(
      screen.getByRole('button', { name: 'Buka pemindai kata' }),
    ).toBeOnTheScreen();
    expect(
      screen.queryByRole('button', { name: 'Lihat status pemindai' }),
    ).toBeNull();
  });
});
