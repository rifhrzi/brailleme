import { render, screen, userEvent } from '@testing-library/react-native';

import LiveRuntimeScreen from '@/features/live/live-runtime-screen';

const mockModel = { id: 'validated-model' };
const mockRetry = jest.fn();
let mockModelState: Record<string, unknown> = { status: 'loading' };
let mockCameraProps: Record<string, unknown> | undefined;
const mockLiveScreenView = jest.fn();

jest.mock('@/ml', () => ({
  useBrailleModel: () => mockModelState,
}));

jest.mock('@/services', () => ({
  cameraPermissionPort: { id: 'camera-permission-port' },
  scannerPort: { id: 'scanner-port' },
}));

jest.mock('@/features/live/live-single-cell-camera', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  const { View } = jest.requireActual<typeof import('react-native')>(
    'react-native',
  );

  return function MockLiveSingleCellCamera(props: Record<string, unknown>) {
    mockCameraProps = props;
    return React.createElement(View, { testID: 'single-cell-camera' });
  };
});

jest.mock('@/features/live/live-screen', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  const { View } = jest.requireActual<typeof import('react-native')>(
    'react-native',
  );

  return {
    LiveScreenView(props: Record<string, unknown>) {
      mockLiveScreenView(props);
      const renderScannerPreview = props.renderScannerPreview as
        | ((mockPreviewProps: Record<string, unknown>) => React.ReactNode)
        | undefined;
      return React.createElement(
        View,
        { testID: 'live-screen-view' },
        renderScannerPreview?.({
          active: true,
          resetToken: 7,
          sessionId: 'runtime-session',
          torchEnabled: true,
        }),
      );
    },
  };
});

describe('LiveRuntimeScreen', () => {
  beforeEach(() => {
    mockModelState = { status: 'loading' };
    mockCameraProps = undefined;
  });

  it('tidak menyalakan kamera sebelum model tervalidasi', async () => {
    await render(<LiveRuntimeScreen />);

    expect(screen.getByTestId('live-model-loading')).toBeOnTheScreen();
    expect(screen.queryByTestId('live-screen-view')).toBeNull();
    expect(screen.queryByTestId('single-cell-camera')).toBeNull();
  });

  it('menampilkan pemulihan ramah pengguna tanpa membocorkan error teknis', async () => {
    const user = userEvent.setup();
    mockModelState = {
      status: 'error',
      error: new Error('Self-test Softmax tensor model gagal.'),
      retry: mockRetry,
    };
    const router = jest.requireMock<{
      __router: { back: jest.Mock; push: jest.Mock };
    }>('expo-router').__router;

    await render(<LiveRuntimeScreen />);

    expect(screen.getByTestId('live-model-error')).toBeOnTheScreen();
    expect(
      screen.getByText(/Pemindai belum dapat disiapkan.*buka Bantuan\./s),
    ).toBeOnTheScreen();
    expect(screen.queryByText(/Softmax|tensor/i)).toBeNull();
    expect(screen.queryByTestId('single-cell-camera')).toBeNull();

    await user.press(screen.getByRole('button', { name: 'Coba lagi' }));
    await user.press(screen.getByRole('button', { name: 'Bantuan' }));
    await user.press(screen.getByRole('button', { name: 'Kembali' }));

    expect(mockRetry).toHaveBeenCalledTimes(1);
    expect(router.push).toHaveBeenCalledWith('/help');
    expect(router.back).toHaveBeenCalledTimes(1);
  });

  it('menghubungkan model, sesi, reset, dan torch ke kamera setelah siap', async () => {
    mockModelState = { status: 'ready', model: mockModel };
    const services = jest.requireMock<{
      cameraPermissionPort: { id: string };
      scannerPort: { id: string };
    }>('@/services');

    await render(<LiveRuntimeScreen />);

    expect(screen.getByTestId('live-screen-view')).toBeOnTheScreen();
    expect(mockLiveScreenView).toHaveBeenLastCalledWith(
      expect.objectContaining({
        cameraPermission: services.cameraPermissionPort,
        scanner: services.scannerPort,
      }),
    );
    expect(mockCameraProps).toEqual({
      active: true,
      model: mockModel,
      resetToken: 7,
      scanner: services.scannerPort,
      sessionId: 'runtime-session',
      torchEnabled: true,
    });
  });

  it('kembali ke Beranda ketika rute dibuka tanpa history', async () => {
    const user = userEvent.setup();
    mockModelState = {
      status: 'error',
      error: new Error('model unavailable'),
      retry: mockRetry,
    };
    const router = jest.requireMock<{
      __router: {
        back: jest.Mock;
        canGoBack: jest.Mock;
        replace: jest.Mock;
      };
    }>('expo-router').__router;
    router.canGoBack.mockReturnValue(false);

    await render(<LiveRuntimeScreen />);
    await user.press(screen.getByRole('button', { name: 'Kembali' }));

    expect(router.back).not.toHaveBeenCalled();
    expect(router.replace).toHaveBeenCalledWith('/');
  });
});
