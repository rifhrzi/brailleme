import { render, screen, userEvent } from '@testing-library/react-native';

import LiveRuntimeScreen from '@/features/live/live-runtime-screen';
import {
  INITIAL_WORD_DRAFT_STATE,
  wordDraftReducer,
} from '@/features/live/word-draft';

const mockModel = { id: 'validated-model' };
const mockThresholds = { acceptScore: 0.9 };
const mockAcquireModel = jest.fn(() => jest.fn());
const mockRetry = jest.fn();
let mockModelState: Record<string, unknown> = { status: 'loading' };
let mockCameraProps: Record<string, unknown> | undefined;
const mockLiveScreenView = jest.fn();

jest.mock('@/ml/word-model-loader', () => ({
  useWordModel: () => mockModelState,
}));

jest.mock('@/services', () => ({
  cameraPermissionPort: { id: 'camera-permission-port' },
  scannerPort: { id: 'scanner-port' },
}));

jest.mock('@/features/live/word-photo-camera', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  const { View } =
    jest.requireActual<typeof import('react-native')>('react-native');

  return function MockWordPhotoCamera(props: Record<string, unknown>) {
    mockCameraProps = props;
    return React.createElement(View, { testID: 'word-photo-camera' });
  };
});

jest.mock('@/features/live/word-scan-screen', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  const { View } =
    jest.requireActual<typeof import('react-native')>('react-native');

  return function WordScanScreen(props: Record<string, unknown>) {
    mockLiveScreenView(props);
    const renderScannerPreview = props.renderCamera as
      | ((mockPreviewProps: Record<string, unknown>) => React.ReactNode)
      | undefined;
    return React.createElement(
      View,
      { testID: 'live-screen-view' },
      renderScannerPreview?.({
        active: true,
        sessionId: 'runtime-session',
        torchEnabled: true,
        guideAspectRatio: 4,
      }),
    );
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
    expect(screen.queryByTestId('word-photo-camera')).toBeNull();
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
    expect(screen.queryByTestId('word-photo-camera')).toBeNull();

    await user.press(screen.getByRole('button', { name: 'Coba lagi' }));
    await user.press(screen.getByRole('button', { name: 'Bantuan' }));
    await user.press(screen.getByRole('button', { name: 'Kembali' }));

    expect(mockRetry).toHaveBeenCalledTimes(1);
    expect(router.push).toHaveBeenCalledWith('/help');
    expect(router.back).toHaveBeenCalledTimes(1);
  });

  it('menghubungkan model kata, sesi, lease, dan torch ke kamera setelah siap', async () => {
    mockModelState = {
      status: 'ready',
      model: mockModel,
      thresholds: mockThresholds,
      modelVersion: 'word-test',
      acquireModel: mockAcquireModel,
    };
    const onWordDraftChange = jest.fn();
    const wordDraft = wordDraftReducer(INITIAL_WORD_DRAFT_STATE, {
      type: 'commit_letter',
      evidenceKey: 'runtime-session:0:A',
      letter: 'A',
    });
    const services = jest.requireMock<{
      cameraPermissionPort: { id: string };
      scannerPort: { id: string };
    }>('@/services');

    await render(
      <LiveRuntimeScreen
        onWordDraftChange={onWordDraftChange}
        wordDraft={wordDraft}
      />,
    );

    expect(screen.getByTestId('live-screen-view')).toBeOnTheScreen();
    expect(mockLiveScreenView).toHaveBeenLastCalledWith(
      expect.objectContaining({
        cameraPermission: services.cameraPermissionPort,
        onWordDraftChange,
        wordDraft,
      }),
    );
    expect(mockCameraProps).toEqual({
      active: true,
      model: mockModel,
      thresholds: mockThresholds,
      modelVersion: 'word-test',
      acquireModel: mockAcquireModel,
      sessionId: 'runtime-session',
      torchEnabled: true,
      guideAspectRatio: 4,
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
