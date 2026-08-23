import { render, screen, userEvent } from '@testing-library/react-native';

import DebugRoute from '../../../app/debug';
import LiveRoute, {
  ErrorBoundary as LiveErrorBoundary,
  LiveRouteView,
} from '../../../app/live';
import ModelProbeRoute, {
  ErrorBoundary as ModelProbeErrorBoundary,
} from '../../../app/model-probe';
import {
  resetAlphaPreferences,
  updateAlphaPreferences,
} from '@/storage';
import { minimumTouchTarget } from '@/theme';

const runtimeGlobals = globalThis as typeof globalThis & { __DEV__: boolean };
const originalDev = runtimeGlobals.__DEV__;

describe('guard dan pemulihan route', () => {
  beforeEach(() => {
    runtimeGlobals.__DEV__ = originalDev;
    resetAlphaPreferences();
  });

  afterAll(() => {
    runtimeGlobals.__DEV__ = originalDev;
  });

  it('mengalihkan deep link pemindai ke pengenalan sebelum disetujui', async () => {
    await render(<LiveRoute />);

    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({
      text: '/onboarding',
    });
  });

  it('memerlukan penekanan eksplisit pada setiap mount sebelum runtime kamera dimuat', async () => {
    const user = userEvent.setup();
    updateAlphaPreferences({ onboardingCompleted: true });

    const renderRuntime = jest.fn(() => null);
    const firstMount = await render(
      <LiveRouteView renderRuntime={renderRuntime} />,
    );
    expect(screen.getByTestId('live-activation-gate')).toBeOnTheScreen();
    expect(renderRuntime).not.toHaveBeenCalled();

    const activationButton = screen.getByRole('button', {
      name: 'Aktifkan kamera dan mulai',
    });
    expect(activationButton.props.accessibilityHint).toBe(
      'Membuka kamera dan memulai pemrosesan satu sel di perangkat',
    );
    expect(activationButton).toHaveStyle({
      minHeight: minimumTouchTarget,
      minWidth: minimumTouchTarget,
    });

    await user.press(activationButton);
    expect(renderRuntime).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId('live-activation-gate')).toBeNull();
    await firstMount.unmount();

    renderRuntime.mockClear();
    await render(<LiveRouteView renderRuntime={renderRuntime} />);
    expect(screen.getByTestId('live-activation-gate')).toBeOnTheScreen();
    expect(renderRuntime).not.toHaveBeenCalled();
  });

  it('mengalihkan deep link uji model ke pengenalan sebelum disetujui', async () => {
    await render(<ModelProbeRoute />);

    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({
      text: '/onboarding',
    });
  });

  it('menutup route internal pada build produksi', async () => {
    runtimeGlobals.__DEV__ = false;
    updateAlphaPreferences({ onboardingCompleted: true });

    const modelProbe = await render(<ModelProbeRoute />);
    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({
      text: '/',
    });
    await modelProbe.unmount();

    await render(<DebugRoute />);
    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({
      text: '/settings',
    });
  });

  it('menyediakan pemulihan aman ketika route pemindai gagal', async () => {
    const retry = jest.fn();
    const user = userEvent.setup();
    const routerModule = jest.requireMock<{
      __router: { replace: jest.Mock };
    }>('expo-router');

    await render(<LiveErrorBoundary error={new Error('native')} retry={retry} />);

    expect(
      screen.getByRole('alert', {
        name: /Tidak ada gambar atau hasil yang dibuat/i,
      }),
    ).toBeOnTheScreen();
    await user.press(screen.getByRole('button', { name: 'Coba lagi' }));
    expect(retry).toHaveBeenCalledTimes(1);
    await user.press(
      screen.getByRole('button', { name: 'Kembali ke Beranda' }),
    );
    expect(routerModule.__router.replace).toHaveBeenCalledWith('/');
  });

  it('menyediakan pemulihan aman ketika route uji model gagal', async () => {
    const retry = jest.fn();
    const user = userEvent.setup();

    await render(
      <ModelProbeErrorBoundary error={new Error('native')} retry={retry} />,
    );

    expect(
      screen.getByRole('header', { name: 'Uji model tidak dapat dibuka' }),
    ).toBeOnTheScreen();
    await user.press(screen.getByRole('button', { name: 'Coba lagi' }));
    expect(retry).toHaveBeenCalledTimes(1);
  });
});
