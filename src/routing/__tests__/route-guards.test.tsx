import { act, render, screen, userEvent } from '@testing-library/react-native';
import { useEffect } from 'react';
import { Text } from 'react-native';

import LiveRoute, {
  ErrorBoundary as LiveErrorBoundary,
  LiveRouteView,
  type LiveRuntimeRenderProps,
} from '../../../app/(tabs)/live';
import {
  INITIAL_WORD_DRAFT_STATE,
  getWordText,
  wordDraftReducer,
} from '@/features/live/word-draft';
import {
  resetAppPreferences,
  updateAppPreferences,
} from '@/storage';
import { minimumTouchTarget } from '@/theme';

describe('guard dan pemulihan route', () => {
  beforeEach(() => {
    resetAppPreferences();
  });

  it('mengalihkan deep link pemindai ke pengenalan sebelum disetujui', async () => {
    await render(<LiveRoute />);

    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({
      text: '/onboarding',
    });
  });

  it('memerlukan penekanan eksplisit pada setiap mount sebelum runtime kamera dimuat', async () => {
    const user = userEvent.setup();
    updateAppPreferences({ onboardingCompleted: true });

    const renderRuntime = jest.fn(() => null);
    const firstMount = await render(
      <LiveRouteView renderRuntime={renderRuntime} />,
    );
    expect(screen.getByTestId('live-activation-gate')).toBeOnTheScreen();
    expect(renderRuntime).not.toHaveBeenCalled();

    const activationButton = screen.getByRole('button', {
      name: 'Aktifkan kamera',
    });
    expect(activationButton.props.accessibilityHint).toBe(
      'Membuka pratinjau kamera untuk memindai satu kata atau rangkaian angka',
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

  it('menutup runtime dan meminta aktivasi ulang ketika tab Pindai kehilangan fokus', async () => {
    const user = userEvent.setup();
    const runtimeUnmounted = jest.fn();
    const routerModule = jest.requireMock<{
      __setRouteFocused(focused: boolean): void;
    }>('expo-router');
    updateAppPreferences({ onboardingCompleted: true });

    function RuntimeProbe() {
      useEffect(() => runtimeUnmounted, []);
      return <Text>Runtime kamera aktif</Text>;
    }

    await render(<LiveRouteView renderRuntime={() => <RuntimeProbe />} />);
    await user.press(
      screen.getByRole('button', { name: 'Aktifkan kamera' }),
    );
    expect(screen.getByText('Runtime kamera aktif')).toBeOnTheScreen();

    await act(async () => {
      routerModule.__setRouteFocused(false);
      await Promise.resolve();
    });

    expect(runtimeUnmounted).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('live-activation-gate')).toBeOnTheScreen();
    expect(screen.queryByText('Runtime kamera aktif')).toBeNull();

    await act(async () => {
      routerModule.__setRouteFocused(true);
      await Promise.resolve();
    });
    expect(screen.getByTestId('live-activation-gate')).toBeOnTheScreen();
  });

  it('menghapus hasil kata ketika runtime kamera ditutup karena blur', async () => {
    const user = userEvent.setup();
    const routerModule = jest.requireMock<{
      __setRouteFocused(focused: boolean): void;
    }>('expo-router');
    let latestRuntimeProps: LiveRuntimeRenderProps | undefined;
    updateAppPreferences({ onboardingCompleted: true });

    const renderRuntime = jest.fn((props: LiveRuntimeRenderProps) => {
      latestRuntimeProps = props;
      return <Text>Kata sementara: {getWordText(props.wordDraft) || 'kosong'}</Text>;
    });

    await render(<LiveRouteView renderRuntime={renderRuntime} />);
    await user.press(
      screen.getByRole('button', { name: 'Aktifkan kamera' }),
    );

    const draftA = wordDraftReducer(INITIAL_WORD_DRAFT_STATE, {
      type: 'commit_letter',
      evidenceKey: 'session-1:epoch-0:A',
      letter: 'A',
    });
    await act(() => {
      if (!latestRuntimeProps) {
        throw new Error('Runtime belum dirender.');
      }
      latestRuntimeProps.onWordDraftChange(draftA);
    });
    expect(screen.getByText('Kata sementara: A')).toBeOnTheScreen();

    await act(async () => {
      routerModule.__setRouteFocused(false);
      await Promise.resolve();
    });
    expect(screen.getByTestId('live-activation-gate')).toBeOnTheScreen();
    expect(screen.queryByText('Kata sementara: A')).toBeNull();

    await act(async () => {
      routerModule.__setRouteFocused(true);
      await Promise.resolve();
    });
    await user.press(
      screen.getByRole('button', { name: 'Aktifkan kamera' }),
    );
    expect(screen.getByText('Kata sementara: kosong')).toBeOnTheScreen();
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
        name: /Coba buka kembali pemindai/i,
      }),
    ).toBeOnTheScreen();
    await user.press(screen.getByRole('button', { name: 'Coba lagi' }));
    expect(retry).toHaveBeenCalledTimes(1);
    await user.press(
      screen.getByRole('button', { name: 'Kembali ke Beranda' }),
    );
    expect(routerModule.__router.replace).toHaveBeenCalledWith('/');
  });

});
