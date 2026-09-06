import { act, fireEvent, render, screen } from '@testing-library/react-native';
import { useEffect, useImperativeHandle } from 'react';
import {
  AccessibilityInfo,
  AppState,
  View,
  type AppStateStatus,
} from 'react-native';

import type { WordPhotoPreviewProps } from '@/features/live/word-photo-camera';
import WordScanScreen from '@/features/live/word-scan-screen';
import { WORD_LATIN_MASKS } from '@/ml/word-model-contract';
import type { CameraPermissionPort, TtsPort } from '@/services';
import type { TtsPlaybackListener } from '@/services/ports';
import { resetAppPreferences, updateAppPreferences } from '@/storage';
import type { WordScanEvent } from '@/types/word-scan';

jest.mock('expo-haptics', () => ({
  NotificationFeedbackType: { Success: 'success' },
  notificationAsync: jest.fn(async () => undefined),
}));

let cameraProps: WordPhotoPreviewProps;
const capture = jest.fn();
function CameraHarness(props: WordPhotoPreviewProps) {
  const { onAvailability } = props;
  useEffect(() => {
    cameraProps = props;
  }, [props]);
  useImperativeHandle(props.cameraRef, () => ({ capture }), []);
  useEffect(() => {
    onAvailability(true, true);
  }, [onAvailability]);
  return <View testID="test-word-camera" />;
}
function createSuccess(
  text: string,
  overrides: Partial<WordScanEvent> = {},
): WordScanEvent {
  const cells = [...text].map((letter, index) => ({
    letter,
    mask: WORD_LATIN_MASKS[letter.charCodeAt(0) - 65]!,
    confidence: 0.98,
    box: { x: 0.1 + index * 0.07, y: 0.4, width: 0.05, height: 0.15 },
  }));
  return {
    schemaVersion: 3,
    sessionId: cameraProps.sessionId,
    captureId: 1,
    timestampMs: 1000,
    status: 'success',
    text,
    confidence: 0.98,
    cells,
    boxes: cells.map((cell) => cell.box),
    decoderVersion: 'six-dot-word-v2',
    ...overrides,
  };
}
async function setup(
  permissionOverride?: CameraPermissionPort,
  ttsOverride: Partial<TtsPort> = {},
) {
  const permission: CameraPermissionPort = permissionOverride ?? {
    getStatus: jest.fn(async () => ({
      status: 'granted' as const,
      canAskAgain: true,
    })),
    request: jest.fn(async () => ({
      status: 'granted' as const,
      canAskAgain: true,
    })),
    openAppSettings: jest.fn(async () => ({ ok: true as const })),
  };
  const tts: TtsPort = {
    getAvailability: jest.fn(async () => ({
      status: 'available' as const,
      localVoiceAvailable: true,
      message: 'ready',
    })),
    speak: jest.fn(async () => ({ ok: true as const })),
    stop: jest.fn(async () => ({ ok: true as const })),
    repeat: jest.fn(async () => ({ ok: true as const })),
    openTtsSettings: jest.fn(async () => ({ ok: true as const })),
    ...ttsOverride,
  };
  const onWordDraftChange = jest.fn();
  const view = await render(
    <WordScanScreen
      renderCamera={(props) => (
        <CameraHarness key={props.sessionId} {...props} />
      )}
      cameraPermission={permission}
      tts={tts}
      onWordDraftChange={onWordDraftChange}
    />,
  );
  return { view, tts, permission, onWordDraftChange };
}

describe('WordScanScreen', () => {
  beforeEach(() => {
    capture.mockClear();
    resetAppPreferences();
    Object.defineProperty(AppState, 'currentState', {
      configurable: true,
      value: 'active',
    });
  });

  it('reads repeated letters as a whole word and replaces the previous capture on rescan', async () => {
    const test = await setup();
    expect(capture).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Tambah huruf' })).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    expect(capture).toHaveBeenCalledTimes(1);
    await act(() => cameraProps.onEvent(createSuccess('MAMA')));
    expect(screen.getByText('MAMA')).toBeOnTheScreen();
    expect(test.onWordDraftChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ letters: ['M', 'A', 'M', 'A'] }),
    );
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    expect(test.tts.speak).toHaveBeenCalledWith('MAMA', { rate: 1, volume: 1 });
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai ulang' }));
    expect(screen.queryByText('MAMA')).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    await act(() => cameraProps.onEvent(createSuccess('KAKAK')));
    expect(screen.getByText('KAKAK')).toBeOnTheScreen();
    expect(test.onWordDraftChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ letters: ['K', 'A', 'K', 'A', 'K'] }),
    );
  });

  it('rejects stale and partial results and never enables speech for uncertain captures', async () => {
    await setup();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    const stale = createSuccess('MAMA');
    await act(() =>
      cameraProps.onEvent({ ...stale, boxes: stale.boxes.slice(1) }),
    );
    expect(screen.queryByText('MAMA')).toBeNull();
    await act(() =>
      cameraProps.onEvent({
        schemaVersion: 3,
        sessionId: cameraProps.sessionId,
        captureId: 1,
        timestampMs: 1000,
        status: 'rejected',
        reason: 'low_confidence',
        cells: [],
        boxes: [],
        decoderVersion: 'test',
      }),
    );
    expect(screen.queryByRole('button', { name: 'Bacakan' })).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai ulang' }));
    await act(() => cameraProps.onEvent(stale));
    expect(screen.queryByText('MAMA')).toBeNull();
    expect(screen.getByRole('button', { name: 'Pindai kata' })).toBeEnabled();
  });

  it('does not create the camera with denied permission and offers recovery', async () => {
    const permission: CameraPermissionPort = {
      getStatus: jest.fn(async () => ({
        status: 'blocked' as const,
        canAskAgain: false,
      })),
      request: jest.fn(),
      openAppSettings: jest.fn(async () => ({ ok: true as const })),
    };
    await setup(permission);
    expect(screen.queryByTestId('test-word-camera')).toBeNull();
    await fireEvent.press(
      screen.getByRole('button', { name: 'Buka pengaturan aplikasi' }),
    );
    expect(permission.openAppSettings).toHaveBeenCalledTimes(1);
  });

  it('clears the result and stops speech when the route loses focus', async () => {
    const test = await setup();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    const stale = createSuccess('A');
    await act(() => cameraProps.onEvent(stale));
    const router = jest.requireMock<{
      __setRouteFocused(value: boolean): void;
    }>('expo-router');
    await act(() => router.__setRouteFocused(false));
    expect(screen.queryByTestId('test-word-camera')).toBeNull();
    expect(screen.queryByText('A')).toBeNull();
    expect(test.tts.stop).toHaveBeenCalled();
    await act(() => router.__setRouteFocused(true));
    await act(() => cameraProps.onEvent(stale));
    expect(screen.queryByText('A')).toBeNull();
  });

  it('rechecks permission after backgrounding and rejects the abandoned capture', async () => {
    let onAppState: ((state: AppStateStatus) => void) | undefined;
    const originalAddEventListener = AppState.addEventListener;
    AppState.addEventListener = jest.fn((_event, callback) => {
      onAppState = callback;
      return { remove: jest.fn() };
    });
    try {
      const test = await setup();
      await fireEvent.press(
        screen.getByRole('button', { name: 'Pindai kata' }),
      );
      const abandoned = createSuccess('MAMA');
      await act(() => onAppState?.('background'));
      expect(screen.queryByTestId('test-word-camera')).toBeNull();
      (test.permission.getStatus as jest.Mock).mockResolvedValue({
        status: 'blocked',
        canAskAgain: false,
      });
      await act(async () => onAppState?.('active'));
      await act(() => cameraProps.onEvent(abandoned));
      expect(screen.queryByTestId('test-word-camera')).toBeNull();
      expect(screen.queryByText('MAMA')).toBeNull();
      expect(
        screen.getByRole('button', { name: 'Buka pengaturan aplikasi' }),
      ).toBeOnTheScreen();
      expect(test.tts.stop).toHaveBeenCalled();
    } finally {
      AppState.addEventListener = originalAddEventListener;
    }
  });

  it('disables reading when muted and allows it again after speech is stopped', async () => {
    const test = await setup();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    await act(() => cameraProps.onEvent(createSuccess('MAMA')));
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    await act(() => updateAppPreferences({ muted: true }));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeDisabled();
    await act(() => updateAppPreferences({ muted: false }));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeEnabled();
    expect(test.tts.stop).toHaveBeenCalled();
  });

  it('hides the width control after the shutter and restores the selected width on rescan', async () => {
    await setup();
    const slider = () =>
      screen.getByRole('adjustable', { name: 'Lebar bingkai' });
    expect(cameraProps.guideAspectRatio).toBe(4);
    expect(slider()).toHaveAccessibilityValue({
      min: 1,
      max: 8,
      now: 4,
      text: '50 persen lebar maksimum',
    });
    await fireEvent(slider(), 'valueChange', 6.35);
    expect(cameraProps.guideAspectRatio).toBe(6.35);
    const lateWidthChange = slider().props.onValueChange;
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    expect(
      screen.queryByRole('adjustable', { name: 'Lebar bingkai' }),
    ).toBeNull();
    expect(screen.queryByText('Lebar bingkai')).toBeNull();
    expect(
      screen.queryByText('Sesuaikan bingkai agar hanya memuat satu kata utuh.'),
    ).toBeNull();
    expect(screen.queryByText('Pendek')).toBeNull();
    expect(screen.queryByText('Panjang')).toBeNull();
    await act(() => lateWidthChange(1));
    expect(cameraProps.guideAspectRatio).toBe(6.35);
    await act(() =>
      cameraProps.onEvent({
        schemaVersion: 3,
        sessionId: cameraProps.sessionId,
        captureId: 1,
        timestampMs: 900,
        status: 'processing',
        cells: [],
        boxes: [],
        decoderVersion: 'six-dot-word-v2',
      }),
    );
    expect(
      screen.queryByRole('adjustable', { name: 'Lebar bingkai' }),
    ).toBeNull();
    await act(() => cameraProps.onEvent(createSuccess('MAMA')));
    expect(
      screen.queryByRole('adjustable', { name: 'Lebar bingkai' }),
    ).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai ulang' }));
    expect(slider()).toBeEnabled();
    expect(slider()).toHaveAccessibilityValue({ now: 6.35 });
    expect(cameraProps.guideAspectRatio).toBe(6.35);
    await fireEvent(slider(), 'valueChange', 1);
    expect(cameraProps.guideAspectRatio).toBe(1);
  });

  it('supports accessible frame adjustment and clamps its range', async () => {
    await setup();
    const slider = () =>
      screen.getByRole('adjustable', { name: 'Lebar bingkai' });
    await fireEvent(slider(), 'accessibilityAction', {
      nativeEvent: { actionName: 'increment' },
    });
    expect(cameraProps.guideAspectRatio).toBe(4.25);
    await fireEvent(slider(), 'accessibilityAction', {
      nativeEvent: { actionName: 'decrement' },
    });
    expect(cameraProps.guideAspectRatio).toBe(4);
    await fireEvent(slider(), 'valueChange', 20);
    expect(cameraProps.guideAspectRatio).toBe(8);
    await fireEvent(slider(), 'valueChange', -2);
    expect(cameraProps.guideAspectRatio).toBe(1);
    await fireEvent(slider(), 'valueChange', Number.NaN);
    expect(cameraProps.guideAspectRatio).toBe(1);
  });

  it.each(['background', 'blur'] as const)(
    'cancels queued TalkBack guidance on %s',
    async (exitKind) => {
      jest.useFakeTimers();
      const announce = jest.spyOn(AccessibilityInfo, 'announceForAccessibility');
      let onAppState: ((state: AppStateStatus) => void) | undefined;
      const originalAddEventListener = AppState.addEventListener;
      AppState.addEventListener = jest.fn((_event, callback) => {
        onAppState = callback;
        return { remove: jest.fn() };
      });
      try {
        const test = await setup();
        await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
        await act(() =>
          cameraProps.onEvent({
            schemaVersion: 3,
            sessionId: cameraProps.sessionId,
            captureId: 1,
            timestampMs: 1000,
            status: 'rejected',
            reason: 'dark',
            cells: [],
            boxes: [],
            decoderVersion: 'test',
          }),
        );
        // The rejected guidance is throttled behind the initial camera hint.
        expect(announce).toHaveBeenCalledTimes(1);
        announce.mockClear();
        await act(() => {
          if (exitKind === 'background') onAppState?.('background');
          else
            jest.requireMock<{
              __setRouteFocused(value: boolean): void;
            }>('expo-router').__setRouteFocused(false);
        });
        await act(() => jest.advanceTimersByTime(4000));
        expect(announce).not.toHaveBeenCalled();
        await test.view.unmount();
      } finally {
        AppState.addEventListener = originalAddEventListener;
        announce.mockRestore();
        jest.useRealTimers();
      }
    },
  );

  it('reads only the latest successful capture and ignores an abandoned speech completion', async () => {
    let onPlayback: TtsPlaybackListener | undefined;
    let finishSpeech: ((value: { ok: true }) => void) | undefined;
    const pending = new Promise<{ ok: true }>((resolve) => {
      finishSpeech = resolve;
    });
    const test = await setup(undefined, {
      speak: jest.fn().mockReturnValueOnce(pending).mockResolvedValue({ ok: true }),
      subscribePlayback: (listener) => {
        onPlayback = listener;
        return jest.fn();
      },
    });
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    await act(() => cameraProps.onEvent(createSuccess('MAMA')));
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    await act(() => onPlayback?.({ status: 'speaking', text: 'MAMA' }));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeDisabled();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai ulang' }));
    expect(screen.queryByRole('button', { name: 'Bacakan' })).toBeNull();
    await act(() => finishSpeech?.({ ok: true }));
    expect(screen.queryByRole('button', { name: 'Hentikan suara' })).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    await act(() => cameraProps.onEvent(createSuccess('KAKAK')));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeEnabled();
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    expect(test.tts.speak).toHaveBeenLastCalledWith('KAKAK', { rate: 1, volume: 1 });
  });

  it('keeps reading disabled when the available voice requires network access', async () => {
    const test = await setup(undefined, {
      getAvailability: jest.fn(async () => ({
        status: 'available' as const,
        localVoiceAvailable: false,
        message: 'Only network voice',
      })),
    });
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai kata' }));
    await act(() => cameraProps.onEvent(createSuccess('A')));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeDisabled();
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    expect(test.tts.speak).not.toHaveBeenCalled();
  });
});
