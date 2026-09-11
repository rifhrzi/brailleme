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
import { WORD_DECODER_VERSION, WORD_LATIN_MASKS } from '@/ml/word-model-contract';
import { parseWordMasks } from '@/ml/word-symbols';
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
  physicalMasks?: readonly number[],
): WordScanEvent {
  const masks = physicalMasks ?? (/^\d+$/.test(text)
    ? [60, ...[...text].map((digit) => WORD_LATIN_MASKS['1234567890'.indexOf(digit)]!)]
    : [...text].flatMap((letter) => [
      ...(letter === letter.toUpperCase() ? [32] : []),
      WORD_LATIN_MASKS[letter.toUpperCase().charCodeAt(0) - 65]!,
    ]));
  const parsed = parseWordMasks(masks);
  if (parsed.status !== 'success' || parsed.text !== text) throw new Error('Invalid test token');
  const cells = parsed.symbols.map((symbol, index) => ({
    ...symbol,
    confidence: 0.98,
    box: { x: 0.1 + index * 0.03, y: 0.4, width: 0.025, height: 0.15 },
  }));
  return {
    schemaVersion: 4,
    sessionId: cameraProps.sessionId,
    captureId: 1,
    timestampMs: 1000,
    status: 'success',
    text,
    confidence: 0.98,
    cells,
    boxes: cells.map((cell) => cell.box),
    decoderVersion: WORD_DECODER_VERSION,
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
  it('does not enter an endless capture when the camera handle disappears before the tap', async () => {
    await setup();
    await act(() => {
      if (typeof cameraProps.cameraRef !== 'function') throw new Error('Expected callback camera ref');
      cameraProps.cameraRef(null);
    });
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    expect(capture).not.toHaveBeenCalled();
    expect(screen.queryByText('Memproses foto')).toBeNull();
    expect(screen.getByRole('button', { name: 'Pindai' })).toBeDisabled();
  });

  it('explains a startup failure and opens a fresh camera session on retry', async () => {
    await setup();
    const failedSession = cameraProps.sessionId;
    await act(() => cameraProps.onEvent({
      schemaVersion: 4, sessionId: failedSession, captureId: 0,
      timestampMs: performance.now(), status: 'error', reason: 'capture_error',
      cells: [], boxes: [], decoderVersion: WORD_DECODER_VERSION,
    }));
    expect(screen.getByText(/Kamera belum dapat dibuka/)).toBeOnTheScreen();
    expect(screen.queryByText(/Foto belum dapat diproses/)).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai ulang' }));
    expect(cameraProps.sessionId).not.toBe(failedSession);
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    expect(capture).toHaveBeenCalledTimes(1);
  });
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
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
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
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    await act(() => cameraProps.onEvent(createSuccess('KAKAK')));
    expect(screen.getByText('KAKAK')).toBeOnTheScreen();
    expect(test.onWordDraftChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ letters: ['K', 'A', 'K', 'A', 'K'] }),
    );
  });

  it.each(['low_confidence', 'ambiguous_symbol'] as const)('rejects stale and partial results and never enables speech for %s captures', async (reason) => {
    await setup();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    const stale = createSuccess('MAMA');
    await act(() =>
      cameraProps.onEvent({ ...stale, boxes: stale.boxes.slice(1) }),
    );
    expect(screen.queryByText('MAMA')).toBeNull();
    await act(() =>
      cameraProps.onEvent({
        schemaVersion: 4,
        sessionId: cameraProps.sessionId,
        captureId: 1,
        timestampMs: 1000,
        status: 'rejected',
        reason,
        cells: [],
        boxes: [],
        decoderVersion: WORD_DECODER_VERSION,
      }),
    );
    expect(screen.queryByRole('button', { name: 'Bacakan' })).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai ulang' }));
    await act(() => cameraProps.onEvent(stale));
    expect(screen.queryByText('MAMA')).toBeNull();
    expect(screen.getByRole('button', { name: 'Pindai' })).toBeEnabled();
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
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
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
        screen.getByRole('button', { name: 'Pindai' }),
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
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    await act(() => cameraProps.onEvent(createSuccess('MAMA')));
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    await act(() => updateAppPreferences({ muted: true }));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeDisabled();
    await act(() => updateAppPreferences({ muted: false }));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeEnabled();
    expect(test.tts.stop).toHaveBeenCalled();
  });

  it('shows tracking states without a slider and requires an explicit shutter', async () => {
    await setup();
    expect(screen.queryByRole('adjustable')).toBeNull();
    await act(() => cameraProps.onTracking('searching'));
    await act(() => cameraProps.onAvailability(false, true));
    expect(screen.getByText('Mencari Braille')).toBeOnTheScreen();
    expect(screen.getByRole('button', { name: 'Pindai' })).toBeDisabled();
    await act(() => cameraProps.onTracking('detected'));
    expect(screen.getByText('Braille terdeteksi')).toBeOnTheScreen();
    await act(() => cameraProps.onTracking('ready'));
    await act(() => cameraProps.onAvailability(true, true));
    expect(screen.getByText('Tahan kamera, lalu tekan Pindai')).toBeOnTheScreen();
    expect(capture).not.toHaveBeenCalled();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    expect(capture).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['abc', undefined], ['Abc', undefined], ['ABC', undefined], ['2026', undefined], ['001', undefined],
    ['Baca?', [32, 3, 1, 9, 1, 38]], ['ABC!', [32, 32, 1, 3, 9, 22]], ['(001)', [54, 60, 26, 26, 1, 54]],
  ] as const)(
    'preserves %s in the result, draft, and speech request', async (text, masks) => {
      const test = await setup();
      await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
      await act(() => cameraProps.onEvent(createSuccess(text, {}, masks)));
      expect(screen.getByText(text)).toBeOnTheScreen();
      expect(test.onWordDraftChange).toHaveBeenLastCalledWith(expect.objectContaining({ letters: [...text] }));
      await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
      expect(test.tts.speak).toHaveBeenLastCalledWith(text, { rate: 1, volume: 1 });
    },
  );
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
        await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
        await act(() =>
          cameraProps.onEvent({
            schemaVersion: 4,
            sessionId: cameraProps.sessionId,
            captureId: 1,
            timestampMs: 1000,
            status: 'rejected',
            reason: 'dark',
            cells: [],
            boxes: [],
            decoderVersion: WORD_DECODER_VERSION,
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
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    await act(() => cameraProps.onEvent(createSuccess('MAMA')));
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    await act(() => onPlayback?.({ status: 'speaking', text: 'MAMA' }));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeDisabled();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai ulang' }));
    expect(screen.queryByRole('button', { name: 'Bacakan' })).toBeNull();
    await act(() => finishSpeech?.({ ok: true }));
    expect(screen.queryByRole('button', { name: 'Hentikan suara' })).toBeNull();
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
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
    await fireEvent.press(screen.getByRole('button', { name: 'Pindai' }));
    await act(() => cameraProps.onEvent(createSuccess('A')));
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeDisabled();
    await fireEvent.press(screen.getByRole('button', { name: 'Bacakan' }));
    expect(test.tts.speak).not.toHaveBeenCalled();
  });
});
