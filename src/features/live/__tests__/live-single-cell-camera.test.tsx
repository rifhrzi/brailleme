import { act, render } from '@testing-library/react-native';
import { AppState, type AppStateStatus } from 'react-native';
import type { TensorflowModel } from 'react-native-fast-tflite';

import LiveSingleCellCamera from '@/features/live/live-single-cell-camera';
import {
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_LENGTH,
  BRAILLE_MODEL_INPUT_WIDTH,
} from '@/ml/braille-model-contract';
import { SingleCellScannerPort } from '@/services';
import type { LiveScanEvent, ScannerStartResult } from '@/types';

type FrameCallback = (frame: { dispose(): void }) => void;

let mockFrameCallback: FrameCallback | undefined;
let mockNow = 1_000;
let mockCameraDevice: { readonly hasTorch: boolean; readonly id: string } | undefined;
let mockCameraProps: Record<string, unknown> | undefined;
let mockFrameOutputConfiguration: Record<string, unknown> | undefined;
const mockConvertFrameToImage = jest.fn();
const originalAppState = AppState.currentState;

// `@/ml` juga mengekspor loader native. Kamera di suite ini hanya membutuhkan
// kontrak model sinkron, jadi cegah TurboModule dievaluasi di lingkungan Jest.
jest.mock('react-native-fast-tflite', () => ({}));

jest.mock('@/ml', () => ({
  ...jest.requireActual('@/ml/braille-model-contract'),
  ...jest.requireActual('@/ml/frame-quality'),
  ...jest.requireActual('@/ml/frame-preprocessing'),
  ...jest.requireActual('@/ml/prediction-stabilizer'),
}));

jest.mock('expo-router', () => {
  const React = jest.requireActual<typeof import('react')>('react');

  return {
    __resetRouterMock: jest.fn(),
    useFocusEffect(effect: () => void | (() => void)) {
      React.useEffect(effect, [effect]);
    },
  };
});

jest.mock('react-native-vision-camera', () => ({
  Camera: (props: Record<string, unknown>) => {
    mockCameraProps = props;
    return null;
  },
  HybridFrameConverter: {
    convertFrameToImage: (...args: unknown[]) =>
      mockConvertFrameToImage(...args),
  },
  useCameraDevice: () => mockCameraDevice,
  useFrameOutput: (configuration: { onFrame: FrameCallback }) => {
    mockFrameOutputConfiguration = configuration;
    mockFrameCallback = configuration.onFrame;
    return { id: 'test-frame-output' };
  },
}));

jest.mock('react-native-worklets', () => ({
  createSynchronizable: <T,>(initialValue: T) => {
    let value = initialValue;
    return {
      getBlocking: () => value,
      setBlocking: (nextValue: T) => {
        value = nextValue;
      },
    };
  },
}));

function requireSession(result: ScannerStartResult): string {
  if (!result.ok) {
    throw new Error(result.message);
  }
  return result.sessionId;
}

function createModel(output?: Float32Array) {
  const probabilities = output ?? new Float32Array(26);
  if (!output) {
    probabilities[0] = 0.95;
    probabilities[1] = 0.05;
  }

  return {
    runSync: jest.fn(() => [probabilities.buffer]),
  } as unknown as TensorflowModel;
}

function createContrastingRgbPixels(): Uint8Array {
  const pixels = BRAILLE_MODEL_INPUT_WIDTH * BRAILLE_MODEL_INPUT_HEIGHT;
  const bytes = new Uint8Array(BRAILLE_MODEL_INPUT_LENGTH);
  for (let index = 0; index < pixels; index += 1) {
    if (index % 2 === 1) {
      bytes.set([255, 255, 255], index * 3);
    }
  }
  return bytes;
}

function createFrameResources() {
  // Piksel gelap/terang memberi luminance dan kontras yang melewati
  // quality gate sehingga jalur inference dapat diuji tanpa gambar native.
  const rawPixels = {
    buffer: createContrastingRgbPixels().buffer,
    width: BRAILLE_MODEL_INPUT_WIDTH,
    height: BRAILLE_MODEL_INPUT_HEIGHT,
    pixelFormat: 'RGB',
  };
  const resizedImage = {
    dispose: jest.fn(),
    toRawPixelData: jest.fn(() => rawPixels),
  };
  const croppedImage = {
    dispose: jest.fn(),
    resize: jest.fn(() => resizedImage),
  };
  const sourceImage = {
    width: 100,
    height: 80,
    crop: jest.fn(() => croppedImage),
    dispose: jest.fn(),
  };
  const frame = { dispose: jest.fn() };
  mockConvertFrameToImage.mockReturnValue(sourceImage);

  return { croppedImage, frame, resizedImage, sourceImage };
}

function processFrame(frame: { dispose(): void }) {
  if (!mockFrameCallback) {
    throw new Error('Frame processor belum terpasang.');
  }
  mockFrameCallback(frame);
}

async function setupCamera(
  model: TensorflowModel = createModel(),
  torchEnabled = false,
  resetToken = 0,
) {
  const scanner = new SingleCellScannerPort();
  const listener = jest.fn<void, [LiveScanEvent]>();
  scanner.subscribe(listener);
  const sessionId = requireSession(await scanner.start());
  const view = await render(
    <LiveSingleCellCamera
      active
      model={model}
      resetToken={resetToken}
      scanner={scanner}
      sessionId={sessionId}
      torchEnabled={torchEnabled}
    />,
  );

  listener.mockClear();
  return { listener, scanner, sessionId, view };
}

describe('LiveSingleCellCamera bridge', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockFrameCallback = undefined;
    mockCameraDevice = { hasTorch: true, id: 'back-camera' };
    mockCameraProps = undefined;
    mockFrameOutputConfiguration = undefined;
    mockNow = 1_000;
    jest.spyOn(performance, 'now').mockImplementation(() => mockNow);
    (AppState as { currentState: AppStateStatus }).currentState = 'active';
  });

  afterEach(() => {
    jest.restoreAllMocks();
    (AppState as { currentState: AppStateStatus }).currentState = originalAppState;
    jest.useRealTimers();
  });

  it('melepas frame dan seluruh image tepat sekali setelah inference', async () => {
    const { listener, sessionId, view } = await setupCamera();
    const resources = createFrameResources();

    processFrame(resources.frame);

    expect(resources.frame.dispose).toHaveBeenCalledTimes(1);
    expect(resources.resizedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.croppedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.sourceImage.dispose).toHaveBeenCalledTimes(1);

    await act(() => jest.advanceTimersByTime(200));
    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId,
        status: 'candidate',
        text: 'A',
      }),
    );
    await view.unmount();
  });

  it('mengubah output model malformed menjadi error dan tetap membersihkan resource', async () => {
    const warning = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    const malformedOutput = new Float32Array([0.7, 0.3]);
    const { listener, sessionId, view } = await setupCamera(
      createModel(malformedOutput),
    );
    const resources = createFrameResources();

    processFrame(resources.frame);
    await act(() => jest.advanceTimersByTime(200));

    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId, status: 'error' }),
    );
    expect(listener).not.toHaveBeenCalledWith(
      expect.objectContaining({ status: 'candidate' }),
    );
    expect(warning).toHaveBeenCalledWith(
      expect.stringContaining('Model menghasilkan probabilitas tidak valid'),
    );
    expect(resources.frame.dispose).toHaveBeenCalledTimes(1);
    expect(resources.resizedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.croppedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.sourceImage.dispose).toHaveBeenCalledTimes(1);
    await view.unmount();
  });

  it('menolak dimensi raw native malformed sebelum memanggil model', async () => {
    const warning = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    const model = createModel();
    const { listener, sessionId, view } = await setupCamera(model);
    const resources = createFrameResources();
    resources.resizedImage.toRawPixelData.mockReturnValue({
      buffer: Uint8Array.from([0, 0, 0, 255, 255, 255]).buffer,
      width: 2,
      height: 1,
      pixelFormat: 'RGB',
    });

    processFrame(resources.frame);
    await act(() => jest.advanceTimersByTime(200));

    expect(model.runSync).not.toHaveBeenCalled();
    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId, status: 'error' }),
    );
    expect(warning).toHaveBeenCalledWith(
      expect.stringContaining('Dimensi pixel kamera harus tepat 28 x 28'),
    );
    expect(resources.frame.dispose).toHaveBeenCalledTimes(1);
    expect(resources.resizedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.croppedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.sourceImage.dispose).toHaveBeenCalledTimes(1);
    await view.unmount();
  });

  it('melatch error pemrosesan agar frame berikutnya tidak menimpa status', async () => {
    const warning = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    const malformed = new Float32Array([0.7, 0.3]);
    const valid = new Float32Array(26);
    valid[0] = 0.95;
    valid[1] = 0.05;
    const model = {
      runSync: jest
        .fn()
        .mockReturnValueOnce([malformed.buffer])
        .mockReturnValue([valid.buffer]),
    } as unknown as TensorflowModel;
    const { listener, sessionId, view } = await setupCamera(model);

    const malformedFrame = createFrameResources();
    processFrame(malformedFrame.frame);
    await act(() => jest.advanceTimersByTime(200));
    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId, status: 'error' }),
    );
    expect(warning).toHaveBeenCalledTimes(1);
    listener.mockClear();

    mockNow = 1_300;
    const lateValidFrame = createFrameResources();
    processFrame(lateValidFrame.frame);
    await act(() => jest.advanceTimersByTime(200));

    expect(model.runSync).toHaveBeenCalledTimes(1);
    expect(listener).not.toHaveBeenCalled();
    expect(lateValidFrame.frame.dispose).toHaveBeenCalledTimes(1);
    await view.unmount();
  });

  it('reset token menghapus kandidat lama sebelum stabilisasi berikutnya', async () => {
    const model = createModel();
    const { listener, scanner, sessionId, view } = await setupCamera(model);

    processFrame(createFrameResources().frame);
    await act(() => jest.advanceTimersByTime(200));
    expect(listener).toHaveBeenLastCalledWith(
      expect.objectContaining({ status: 'candidate' }),
    );

    mockNow = 1_300;
    processFrame(createFrameResources().frame);
    await act(() => jest.advanceTimersByTime(200));
    expect(listener).toHaveBeenLastCalledWith(
      expect.objectContaining({ status: 'candidate' }),
    );

    await view.rerender(
      <LiveSingleCellCamera
        active
        model={model}
        resetToken={1}
        scanner={scanner}
        sessionId={sessionId}
        torchEnabled={false}
      />,
    );
    listener.mockClear();

    mockNow = 1_600;
    processFrame(createFrameResources().frame);
    await act(() => jest.advanceTimersByTime(200));

    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId, status: 'candidate' }),
    );
    expect(listener).not.toHaveBeenCalledWith(
      expect.objectContaining({ status: 'stable' }),
    );
    await view.unmount();
  });

  it('tetap mencoba seluruh cleanup ketika satu native dispose melempar', async () => {
    const { view } = await setupCamera();
    const resources = createFrameResources();
    resources.resizedImage.dispose.mockImplementation(() => {
      throw new Error('Native image sudah dilepas.');
    });

    expect(() => processFrame(resources.frame)).not.toThrow();

    expect(resources.resizedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.croppedImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.sourceImage.dispose).toHaveBeenCalledTimes(1);
    expect(resources.frame.dispose).toHaveBeenCalledTimes(1);
    await view.unmount();
  });

  it('membuang hasil frame sesi lama walau preview sudah memakai sesi baru', async () => {
    const model = createModel();
    const { listener, scanner, sessionId: oldSessionId, view } =
      await setupCamera(model);
    const resources = createFrameResources();
    processFrame(resources.frame);

    await scanner.stop();
    const newSessionId = requireSession(await scanner.start());
    expect(newSessionId).not.toBe(oldSessionId);
    await view.rerender(
      <LiveSingleCellCamera
        active
        model={model}
        scanner={scanner}
        sessionId={newSessionId}
        torchEnabled={false}
      />,
    );
    listener.mockClear();

    await act(() => jest.advanceTimersByTime(200));

    expect(model.runSync).toHaveBeenCalledTimes(1);
    expect(listener).not.toHaveBeenCalled();
    expect(resources.frame.dispose).toHaveBeenCalledTimes(1);
    await view.unmount();
  });

  it('membuang frame yang datang sebelum interval tanpa membuat image native', async () => {
    const { view } = await setupCamera();
    const first = createFrameResources();
    processFrame(first.frame);
    expect(mockConvertFrameToImage).toHaveBeenCalledTimes(1);

    mockNow = 1_100;
    const throttled = { dispose: jest.fn() };
    processFrame(throttled);

    expect(throttled.dispose).toHaveBeenCalledTimes(1);
    expect(mockConvertFrameToImage).toHaveBeenCalledTimes(1);
    await view.unmount();
  });

  it('menjaga orientasi fisik dan tidak meminta torch yang tidak dimiliki kamera', async () => {
    mockCameraDevice = { hasTorch: false, id: 'back-camera-no-torch' };
    const { view } = await setupCamera(createModel(), true);

    expect(mockFrameOutputConfiguration).toEqual(
      expect.objectContaining({
        enablePhysicalBufferRotation: true,
        targetResolution: { width: 640, height: 480 },
      }),
    );
    expect(mockCameraProps?.torchMode).toBeUndefined();
    expect(mockCameraProps?.constraints).toEqual([{ fps: 30 }]);
    await view.unmount();
  });

  it('menolak frame gelap sebelum menjalankan model', async () => {
    const model = createModel();
    const { listener, sessionId, view } = await setupCamera(model);
    const resources = createFrameResources();
    resources.resizedImage.toRawPixelData.mockReturnValue({
      buffer: new Uint8Array(BRAILLE_MODEL_INPUT_LENGTH).buffer,
      width: BRAILLE_MODEL_INPUT_WIDTH,
      height: BRAILLE_MODEL_INPUT_HEIGHT,
      pixelFormat: 'RGB',
    });

    processFrame(resources.frame);
    await act(() => jest.advanceTimersByTime(200));

    expect(model.runSync).not.toHaveBeenCalled();
    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId,
        status: 'quality_issue',
        reason: 'dark',
      }),
    );
    await view.unmount();
  });

  it('mempertahankan error worklet walau frame valid datang sebelum polling', async () => {
    const malformed = new Float32Array([0.7, 0.3]);
    const model = createModel(malformed);
    const warning = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
    const { listener, sessionId, view } = await setupCamera(model);

    processFrame(createFrameResources().frame);
    mockNow = 1_300;
    processFrame(createFrameResources().frame);
    await act(() => jest.advanceTimersByTime(200));

    expect(model.runSync).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId, status: 'error' }),
    );
    expect(warning).toHaveBeenCalledTimes(1);
    await view.unmount();
  });

  it('mengabaikan hasil callback lama setelah reset generation', async () => {
    const model = createModel();
    const { listener, scanner, sessionId, view } = await setupCamera(model);
    const oldFrameCallback = mockFrameCallback;

    await view.rerender(
      <LiveSingleCellCamera
        active
        model={model}
        resetToken={1}
        scanner={scanner}
        sessionId={sessionId}
        torchEnabled={false}
      />,
    );
    listener.mockClear();
    mockNow = 1_300;
    oldFrameCallback?.(createFrameResources().frame);
    await act(() => jest.advanceTimersByTime(200));

    expect(model.runSync).toHaveBeenCalledTimes(1);
    expect(listener).not.toHaveBeenCalledWith(
      expect.objectContaining({ status: 'candidate' }),
    );
    await view.unmount();
  });

  it('timeout pencarian kamera hanya memberi status lambat, bukan error fatal', async () => {
    mockCameraDevice = undefined;
    const { listener, scanner, sessionId, view } = await setupCamera();

    await act(() => jest.advanceTimersByTime(10_000));

    expect(
      view.getByText(/Kamera belakang masih disiapkan/i),
    ).toBeOnTheScreen();
    expect(listener).not.toHaveBeenCalledWith(
      expect.objectContaining({ status: 'error' }),
    );

    await view.rerender(
      <LiveSingleCellCamera
        active={false}
        model={createModel()}
        resetToken={1}
        scanner={scanner}
        sessionId={sessionId}
        torchEnabled={false}
      />,
    );
    expect(view.getByText('Menyiapkan kamera belakang.')).toBeOnTheScreen();
    await view.rerender(
      <LiveSingleCellCamera
        active
        model={createModel()}
        resetToken={1}
        scanner={scanner}
        sessionId={sessionId}
        torchEnabled={false}
      />,
    );
    await act(() => jest.advanceTimersByTime(10_000));
    expect(
      view.getByText(/Kamera belakang masih disiapkan/i),
    ).toBeOnTheScreen();

    mockCameraDevice = { hasTorch: true, id: 'late-back-camera' };
    await view.rerender(
      <LiveSingleCellCamera
        active
        model={createModel()}
        resetToken={1}
        scanner={scanner}
        sessionId={sessionId}
        torchEnabled={false}
      />,
    );
    expect(
      view.getByTestId('live-camera-preview', {
        includeHiddenElements: true,
      }),
    ).toBeOnTheScreen();

    await act(() => {
      (mockCameraProps?.onStarted as (() => void) | undefined)?.();
    });
    mockCameraDevice = undefined;
    await view.rerender(
      <LiveSingleCellCamera
        active
        model={createModel()}
        resetToken={1}
        scanner={scanner}
        sessionId={sessionId}
        torchEnabled={false}
      />,
    );
    expect(view.getByText('Menyiapkan kamera belakang.')).toBeOnTheScreen();
    await view.unmount();
  });

  it('menjaga preview persegi agar kotak panduan sesuai crop tengah', async () => {
    const { view } = await setupCamera();

    expect(
      view.getByTestId('live-camera-preview', {
        includeHiddenElements: true,
      }),
    ).toHaveStyle({
      width: '100%',
      maxWidth: 360,
      aspectRatio: 1,
      alignSelf: 'center',
    });
    await view.unmount();
  });

  it('menghentikan polling bridge ketika preview tidak aktif', async () => {
    const setIntervalSpy = jest.spyOn(globalThis, 'setInterval');
    const clearIntervalSpy = jest.spyOn(globalThis, 'clearInterval');
    const model = createModel();
    const { scanner, sessionId, view } = await setupCamera(model);
    const bridgeIntervalCall = setIntervalSpy.mock.calls.findIndex(
      (call) => call[1] === 200,
    );
    expect(bridgeIntervalCall).toBeGreaterThanOrEqual(0);
    const bridgeInterval = setIntervalSpy.mock.results[bridgeIntervalCall]?.value;

    await view.rerender(
      <LiveSingleCellCamera
        active={false}
        model={model}
        scanner={scanner}
        sessionId={sessionId}
        torchEnabled={false}
      />,
    );

    expect(clearIntervalSpy).toHaveBeenCalledWith(bridgeInterval);
    await view.unmount();
  });
});
