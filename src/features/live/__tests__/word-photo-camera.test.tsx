import { act, fireEvent, render, screen } from '@testing-library/react-native';
import { createRef } from 'react';
import type { TensorflowModel } from 'react-native-fast-tflite';

import WordPhotoCamera, {
  type WordPhotoCameraHandle,
} from '@/features/live/word-photo-camera';
import { DEFAULT_WORD_MODEL_THRESHOLDS } from '@/ml/word-model-contract';

const mockCapture = jest.fn();
const mockRecognize = jest.fn();
const mockGeometry = jest.fn();
const mockSensorPoint = jest.fn();
const mockSetTorch = jest.fn();
let mockCameraProps: Record<string, unknown>;
let mockOutputOptions: Record<string, unknown>;
let mockImage: unknown;
let mockRun: (() => Promise<unknown>) | undefined;
let mockScheduledRuntimes: unknown[] = [];
let mockAppEnvironment = 'production';
let mockHasTorch = true;
const mockOutput = { capturePhoto: mockCapture };
let mockTrackedBox = { x: 0.275, y: 0.44375, width: 0.45, height: 0.1125 };
const mockTrackingOutput = {};
const mockPixelProcessor = { resizePixels: jest.fn() };
const mockTrackingSnapshot = jest.fn();
let mockTrackingCurrent = true;

jest.mock('@/features/live/use-word-camera-tracking', () => ({
  useWordCameraTracking: (options: { active: boolean; onStatus: (status: string) => void }) => {
    const React = jest.requireActual('react');
    const callbacks = React.useRef(options);
    callbacks.current = options;
    React.useEffect(() => {
      callbacks.current.onStatus(options.active ? 'ready' : 'searching');
    }, [options.active]);
    return {
      output: mockTrackingOutput,
      display: { box: mockTrackedBox, boxes: [mockTrackedBox], score: 0.99, complete: true },
      status: options.active ? 'ready' : 'searching',
      snapshot: mockTrackingSnapshot,
      isCurrent: () => mockTrackingCurrent,
    };
  },
}));

jest.mock('@/config/app-environment', () => ({
  get APP_ENVIRONMENT() { return mockAppEnvironment; },
}));
jest.mock('react-native-vision-camera', () => ({
  Camera: (props: Record<string, unknown>) => {
    mockCameraProps = props;
    (props.ref as { current: unknown }).current = {
      controller: { setTorchMode: mockSetTorch },
      convertViewPointToCameraPoint: mockSensorPoint,
      preview: { previewOutput: { currentResolution: { width: 1920, height: 1440 } } },
    };
    return null;
  },
  useCameraDevice: () => ({ id: 'back', hasTorch: mockHasTorch }),
  usePhotoOutput: (options: Record<string, unknown>) => {
    mockOutputOptions = options;
    return mockOutput;
  },
}));
jest.mock('../../../../modules/braille-word-geometry/src', () => ({
  getWordPhotoGeometry: (...args: unknown[]) => mockGeometry(...args),
  createWordPixelProcessor: () => mockPixelProcessor,
}));
jest.mock('react-native-nitro-image', () => ({
  NitroImage: ({ image }: { image: unknown }) => {
    mockImage = image;
    return null;
  },
}));
jest.mock('react-native-worklets', () => ({
  createWorkletRuntime: () => ({ name: 'test-worker' }),
  runOnRuntimeAsync: (
    _runtime: unknown,
    worklet: (input: unknown) => unknown,
    input: unknown,
  ) => {
    mockScheduledRuntimes.push(_runtime);
    return mockRun ? mockRun() : Promise.resolve(worklet(input));
  },
}));
jest.mock('@/ml/word-image-processing', () => ({
  ...jest.requireActual('@/ml/word-image-processing'),
  recognizeTrackedWordImage: (...args: unknown[]) => mockRecognize(...args),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { resolve, promise };
}
const rejection = {
  status: 'rejected',
  reason: 'no_grid',
  cells: [],
  boxes: [],
};

async function setup(_legacyAspectRatio = 4, startPreview = true, torchEnabled = true) {
  const handle = createRef<WordPhotoCameraHandle>();
  const events = jest.fn();
  const release = jest.fn();
  const acquire = jest.fn(() => release);
  const image = { width: 1440, height: 1920, dispose: jest.fn() };
  const photo = {
    toImageAsync: jest.fn(async () => image),
    dispose: jest.fn(),
  };
  mockCapture.mockResolvedValue(photo);
  const props = {
    active: true,
    sessionId: 'photo-session',
    torchEnabled,
    cameraRef: handle,
    onEvent: events,
    onAvailability: jest.fn(),
    onTracking: jest.fn(),
    model: {} as TensorflowModel,
    trackingModel: {} as TensorflowModel,
    acquireTrackingModel: () => () => {},
    thresholds: DEFAULT_WORD_MODEL_THRESHOLDS,
    modelVersion: 'word-test',
    acquireModel: acquire,
  };
  const view = await render(<WordPhotoCamera {...props} />);
  await fireEvent(screen.getByTestId('word-photo-preview'), 'layout', {
    nativeEvent: { layout: { width: 320, height: 320 } },
  });
  await act(() => (mockCameraProps.onStarted as () => void)());
  if (startPreview) await act(() => (mockCameraProps.onPreviewStarted as () => void)());
  return { handle, events, acquire, release, image, photo, view, props };
}

describe('WordPhotoCamera', () => {
  beforeEach(() => {
    mockTrackingCurrent = true;
    mockCapture.mockReset();
    mockSetTorch.mockReset().mockResolvedValue(undefined);
    mockRecognize.mockReset().mockReturnValue(rejection);
    mockSensorPoint.mockReset().mockImplementation((point: { x: number; y: number }) => ({ x: point.x * 4.5, y: point.y * 4.5 + 240 }));
    mockGeometry.mockReset().mockReturnValue({ width: 1440, height: 1920, sensorToBuffer: [1, 0, 0, 0, 1, 0, 0, 0, 1], rotationDegrees: 0, isMirrored: false, cropRect: [0, 0, 1440, 1920] });
    mockRun = undefined;
    mockImage = undefined;
    mockScheduledRuntimes = [];
    mockAppEnvironment = 'production';
    mockHasTorch = true;
    mockTrackedBox = { x: 0.275, y: 0.44375, width: 0.45, height: 0.1125 };
    mockTrackingSnapshot.mockReset().mockImplementation(() => {
      const { x, y, width, height } = mockTrackedBox;
      return { generation: 1, timestampMs: performance.now(), points: [
        { x, y }, { x: x + width, y },
        { x: x + width, y: y + height }, { x, y: y + height },
      ].map((point) => mockSensorPoint({ x: point.x * 320, y: point.y * 320 })) };
    });
  });
  afterEach(() => {
    jest.restoreAllMocks();
    jest.useRealTimers();
  });

  it('does not send torch controls while opening the camera or when the torch starts off', async () => {
    const test = await setup(4, false, false);
    expect(mockCameraProps).not.toHaveProperty('torchMode');
    expect(mockSetTorch).not.toHaveBeenCalled();
    await act(() => (mockCameraProps.onPreviewStarted as () => void)());
    expect(mockSetTorch).not.toHaveBeenCalled();
    expect(test.props.onAvailability).toHaveBeenLastCalledWith(true, true);
    await test.view.rerender(<WordPhotoCamera {...test.props} torchEnabled />);
    expect(mockSetTorch).toHaveBeenLastCalledWith('on');
    await test.view.rerender(<WordPhotoCamera {...test.props} torchEnabled={false} />);
    expect(mockSetTorch).toHaveBeenLastCalledWith('off');
  });

  it('keeps preview and capture usable when a phone advertises a torch but rejects the control', async () => {
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    mockSetTorch.mockRejectedValue(new Error('Camera is not active'));
    const test = await setup();
    expect(test.props.onAvailability).toHaveBeenLastCalledWith(true, false);
    expect(test.events).not.toHaveBeenCalled();
    expect(screen.queryByText('Kamera belum dapat digunakan.')).toBeNull();
    await act(async () => test.handle.current?.capture(1));
    expect(mockCapture).toHaveBeenCalledTimes(1);
    expect(mockRecognize).toHaveBeenCalledTimes(1);
  });

  it('times out a detected camera that never streams and makes retry available', async () => {
    jest.useFakeTimers();
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    const test = await setup(4, false, false);
    await act(async () => jest.advanceTimersByTime(15_000));
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'error', captureId: 0 }));
    expect(test.props.onAvailability).toHaveBeenLastCalledWith(false, false);
    expect(screen.getByText('Kamera belum dapat digunakan.')).toBeOnTheScreen();
    expect(screen.queryByTestId('word-guide', { includeHiddenElements: true })).toBeNull();
  });

  it('cancels the startup timeout once preview is streaming', async () => {
    jest.useFakeTimers();
    const test = await setup();
    await act(async () => jest.advanceTimersByTime(20_000));
    expect(test.events).not.toHaveBeenCalled();
    expect(test.props.onAvailability).toHaveBeenLastCalledWith(true, true);
  });

  it('reports a real session failure and a fresh session can open and capture', async () => {
    const log = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const first = await setup();
    await act(() => (mockCameraProps.onError as (error: Error) => void)(new Error('Camera device unavailable')));
    expect(first.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'error', captureId: 0 }));
    expect(log).toHaveBeenCalledWith('BRAILLE_WORD_CAMERA_ERROR', expect.stringContaining('Camera device unavailable'));
    await first.view.unmount();
    const second = await setup();
    await act(async () => second.handle.current?.capture(1));
    expect(mockRecognize).toHaveBeenCalledTimes(1);
  });

  it('ignores a late camera shutdown error after a photo is captured', async () => {
    const test = await setup();
    await act(async () => test.handle.current?.capture(1));
    const count = test.events.mock.calls.length;
    await act(() => (mockCameraProps.onError as (error: Error) => void)(new Error('Camera is closed')));
    expect(test.events).toHaveBeenCalledTimes(count);
    expect(mockImage).toBe(test.image);
  });

  it('attributes a camera failure during the shutter to the active capture so the screen can retry', async () => {
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    const test = await setup();
    const pending = deferred<typeof test.photo>();
    mockCapture.mockReturnValue(pending.promise);
    await act(() => test.handle.current?.capture(7));
    await act(() => (mockCameraProps.onError as (error: Error) => void)(new Error('Camera disconnected')));
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'error', captureId: 7 }));
    await test.view.unmount();
    await act(async () => pending.resolve(test.photo));
    expect(test.photo.dispose).toHaveBeenCalledTimes(1);
    expect(mockRecognize).not.toHaveBeenCalled();
  });

  it('keeps a camera without a flash usable without sending unsupported torch commands', async () => {
    mockHasTorch = false;
    const test = await setup();
    expect(mockCameraProps).not.toHaveProperty('torchMode');
    expect(test.props.onAvailability).toHaveBeenLastCalledWith(true, false);
    await act(async () => test.handle.current?.capture(1));
    expect(mockCapture).toHaveBeenCalledTimes(1);
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'rejected', reason: 'no_grid' }));
  });

  it('logs only explicit numeric geometry metadata in the preview APK', async () => {
    const log = jest.spyOn(console, 'info').mockImplementation(() => {});
    mockAppEnvironment = 'preview';
    const test = await setup();
    await act(async () => test.handle.current?.capture(7));
    expect(log).toHaveBeenCalledTimes(2);
    expect(log.mock.calls[0]?.[0]).toBe('BRAILLE_WORD_CAPTURE_GEOMETRY');
    expect(JSON.parse(log.mock.calls[0]?.[1] as string)).toEqual({
      schemaVersion: 1,
      captureId: 7,
      preview: {
        widthDp: 320, heightDp: 320,
        resolution: { width: 1920, height: 1440 },
        sensorPoints: [
          { x: 396, y: 879 }, { x: expect.closeTo(1044), y: 879 },
          { x: expect.closeTo(1044), y: 1041 }, { x: 396, y: 1041 },
        ],
      },
      photo: mockGeometry.mock.results[0]?.value,
      upright: { width: 1440, height: 1920 },
      roi: mockRecognize.mock.calls[0]?.[3],
    });
  });

  it('does not emit capture geometry in the production APK', async () => {
    const log = jest.spyOn(console, 'info').mockImplementation(() => {});
    const test = await setup();
    await act(async () => test.handle.current?.capture(1));
    expect(mockRecognize).toHaveBeenCalledTimes(1);
    expect(log).not.toHaveBeenCalled();
  });

  it('still processes a preview capture when the diagnostic logger fails', async () => {
    jest.spyOn(console, 'info').mockImplementation(() => { throw new Error('logger unavailable'); });
    mockAppEnvironment = 'preview';
    const test = await setup();
    await act(async () => test.handle.current?.capture(1));
    expect(mockRecognize).toHaveBeenCalledTimes(1);
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'rejected', reason: 'no_grid' }));
  });

  it('captures only on request, targets quality photos, and maps the guide into the upright photo', async () => {
    const test = await setup();
    expect(mockCapture).not.toHaveBeenCalled();
    expect(mockOutputOptions).toEqual(
      expect.objectContaining({
        targetResolution: { width: 1920, height: 1440 },
        qualityPrioritization: 'quality',
      }),
    );
    await act(async () => {
      test.handle.current?.capture(1);
    });
    expect(mockCapture).toHaveBeenCalledTimes(1);
    expect(test.photo.toImageAsync).toHaveBeenCalledTimes(1);
    expect(test.photo.dispose).toHaveBeenCalledTimes(1);
    expect(mockImage).toBe(test.image);
    expect(test.image.dispose).not.toHaveBeenCalled();
    expect(mockRecognize).toHaveBeenCalledWith(
      test.image,
      {},
      DEFAULT_WORD_MODEL_THRESHOLDS,
      expect.objectContaining({
        x: expect.closeTo(0.275),
        y: expect.closeTo(0.4578125),
        width: expect.closeTo(0.45),
        height: expect.closeTo(0.084375),
      }),
      mockPixelProcessor,
    );
    expect(screen.queryByTestId('word-guide', { includeHiddenElements: true })).toBeNull();
    expect(test.events.mock.calls.map(([event]) => event.status)).toEqual([
      'capturing',
      'processing',
      'rejected',
    ]);
    expect(test.acquire).toHaveBeenCalledTimes(1);
    expect(test.release).toHaveBeenCalledTimes(1);
    expect(mockCameraProps.isActive).toBe(false);
    await test.view.unmount();
    expect(test.image.dispose).toHaveBeenCalledTimes(1);
  });

  it('prevents duplicate captures and defers image disposal until background inference settles', async () => {
    const pending = deferred<unknown>();
    mockRun = () => pending.promise;
    const test = await setup();
    await act(async () => {
      test.handle.current?.capture(1);
      test.handle.current?.capture(2);
    });
    expect(mockCapture).toHaveBeenCalledTimes(1);
    await test.view.unmount();
    expect(test.image.dispose).not.toHaveBeenCalled();
    expect(test.release).not.toHaveBeenCalled();
    await act(async () => pending.resolve(rejection));
    expect(test.image.dispose).toHaveBeenCalledTimes(1);
    expect(test.release).toHaveBeenCalledTimes(1);
    expect(test.events.mock.calls.map(([event]) => event.status)).toEqual([
      'capturing',
      'processing',
    ]);
  });

  it('hides the live rectangle while the still image is being re-localized', async () => {
    const pending = deferred<unknown>();
    mockRun = () => pending.promise;
    const test = await setup();
    await act(async () => test.handle.current?.capture(1));
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'processing' }));
    expect(screen.queryByTestId('word-guide', { includeHiddenElements: true })).toBeNull();
    await act(async () => pending.resolve(rejection));
    expect(screen.queryByTestId('word-guide', { includeHiddenElements: true })).toBeNull();
  });

  it('rejects a shutter whose viewport identity changed before the photo arrived', async () => {
    const test = await setup();
    const pending = deferred<typeof test.photo>();
    mockCapture.mockReturnValue(pending.promise);
    await act(() => test.handle.current?.capture(1));
    mockTrackingCurrent = false;
    await act(async () => pending.resolve(test.photo));
    expect(test.photo.dispose).toHaveBeenCalledTimes(1);
    expect(test.photo.toImageAsync).not.toHaveBeenCalled();
    expect(mockRecognize).not.toHaveBeenCalled();
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'error', reason: 'capture_error' }));
  });

  it('discards inference results when orientation changes during processing', async () => {
    const pending = deferred<unknown>();
    mockRun = () => pending.promise;
    const test = await setup();
    await act(async () => test.handle.current?.capture(1));
    mockTrackingCurrent = false;
    await act(async () => pending.resolve({ ...rejection, text: 'stale' }));
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'error', reason: 'processing_error' }));
    expect(test.events.mock.calls.some(([event]) => event.text === 'stale')).toBe(false);
    expect(test.release).toHaveBeenCalledTimes(1);
  });

  it.each(['capture', 'processing'])('includes %s failures in preview timing without recording scan contents', async (failure) => {
    mockAppEnvironment = 'preview';
    const log = jest.spyOn(console, 'info').mockImplementation(() => {});
    try {
      const test = await setup();
      if (failure === 'capture') mockCapture.mockRejectedValue(new Error('private camera detail'));
      else mockRecognize.mockImplementation(() => { throw new Error('private image detail'); });
      await act(async () => test.handle.current?.capture(7));
      const metrics = log.mock.calls.filter(([tag]) => tag === 'BRAILLE_WORD_PHOTO_METRICS');
      expect(metrics).toHaveLength(1);
      expect(JSON.parse(metrics[0]![1])).toEqual({
        schemaVersion: 1, captureId: 7, elapsedMs: expect.any(Number), status: 'error',
      });
    } finally {
      log.mockRestore();
    }
  });

  it('maps a narrower preview sensor crop and draws the captured guide at that actual photo ROI', async () => {
    const test = await setup();
    mockRecognize.mockImplementation((_source, _model, _thresholds, roi) => ({ ...rejection, reason: 'low_confidence', boxes: [roi] }));
    mockSensorPoint.mockImplementation((point: { x: number; y: number }) => ({ x: 180 + point.x * 3.375, y: 420 + point.y * 3.375 }));
    await act(async () => test.handle.current?.capture(1));
    expect(mockRecognize.mock.calls[0]?.[3]).toEqual(expect.objectContaining({
      x: expect.closeTo(0.33125), y: expect.closeTo(0.468359375), width: expect.closeTo(0.3375), height: expect.closeTo(0.06328125),
    }));
    expect(screen.getByTestId('word-guide', { includeHiddenElements: true }).props.style[1]).toMatchObject({ width: 108, height: expect.closeTo(27) });
    expect(mockGeometry).toHaveBeenCalledWith(test.photo);
    expect(mockGeometry.mock.invocationCallOrder[0]).toBeLessThan(test.photo.dispose.mock.invocationCallOrder[0]!);
  });

  it('copies sensor coordinates before awaiting the shutter', async () => {
    const test = await setup();
    const pending = deferred<typeof test.photo>();
    mockCapture.mockReturnValue(pending.promise);
    await act(() => test.handle.current?.capture(1));
    expect(mockSensorPoint).toHaveBeenCalledTimes(4);
    mockSensorPoint.mockImplementation(() => ({ x: NaN, y: NaN }));
    await act(async () => pending.resolve(test.photo));
    expect(mockRecognize).toHaveBeenCalledTimes(1);
    expect(mockSensorPoint).toHaveBeenCalledTimes(4);
  });

  it('rejects missing photo geometry and never runs inference on a guessed ROI', async () => {
    const test = await setup();
    mockGeometry.mockImplementation(() => { throw new Error('native geometry unavailable'); });
    await act(async () => test.handle.current?.capture(1));
    expect(mockRecognize).not.toHaveBeenCalled();
    expect(test.photo.dispose).toHaveBeenCalledTimes(1);
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'error', reason: 'capture_error' }));
  });

  it('waits for actual preview streaming before accepting a shutter request', async () => {
    const test = await setup();
    await act(() => (mockCameraProps.onPreviewStopped as () => void)());
    await act(async () => test.handle.current?.capture(1));
    expect(mockCapture).not.toHaveBeenCalled();
    expect(test.events).toHaveBeenLastCalledWith(expect.objectContaining({ status: 'error', reason: 'capture_error' }));
  });

  it('disposes a late photo without converting it after leaving the screen', async () => {
    const test = await setup();
    const pending = deferred<typeof test.photo>();
    mockCapture.mockReturnValue(pending.promise);
    await act(() => test.handle.current?.capture(1));
    await test.view.unmount();
    await act(async () => pending.resolve(test.photo));
    expect(test.photo.dispose).toHaveBeenCalledTimes(1);
    expect(test.photo.toImageAsync).not.toHaveBeenCalled();
    expect(test.acquire).not.toHaveBeenCalled();
  });

  it('keeps successive sessions on one worker while a previous capture is still processing', async () => {
    const pending = deferred<unknown>();
    mockRun = () => pending.promise;
    const first = await setup();
    await act(async () => first.handle.current?.capture(1));
    await first.view.unmount();
    const next = await setup();
    await act(async () => next.handle.current?.capture(1));
    expect(mockScheduledRuntimes).toHaveLength(2);
    expect(mockScheduledRuntimes[1]).toBe(mockScheduledRuntimes[0]);
    expect(first.release).not.toHaveBeenCalled();
    await act(async () => pending.resolve(rejection));
    expect(first.release).toHaveBeenCalledTimes(1);
    expect(first.image.dispose).toHaveBeenCalledTimes(1);
    expect(next.release).toHaveBeenCalledTimes(1);
    expect(next.image.dispose).not.toHaveBeenCalled();
  });

  it('reports capture/conversion failure while releasing photo resources', async () => {
    const test = await setup();
    test.photo.toImageAsync.mockRejectedValue(new Error('failed bitmap'));
    await act(async () => test.handle.current?.capture(1));
    expect(test.photo.dispose).toHaveBeenCalledTimes(1);
    expect(test.events).toHaveBeenLastCalledWith(
      expect.objectContaining({
        status: 'error',
        reason: 'capture_error',
        captureId: 1,
      }),
    );
    expect(test.acquire).not.toHaveBeenCalled();
  });

  it('reports a stopped camera instead of leaving the screen waiting for a capture', async () => {
    const test = await setup();
    await act(() => (mockCameraProps.onStopped as () => void)());
    await act(async () => test.handle.current?.capture(1));
    expect(mockCapture).not.toHaveBeenCalled();
    expect(test.events).toHaveBeenLastCalledWith(
      expect.objectContaining({
        status: 'error',
        reason: 'capture_error',
        captureId: 1,
      }),
    );
  });

  it('follows tracked geometry and freezes sensor coordinates while the photo is pending', async () => {
    mockTrackedBox = { x: 0.44375, y: 0.44375, width: 0.1125, height: 0.1125 };
    const test = await setup();
    expect(
      screen.getByTestId('word-guide', { includeHiddenElements: true }),
    ).toHaveStyle({ width: 36, height: 36 });
    mockTrackedBox = { x: 0.275, y: 0.44375, width: 0.45, height: 0.1125 };
    await test.view.rerender(<WordPhotoCamera {...test.props} onTracking={jest.fn()} />);
    expect(screen.getByTestId('word-guide', { includeHiddenElements: true }).props.style[1]).toMatchObject({ width: expect.closeTo(144), height: 36 });
    const pending = deferred<typeof test.photo>();
    mockCapture.mockReturnValue(pending.promise);
    await act(() => test.handle.current?.capture(1));
    mockTrackedBox = { x: 0.1, y: 0.2, width: 0.8, height: 0.2 };
    await test.view.rerender(<WordPhotoCamera {...test.props} onTracking={jest.fn()} />);
    await act(async () => pending.resolve(test.photo));
    expect(screen.queryByTestId('word-guide', { includeHiddenElements: true })).toBeNull();
    expect(mockRecognize).toHaveBeenCalledWith(
      test.image,
      {},
      DEFAULT_WORD_MODEL_THRESHOLDS,
      expect.objectContaining({
        x: expect.closeTo(0.275),
        width: expect.closeTo(0.45),
        height: expect.closeTo(0.084375),
      }),
      mockPixelProcessor,
    );
  });

  it('keeps ordered cell boxes aligned with the centered cover photo after a preview resize', async () => {
    const test = await setup();
    const boxes = [
      { x: 0.3, y: 0.47, width: 0.05, height: 0.06 },
      { x: 0.5, y: 0.47, width: 0.05, height: 0.06 },
    ];
    mockRecognize.mockReturnValue({
      status: 'success',
      text: 'aa',
      confidence: 0.98,
      cells: boxes.map((box) => ({ role: 'letter', output: 'a', mask: 1, confidence: 0.98, box })),
      boxes,
    });
    await act(async () => test.handle.current?.capture(1));
    // 1440x1920 upright photo uses scale 320/1440 and a -160/3 top offset.
    const first = () => screen.getByTestId('word-cell-0', { includeHiddenElements: true });
    const second = () => screen.getByTestId('word-cell-1', { includeHiddenElements: true });
    expect(first()).toHaveStyle({ left: 96, width: 16 });
    expect(second()).toHaveStyle({ left: 160, width: 16 });
    expect(first().props.style[1].top).toBeCloseTo(147.2);
    expect(first().props.style[1].height).toBeCloseTo(25.6);
    await fireEvent(screen.getByTestId('word-photo-preview'), 'layout', {
      nativeEvent: { layout: { width: 480, height: 480 } },
    });
    expect(first()).toHaveStyle({ left: 144, width: 24 });
    expect(second()).toHaveStyle({ left: 240, width: 24 });
    expect(first().props.style[1].top).toBeCloseTo(220.8);
    expect(first().props.style[1].height).toBeCloseTo(38.4);
    expect(screen.queryByTestId('word-cell-0')).toBeNull();
  });
});
