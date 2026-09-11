import { act, renderHook } from '@testing-library/react-native';
import type { TensorflowModel } from 'react-native-fast-tflite';
import type { CameraRef } from 'react-native-vision-camera';
import { useWordCameraTracking } from '@/features/live/use-word-camera-tracking';
import { DEFAULT_WORD_MODEL_THRESHOLDS } from '@/ml/word-model-contract';
import type { WordTrackingGroup } from '@/ml/word-tracking';

let mockNow = 0;
let mockEnvironment = 'production';
const mockBooleanFlags: { getBlocking: () => boolean; setBlocking: (value: boolean) => void; lock: jest.Mock; unlock: jest.Mock; isLocked: () => boolean }[] = [];
let mockOrientation = 'up';
let mockOnFrame: (frame: ReturnType<typeof makeFrame>) => void;
let mockOutputOptions: Record<string, unknown>;
const mockDetect = jest.fn();
const mockSource = { width: 1000, height: 1000, dispose: jest.fn() };
const mockConvert = jest.fn(() => mockSource);
const mockOutput = { id: 'tracking-output' };

jest.mock('@/config/app-environment', () => ({ get APP_ENVIRONMENT() { return mockEnvironment; } }));
jest.mock('react-native-vision-camera', () => ({
  useOrientation: () => mockOrientation,
  useFrameOutput: (options: Record<string, unknown>) => {
    mockOutputOptions = options;
    mockOnFrame = options.onFrame as typeof mockOnFrame;
    return mockOutput;
  },
  HybridFrameConverter: { convertFrameToImage: () => mockConvert() },
}));
jest.mock('react-native-worklets', () => ({
  createSynchronizable: <T,>(initial: T) => {
    let value = initial;
    let locked = false;
    const flag = {
      getBlocking: () => value, setBlocking: (next: T) => { value = next; },
      lock: jest.fn(() => { if (locked) throw new Error('Mutex already locked'); locked = true; }),
      unlock: jest.fn(() => { locked = false; }), isLocked: () => locked,
    };
    if (typeof initial === 'boolean') mockBooleanFlags.push(flag as unknown as typeof mockBooleanFlags[number]);
    return flag;
  },
}));
jest.mock('@/ml/word-image-processing', () => ({ detectWordImageCells: (...args: unknown[]) => mockDetect(...args) }));

function makeFrame() {
  return {
    width: 1000, height: 1000, orientation: 'up' as 'up' | 'right' | 'down' | 'left', isMirrored: false,
    convertFramePointToCameraPoint: ({ x, y }: { x: number; y: number }) => ({ x, y }),
    convertCameraPointToFramePoint: ({ x, y }: { x: number; y: number }) => ({ x, y }),
    dispose: jest.fn(),
  };
}
function detected(x = 0.4): WordTrackingGroup {
  return { box: { x, y: 0.45, width: 0.15, height: 0.1 }, boxes: [
    { x, y: 0.45, width: 0.05, height: 0.1 }, { x: x + 0.1, y: 0.45, width: 0.05, height: 0.1 },
  ], score: 0.98, complete: true };
}
async function setup(sharedModel?: TensorflowModel) {
  const release = jest.fn();
  const props = {
    active: true, sessionId: 'session-a', layout: { width: 320, height: 320 },
    camera: { current: {
      convertCameraPointToViewPoint: ({ x, y }: { x: number; y: number }) => ({ x: x * 0.32, y: y * 0.32 }),
      convertViewPointToCameraPoint: ({ x, y }: { x: number; y: number }) => ({ x: x / 0.32, y: y / 0.32 }),
    } as CameraRef },
    model: sharedModel ?? {} as TensorflowModel, thresholds: DEFAULT_WORD_MODEL_THRESHOLDS,
    acquireModel: jest.fn(() => release), onStatus: jest.fn(), onError: jest.fn(),
  };
  const hook = await renderHook((options: typeof props) => useWordCameraTracking(options), { initialProps: props });
  return { hook, props, release };
}
async function tick(ms: number) {
  await act(() => { mockNow += ms; jest.advanceTimersByTime(ms); });
}
async function frame() {
  const next = makeFrame();
  await act(() => { mockOnFrame(next); });
  return next;
}
async function acquire() {
  await frame(); await tick(200);
  await frame(); await tick(100);
}

beforeEach(() => {
  jest.useFakeTimers({ doNotFake: ['performance'] });
  jest.spyOn(performance, 'now').mockImplementation(() => mockNow);
  mockNow = 0;
  mockEnvironment = 'production';
  mockBooleanFlags.length = 0;
  mockOrientation = 'up';
  mockDetect.mockReturnValue({ status: 'success', groups: [detected()], context: { x: 0, y: 0.3, width: 1, height: 0.4 } });
});
afterEach(async () => {
  await act(() => { jest.runOnlyPendingTimers(); });
  jest.restoreAllMocks();
  jest.useRealTimers();
});

describe('word camera tracking lifecycle and frame ownership', () => {
  it('confirms a candidate in a focused context and keeps the native output resolution object stable', async () => {
    const { hook, props } = await setup();
    const resolution = mockOutputOptions.targetResolution;
    await acquire();
    expect(hook.result.current.status).toBe('ready');
    expect(props.onStatus).toHaveBeenLastCalledWith('ready');
    expect((mockDetect.mock.calls[0]![3] as { width: number }).width).toBe(1);
    expect((mockDetect.mock.calls[1]![3] as { width: number }).width).toBeLessThan(1);
    expect(mockOutputOptions.targetResolution).toBe(resolution);
    expect(mockSource.dispose).toHaveBeenCalledTimes(2);
  });

  it('processes directly on the frame thread at most5Hz and disposes throttled frames', async () => {
    await setup();
    const first = await frame();
    expect(mockDetect).toHaveBeenCalledTimes(1);
    expect(first.dispose).toHaveBeenCalledTimes(1);
    expect(mockOutputOptions.dropFramesWhileBusy).toBe(true);
    expect(mockOutputOptions.enablePhysicalBufferRotation).toBe(false);
    await tick(100);
    const throttled = await frame();
    expect(mockDetect).toHaveBeenCalledTimes(1);
    expect(throttled.dispose).toHaveBeenCalledTimes(1);
    await tick(100);
    await frame();
    expect(mockDetect).toHaveBeenCalledTimes(2);
  });

  it('drops an overlapping callback without clearing ownership of its still-running frame', async () => {
    await setup();
    const callback = mockOnFrame;
    const overlapping = makeFrame();
    mockDetect.mockImplementationOnce(() => {
      callback(overlapping);
      expect(mockBooleanFlags[0]!.getBlocking()).toBe(true);
      expect(mockBooleanFlags[1]!.getBlocking()).toBe(true);
      expect(mockBooleanFlags[1]!.isLocked()).toBe(false);
      return { status: 'success', groups: [detected()] };
    });
    await frame();
    expect(mockDetect).toHaveBeenCalledTimes(1);
    expect(overlapping.dispose).toHaveBeenCalledTimes(1);
    expect(mockBooleanFlags[0]!.getBlocking()).toBe(false);
    expect(mockBooleanFlags[1]!.getBlocking()).toBe(false);
  });

  it('reports failed mutex admission safely and releases its lease', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { hook, props, release } = await setup();
    mockBooleanFlags[1]!.lock.mockImplementationOnce(() => { throw new Error('Mutex unavailable'); });
    const failed = await frame();
    await tick(100);
    expect(props.onError).toHaveBeenCalledTimes(1);
    expect(failed.dispose).toHaveBeenCalledTimes(1);
    const report = info.mock.calls.find(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(JSON.parse(report![1] as string)).toMatchObject({ stage: 'admission', code: 'unknown', errorName: 'Error' });
    await hook.unmount();
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('turns detector errors into one recoverable scanner error and releases both resources', async () => {
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    mockDetect.mockReturnValue({ status: 'error', reason: 'processing_error', groups: [] });
    const failed = await frame(); await tick(100);
    expect(props.onError).toHaveBeenCalledTimes(1);
    expect(mockSource.dispose).toHaveBeenCalledTimes(1);
    expect(failed.dispose).toHaveBeenCalledTimes(1);
    await tick(200);
    const ignored = await frame();
    expect(ignored.dispose).toHaveBeenCalledTimes(1);
    expect(mockDetect).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });

  it('reports conversion failure safely in preview and can acquire after the scanner retries', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { hook, props } = await setup();
    const privateMessage = 'PRIVATE_SCAN_ABC C:\\private\\capture.jpg coordinates=1,2 pixels=123';
    mockConvert.mockImplementationOnce(() => { throw new TypeError(privateMessage); });
    const failed = await frame(); await tick(100);
    const reports = info.mock.calls.filter(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(reports).toHaveLength(1);
    expect(JSON.parse(reports[0]![1] as string)).toEqual({
      stage: 'frame-conversion', code: 'unknown', errorName: 'TypeError',
      frameWidth: 1000, frameHeight: 1000, frameOrientation: 'up',
    });
    expect(JSON.stringify(info.mock.calls)).not.toContain(privateMessage);
    expect(props.onError).toHaveBeenCalledTimes(1);
    expect(failed.dispose).toHaveBeenCalledTimes(1);
    expect(mockSource.dispose).not.toHaveBeenCalled();
    await hook.rerender({ ...props, sessionId: 'session-retry' });
    await tick(100); await acquire();
    expect(hook.result.current.status).toBe('ready');
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('classifies an actual frame transform error without exposing invalid coordinates', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    const invalid = makeFrame();
    invalid.convertCameraPointToFramePoint = () => ({ x: NaN, y: 123456789 });
    await act(() => { mockOnFrame(invalid); });
    await tick(100);
    const report = info.mock.calls.find(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(JSON.parse(report![1] as string)).toEqual({
      stage: 'frame-transform', code: 'invalid_frame_coordinates', errorName: 'Error',
      frameWidth: 1000, frameHeight: 1000, frameOrientation: 'up',
    });
    expect(JSON.stringify(info.mock.calls)).not.toContain('123456789');
    expect(mockConvert).not.toHaveBeenCalled();
    expect(invalid.dispose).toHaveBeenCalledTimes(1);
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('identifies detector failures using only the fixed detection code', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    mockDetect.mockReturnValueOnce({ status: 'error', reason: 'processing_error', groups: [], text: 'PRIVATE_SCAN_TEXT' });
    const failed = await frame(); await tick(100);
    const report = info.mock.calls.find(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(JSON.parse(report![1] as string)).toMatchObject({ stage: 'cell-detection', code: 'cell_detection_failed', errorName: 'Error' });
    expect(JSON.stringify(info.mock.calls)).not.toContain('PRIVATE_SCAN_TEXT');
    expect(mockSource.dispose).toHaveBeenCalledTimes(1);
    expect(failed.dispose).toHaveBeenCalledTimes(1);
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('distinguishes a failed sensor box transform after detection', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    const invalid = makeFrame();
    invalid.convertFramePointToCameraPoint = () => ({ x: NaN, y: NaN });
    await act(() => { mockOnFrame(invalid); });
    await tick(100);
    const report = info.mock.calls.find(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(JSON.parse(report![1] as string)).toMatchObject({ stage: 'sensor-boxes', code: 'invalid_sensor_coordinates', errorName: 'Error' });
    expect(mockDetect).toHaveBeenCalledTimes(1);
    expect(mockSource.dispose).toHaveBeenCalledTimes(1);
    expect(invalid.dispose).toHaveBeenCalledTimes(1);
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('omits unknown error names and arbitrary thrown properties from diagnostics', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    mockConvert.mockImplementationOnce(() => { throw { name: 'PRIVATE_ERROR_NAME', message: 'PRIVATE_ERROR_MESSAGE', path: 'PRIVATE_PATH' }; });
    await frame(); await tick(100);
    const report = info.mock.calls.find(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(JSON.parse(report![1] as string)).toEqual({ stage: 'frame-conversion', code: 'unknown', frameWidth: 1000, frameHeight: 1000, frameOrientation: 'up' });
    expect(JSON.stringify(info.mock.calls)).not.toContain('PRIVATE_');
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('preserves the scanner retry callback if the diagnostic logger throws', async () => {
    mockEnvironment = 'preview';
    jest.spyOn(console, 'info').mockImplementation(() => { throw new Error('Logger unavailable'); });
    const { props } = await setup();
    mockConvert.mockImplementationOnce(() => { throw new ReferenceError('Private native error'); });
    const failed = await frame(); await tick(100);
    expect(props.onError).toHaveBeenCalledTimes(1);
    expect(failed.dispose).toHaveBeenCalledTimes(1);
  });

  it('marks stale targets searching after500ms and clears their boxes after800ms', async () => {
    const { hook } = await setup();
    await acquire(); // last frame at200, poll at300
    await tick(450); // age550
    expect(hook.result.current.status).toBe('searching');
    expect(hook.result.current.display).toBeDefined();
    await act(() => { expect(hook.result.current.snapshot()).toBeUndefined(); });
    await tick(300);
    expect(hook.result.current.display).toBeUndefined();
  });

  it('snapshots raw sensor corners once and pauses new tracking work', async () => {
    const { hook } = await setup();
    await acquire();
    let snapshot: ReturnType<typeof hook.result.current.snapshot>;
    await act(() => { snapshot = hook.result.current.snapshot(); });
    expect(snapshot!.points[0]).toEqual({ x: 400, y: 450 });
    expect(hook.result.current.isCurrent(snapshot!)).toBe(true);
    await act(() => { expect(hook.result.current.snapshot()).toBeUndefined(); });
    await tick(200);
    const ignored = await frame();
    expect(ignored.dispose).toHaveBeenCalledTimes(1);
    expect(mockDetect).toHaveBeenCalledTimes(2);
  });

  it('consumes a completed fresh frame at the shutter before the next UI poll', async () => {
    const { hook } = await setup();
    await acquire(); // displayed ready at300; original observation timestamp200
    mockNow = 751; // the old capture has expired, but the UI has not polled again
    mockDetect.mockReturnValueOnce({ status: 'success', groups: [detected(0.41)] });
    await frame(); // timestamp751 is completed in the channel, not yet polled
    expect(hook.result.current.status).toBe('ready');
    let snapshot: ReturnType<typeof hook.result.current.snapshot>;
    await act(() => { snapshot = hook.result.current.snapshot(); });
    expect(snapshot?.timestampMs).toBe(751);
    expect(snapshot?.points).toHaveLength(4);
    for (const [index, expected] of [[410, 450], [560, 450], [560, 550], [410, 550]].entries()) {
      expect(snapshot!.points[index]!.x).toBeCloseTo(expected[0]!);
      expect(snapshot!.points[index]!.y).toBeCloseTo(expected[1]!);
    }
    expect(mockDetect).toHaveBeenCalledTimes(3); // refresh consumes only; no new inference
    await act(() => { expect(hook.result.current.snapshot()).toBeUndefined(); });
  });

  it('rejects a shutter after500ms when no newer completed observation exists', async () => {
    const { hook } = await setup();
    await acquire();
    mockNow = 751;
    let snapshot: ReturnType<typeof hook.result.current.snapshot>;
    await act(() => { snapshot = hook.result.current.snapshot(); });
    expect(snapshot).toBeUndefined();
    expect(hook.result.current.status).toBe('searching');
    expect(mockDetect).toHaveBeenCalledTimes(2);
  });

  it('keeps the500ms capture age limit for an unpolled but already stale frame', async () => {
    const { hook } = await setup();
    await acquire();
    mockNow = 750;
    await frame();
    mockNow = 1251;
    let snapshot: ReturnType<typeof hook.result.current.snapshot>;
    await act(() => { snapshot = hook.result.current.snapshot(); });
    expect(snapshot).toBeUndefined();
    expect(hook.result.current.status).toBe('searching');
  });

  it('rejects an unpolled observation when native viewport geometry changed before the shutter', async () => {
    const { hook, props } = await setup();
    await acquire();
    mockNow = 400;
    await frame();
    props.camera.current.convertViewPointToCameraPoint = ({ x, y }) => ({ x: x / 0.32 + 10, y: y / 0.32 });
    let snapshot: ReturnType<typeof hook.result.current.snapshot>;
    await act(() => { snapshot = hook.result.current.snapshot(); });
    expect(snapshot).toBeUndefined();
    expect(hook.result.current.display).toBeUndefined();
    expect(hook.result.current.status).toBe('searching');
    expect(props.onError).not.toHaveBeenCalled();
  });

  it('does not refresh native geometry through a retained snapshot callback after cleanup', async () => {
    const { hook, props } = await setup();
    await acquire();
    const snapshot = hook.result.current.snapshot;
    const convert = jest.spyOn(props.camera.current, 'convertViewPointToCameraPoint');
    await hook.unmount();
    const calls = convert.mock.calls.length;
    expect(snapshot()).toBeUndefined();
    expect(convert).toHaveBeenCalledTimes(calls);
  });

  it('keeps shutter identity when the photo deactivates tracking but invalidates layout/session changes', async () => {
    const { hook, props } = await setup();
    await acquire();
    let snapshot!: NonNullable<ReturnType<typeof hook.result.current.snapshot>>;
    await act(() => { snapshot = hook.result.current.snapshot()!; });
    await hook.rerender({ ...props, active: false });
    expect(hook.result.current.isCurrent(snapshot)).toBe(true);
    await hook.rerender({ ...props, active: false, layout: { width: 321, height: 320 } });
    expect(hook.result.current.isCurrent(snapshot)).toBe(false);
    await hook.rerender({ ...props, active: true, sessionId: 'session-b' });
    expect(hook.result.current.isCurrent(snapshot)).toBe(false);
    await hook.unmount();
    expect(hook.result.current.isCurrent(snapshot)).toBe(false);
  });

  it('drops a completed old-session message before publishing its boxes', async () => {
    const { hook, props } = await setup();
    const old = await frame();
    await hook.rerender({ ...props, sessionId: 'session-b' });
    await tick(100);
    expect(old.dispose).toHaveBeenCalledTimes(1);
    expect(mockDetect).toHaveBeenCalledTimes(1);
    expect(hook.result.current.status).toBe('searching');
    expect(hook.result.current.display).toBeUndefined();
    await act(() => { expect(hook.result.current.snapshot()).toBeUndefined(); });
    expect(props.onError).not.toHaveBeenCalled();
  });

  it('waits for native frame ownership to settle before releasing the model on unmount', async () => {
    const { hook, release } = await setup();
    // Native work runs concurrently with JS. Reproduce the synchronizable state
    // of an admitted frame while exercising the actual JS cleanup timer.
    mockBooleanFlags[0]!.setBlocking(true);
    mockBooleanFlags[1]!.setBlocking(true);
    await hook.unmount();
    await tick(40);
    expect(release).not.toHaveBeenCalled();
    mockBooleanFlags[1]!.setBlocking(false);
    mockBooleanFlags[0]!.setBlocking(false);
    await tick(20);
    expect(release).toHaveBeenCalledTimes(1);
  });

  it.each([
    [0.5, 'preview_outside_subpixel'],
    [7.123456, 'preview_outside_small'],
    [123.456789, 'preview_outside_large'],
  ])('classifies an overshoot of %s pixels using a fixed failure code', async (overshoot, code) => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    const invalid = makeFrame();
    invalid.convertCameraPointToFramePoint = ({ x, y }) => ({ x: x - Number(overshoot), y });
    await act(() => { mockOnFrame(invalid); });
    await tick(100);
    const reports = info.mock.calls.filter(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(reports).toHaveLength(1);
    expect(JSON.parse(reports[0]![1] as string)).toEqual({
      stage: 'frame-transform', code, errorName: 'Error',
      frameWidth: 1000, frameHeight: 1000, frameOrientation: 'up',
    });
    expect(JSON.stringify(reports)).not.toContain(String(overshoot));
    expect(mockConvert).not.toHaveBeenCalled();
    expect(mockDetect).not.toHaveBeenCalled();
    expect(invalid.dispose).toHaveBeenCalledTimes(1);
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('records four native corners and upright bounds once for a failed preview transform', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    const invalid = makeFrame();
    invalid.width = 480;
    invalid.height = 640;
    invalid.orientation = 'right';
    invalid.isMirrored = true;
    const convert = jest.fn(({ x, y }: { x: number; y: number }) => ({ x: x / 2 - 240, y: y / 2, privateText: 'PRIVATE_SCAN' }));
    invalid.convertCameraPointToFramePoint = convert;
    await act(() => { mockOnFrame(invalid); });
    await tick(100);
    const reports = info.mock.calls.filter(([tag]) => tag === 'BRAILLE_WORD_TRACKING_GEOMETRY');
    expect(reports).toHaveLength(1);
    const sensorPoints = convert.mock.calls.slice(0, 4).map(([point]) => point);
    const rawFramePoints = sensorPoints.map(({ x, y }) => ({ x: x / 2 - 240, y: y / 2 }));
    const upright = rawFramePoints.map(({ x, y }) => ({ x: 640 - y, y: 480 - x }));
    expect(JSON.parse(reports[0]![1] as string)).toEqual({
      schemaVersion: 1, sequence: 1, sensorPoints, rawFramePoints,
      frame: { width: 480, height: 640, orientation: 'right', isMirrored: true },
      upright: { width: 640, height: 480, bounds: {
        left: Math.min(...upright.map(({ x }) => x)), top: Math.min(...upright.map(({ y }) => y)),
        right: Math.max(...upright.map(({ x }) => x)), bottom: Math.max(...upright.map(({ y }) => y)),
      } },
    });
    expect(convert).toHaveBeenCalledTimes(8);
    expect(JSON.stringify(info.mock.calls)).not.toContain('PRIVATE_SCAN');
    expect(invalid.dispose).toHaveBeenCalledTimes(1);
    expect(props.onError).toHaveBeenCalledTimes(1);
    await tick(400);
    await frame();
    expect(info.mock.calls.filter(([tag]) => tag === 'BRAILLE_WORD_TRACKING_GEOMETRY')).toHaveLength(1);
  });

  it('does not remeasure or log failed transform geometry in production', async () => {
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    const invalid = makeFrame();
    const convert = jest.fn(({ x, y }: { x: number; y: number }) => ({ x: x - 200, y }));
    invalid.convertCameraPointToFramePoint = convert;
    await act(() => { mockOnFrame(invalid); });
    await tick(100);
    expect(convert).toHaveBeenCalledTimes(4);
    expect(info).not.toHaveBeenCalled();
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('preserves the original transform failure when diagnostic native conversion throws', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    const { props } = await setup();
    const invalid = makeFrame();
    let calls = 0;
    invalid.convertCameraPointToFramePoint = ({ x, y }) => {
      if (++calls > 4) throw new Error('PRIVATE_NATIVE_DIAGNOSTIC_FAILURE');
      return { x: x - 200, y };
    };
    await act(() => { mockOnFrame(invalid); });
    await tick(100);
    expect(info.mock.calls.filter(([tag]) => tag === 'BRAILLE_WORD_TRACKING_GEOMETRY')).toHaveLength(0);
    const report = info.mock.calls.find(([tag]) => tag === 'BRAILLE_WORD_TRACKING_FAILURE');
    expect(JSON.parse(report![1] as string)).toMatchObject({ stage: 'frame-transform', code: 'preview_outside_large' });
    expect(JSON.stringify(info.mock.calls)).not.toContain('PRIVATE_NATIVE');
    expect(invalid.dispose).toHaveBeenCalledTimes(1);
    expect(props.onError).toHaveBeenCalledTimes(1);
  });

  it('atomically disables frame admission before an idle unmount releases the model', async () => {
    const { hook, release } = await setup();
    const callback = mockOnFrame;
    const ownership = mockBooleanFlags[0]!;
    await hook.unmount();
    expect(ownership.lock).toHaveBeenCalledTimes(1);
    expect(ownership.unlock).toHaveBeenCalledTimes(1);
    expect(release).toHaveBeenCalledTimes(1);
    const lateFrame = makeFrame();
    callback(lateFrame);
    expect(mockDetect).not.toHaveBeenCalled();
    expect(lateFrame.dispose).toHaveBeenCalledTimes(1);
    expect(ownership.getBlocking()).toBe(false);
  });

  it('still disposes the frame and releases its lease if bitmap cleanup throws', async () => {
    const { hook, release } = await setup();
    mockSource.dispose.mockImplementationOnce(() => { throw new Error('Cleanup failed'); });
    const pending = await frame();
    expect(pending.dispose).toHaveBeenCalledTimes(1);
    await hook.unmount();
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('logs only numeric aggregate telemetry in preview and never logs detection contents', async () => {
    mockEnvironment = 'preview';
    const info = jest.spyOn(console, 'info').mockImplementation(() => undefined);
    await setup(); await acquire(); await tick(10_000);
    const report = info.mock.calls.find(([tag]) => tag === 'BRAILLE_WORD_TRACKING_METRICS');
    expect(report).toBeDefined();
    const metrics = JSON.parse(report![1] as string);
    expect(metrics).toMatchObject({ admitted: 2, completed: 2, errors: 0 });
    expect(metrics.latencyBuckets).toHaveLength(7);
    expect(Object.keys(metrics)).not.toEqual(expect.arrayContaining(['text', 'groups', 'points', 'boxes', 'image', 'sessionId']));
  });

  it('invalidates tracking on orientation changes even when the viewport stays square', async () => {
    const { hook, props } = await setup();
    await acquire();
    let snapshot!: NonNullable<ReturnType<typeof hook.result.current.snapshot>>;
    await act(() => { snapshot = hook.result.current.snapshot()!; });
    mockOrientation = 'right';
    await hook.rerender({ ...props });
    expect(hook.result.current.status).toBe('searching');
    expect(hook.result.current.display).toBeUndefined();
    expect(hook.result.current.isCurrent(snapshot)).toBe(false);
  });

  it('discards a completed frame after the native preview crop changes', async () => {
    const { hook, props } = await setup();
    await acquire();
    await tick(100);
    const stale = await frame();
    props.camera.current.convertViewPointToCameraPoint = ({ x, y }) => ({ x: x / 0.32 + 10, y: y / 0.32 });
    await tick(100);
    expect(hook.result.current.status).toBe('searching');
    expect(hook.result.current.display).toBeUndefined();
   
    expect(stale.dispose).toHaveBeenCalledTimes(1);
    expect(mockDetect).toHaveBeenCalledTimes(3);
  });

  it('checks the still-pending shutter viewport even after new frame admission has paused', async () => {
    const { hook, props } = await setup();
    await acquire();
    let snapshot!: NonNullable<ReturnType<typeof hook.result.current.snapshot>>;
    await act(() => { snapshot = hook.result.current.snapshot()!; });
    props.camera.current.convertViewPointToCameraPoint = ({ x, y }) => ({ x: x / 0.32 + 10, y: y / 0.32 });
    expect(hook.result.current.isCurrent(snapshot)).toBe(false);
  });

  it('atomically shares model ownership across simultaneous session runtimes', async () => {
    const first = await setup();
    const firstCallback = mockOnFrame;
    const second = await setup(first.props.model);
    const secondCallback = mockOnFrame;
    const dropped = makeFrame();
    const shared = mockBooleanFlags[1]!;
    mockDetect.mockImplementationOnce(() => {
      expect(shared.getBlocking()).toBe(true);
      expect(shared.isLocked()).toBe(false);
      secondCallback(dropped);
      expect(shared.getBlocking()).toBe(true);
      expect(shared.isLocked()).toBe(false);
      return { status: 'success', groups: [detected()] };
    });
    const firstFrame = makeFrame();
    await act(() => { firstCallback(firstFrame); });
    expect(mockDetect).toHaveBeenCalledTimes(1);
    expect(dropped.dispose).toHaveBeenCalledTimes(1);
    expect(firstFrame.dispose).toHaveBeenCalledTimes(1);
    expect(shared.lock).toHaveBeenCalledTimes(2);
    expect(shared.unlock).toHaveBeenCalledTimes(2);
    await act(() => { secondCallback(makeFrame()); });
    expect(mockDetect).toHaveBeenCalledTimes(2);
    await first.hook.unmount();
    await second.hook.unmount();
  });

  it('drops work after remount while the previous native session still owns the interpreter', async () => {
    const first = await setup();
    const oldOwned = mockBooleanFlags[0]!;
    const shared = mockBooleanFlags[1]!;
    oldOwned.setBlocking(true);
    shared.setBlocking(true);
    await first.hook.unmount();
    const second = await setup(first.props.model);
    const dropped = await frame();
    expect(dropped.dispose).toHaveBeenCalledTimes(1);
    expect(mockDetect).not.toHaveBeenCalled();
    expect(shared.getBlocking()).toBe(true);
    expect(first.release).not.toHaveBeenCalled();
    shared.setBlocking(false);
    oldOwned.setBlocking(false);
    await tick(20);
    expect(first.release).toHaveBeenCalledTimes(1);
    await frame();
    expect(mockDetect).toHaveBeenCalledTimes(1);
    await second.hook.unmount();
  });
});
