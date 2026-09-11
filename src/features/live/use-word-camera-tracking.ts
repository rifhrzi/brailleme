import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from 'react';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { HybridFrameConverter, useFrameOutput, useOrientation, type CameraRef } from 'react-native-vision-camera';
import { createSynchronizable, type Synchronizable } from 'react-native-worklets';
import { APP_ENVIRONMENT } from '@/config/app-environment';

import {
  mapWordSensorBoxToFrame, mapWordSensorBoxToPreview,
  snapshotWordFrameBoxSensorPoints, snapshotWordGuideSensorPoints, type SensorPoint,
} from '@/ml/word-camera-geometry';
import { detectWordImageCells, type WordPixelProcessor } from '@/ml/word-image-processing';
import type { WordModelThresholds } from '@/ml/word-model-contract';
import { isNormalizedWordBox } from '@/types/word-scan';
import {
  advanceWordTracking, getWordSearchWindows, getWordTrackingContext,
  getWordTrackingTarget, INITIAL_WORD_TRACKING_STATE, WORD_TRACKING_CAPTURE_AGE_MS, WORD_TRACKING_INTERVAL_MS, type WordTrackingGroup,
} from '@/ml/word-tracking';

type SensorGroup = {
  points: readonly SensorPoint[];
  cells: readonly (readonly SensorPoint[])[];
  score: number;
  complete: boolean;
};
type Controls = {
  enabled: boolean;
  generation: number;
  windows: readonly (readonly SensorPoint[])[];
  target: readonly SensorPoint[] | null;
  confirming: boolean;
};
type TrackingFailureStage = 'admission' | 'frame-transform' | 'frame-conversion' | 'cell-detection' | 'sensor-boxes' | 'publish' | 'preview-geometry';
type TrackingFailure = {
  stage: TrackingFailureStage;
  code: 'unknown' | 'invalid_frame_geometry' | 'invalid_frame_orientation' | 'invalid_sensor_coordinates' | 'invalid_frame_coordinates' | 'preview_outside_frame' | 'preview_outside_subpixel' | 'preview_outside_small' | 'preview_outside_large' | 'invalid_preview_geometry' | 'cell_detection_failed';
  errorName?: 'Error' | 'TypeError' | 'RangeError' | 'ReferenceError';
  frameWidth?: number;
  frameHeight?: number;
  frameOrientation?: 'up' | 'right' | 'down' | 'left';
};
type TrackingFailureGeometry = {
  sensorPoints: readonly SensorPoint[];
  rawFramePoints: readonly SensorPoint[];
  frame: { width: number; height: number; orientation: 'up' | 'right' | 'down' | 'left'; isMirrored: boolean };
  upright: { width: number; height: number; bounds: { left: number; top: number; right: number; bottom: number } };
};
type Message = {
  sequence: number;
  generation: number;
  timestampMs: number;
  durationMs: number;
  groups: readonly SensorGroup[];
  error: boolean;
  failure?: TrackingFailure;
  failureGeometry?: TrackingFailureGeometry;
};
export type TrackingStatus = 'searching' | 'detected' | 'ready';
export type TrackedCapture = {
  readonly points: readonly SensorPoint[];
  readonly timestampMs: number;
  readonly generation: number;
  readonly geometryGeneration: number;
  readonly viewportPoints: readonly SensorPoint[];
};

type Options = {
  active: boolean;
  sessionId: string;
  layout: { width: number; height: number };
  camera: RefObject<CameraRef | null>;
  model: TensorflowModel;
  thresholds: WordModelThresholds;
  acquireModel: () => () => void;
  pixelProcessor?: WordPixelProcessor;
  onStatus: (status: TrackingStatus) => void;
  onError: () => void;
};

// Native useFrameOutput memoizes by object identity, not resolution values.
const TRACKING_RESOLUTION = { width: 640, height: 480 };
const TRACKING_VIEWPORT = { x: 0, y: 0, width: 1, height: 1 };
// Separate camera sessions can overlap while the previous frame finishes.
// The short atomic claim is shared by every borrower of this interpreter.
const modelBusyFlags = new WeakMap<TensorflowModel, Synchronizable<boolean>>();

function trackingFailure(stage: TrackingFailureStage, error: unknown, frame?: { width: number; height: number; orientation: unknown }): TrackingFailure {
  'worklet';
  const failure: TrackingFailure = { stage, code: 'unknown' };
  // Only fixed codes cross runtimes. Never retain a native message, stack or path.
  try {
    if (typeof error === 'object' && error !== null) {
      const { name, message } = error as { name?: unknown; message?: unknown };
      if (name === 'Error' || name === 'TypeError' || name === 'RangeError' || name === 'ReferenceError') failure.errorName = name;
      switch (message) {
        case 'Geometri frame tidak valid.': failure.code = 'invalid_frame_geometry'; break;
        case 'Orientasi frame tidak valid.': failure.code = 'invalid_frame_orientation'; break;
        case 'Koordinat sensor frame tidak valid.':
        case 'Koordinat sensor tidak valid.': failure.code = 'invalid_sensor_coordinates'; break;
        case 'Koordinat frame tidak valid.': failure.code = 'invalid_frame_coordinates'; break;
        case 'Area pratinjau tidak seluruhnya tersedia pada frame.': failure.code = 'preview_outside_frame'; break;
        case '[word-frame:outside-subpixel]': failure.code = 'preview_outside_subpixel'; break;
        case '[word-frame:outside-small]': failure.code = 'preview_outside_small'; break;
        case '[word-frame:outside-large]': failure.code = 'preview_outside_large'; break;
        case 'Area panduan kamera belum sesuai dengan foto. Silakan pindai ulang.': failure.code = 'invalid_preview_geometry'; break;
        case 'Frame inference failed': failure.code = 'cell_detection_failed'; break;
      }
    }
  } catch { /* Diagnostic property access must not replace the original failure. */ }
  try {
    if (frame) {
      if (Number.isSafeInteger(frame.width) && frame.width > 0) failure.frameWidth = frame.width;
      if (Number.isSafeInteger(frame.height) && frame.height > 0) failure.frameHeight = frame.height;
      const orientation = frame.orientation;
      if (orientation === 'up' || orientation === 'right' || orientation === 'down' || orientation === 'left') failure.frameOrientation = orientation;
    }
  } catch { /* A released frame may no longer expose metadata. */ }
  return failure;
}

function logTrackingFailure(failure: TrackingFailure) {
  if (APP_ENVIRONMENT !== 'preview') return;
  try {
    console.info('BRAILLE_WORD_TRACKING_FAILURE', JSON.stringify({
      stage: failure.stage, code: failure.code, errorName: failure.errorName,
      frameWidth: failure.frameWidth, frameHeight: failure.frameHeight, frameOrientation: failure.frameOrientation,
    }));
  } catch { /* Diagnostics must not prevent the existing retry path. */ }
}

function snapshotTrackingFailureGeometry(points: readonly SensorPoint[], frame: Parameters<typeof mapWordSensorBoxToFrame>[1]): TrackingFailureGeometry | undefined {
  'worklet';
  // Re-measure only a failed preview transform, while its native frame is alive.
  // Return numeric geometry only, never native objects or arbitrary properties.
  try {
    const { width, height, orientation, isMirrored } = frame;
    if (points.length !== 4 || ![width, height].every((value) => Number.isSafeInteger(value) && value > 0)
      || !['up', 'right', 'down', 'left'].includes(orientation) || typeof isMirrored !== 'boolean') return;
    const sensorPoints = points.map(({ x, y }) => ({ x, y }));
    if (!sensorPoints.every(({ x, y }) => Number.isFinite(x) && Number.isFinite(y))) return;
    const rawFramePoints = sensorPoints.map((point) => {
      const raw = frame.convertCameraPointToFramePoint(point);
      return { x: raw.x, y: raw.y };
    });
    if (!rawFramePoints.every(({ x, y }) => Number.isFinite(x) && Number.isFinite(y))) return;
    const upright = rawFramePoints.map((raw) => {
      const x = isMirrored ? width - raw.x : raw.x;
      const y = raw.y;
      switch (orientation) {
        case 'right': return { x: height - y, y: x };
        case 'down': return { x: width - x, y: height - y };
        case 'left': return { x: y, y: width - x };
        default: return { x, y };
      }
    });
    const quarterTurn = orientation === 'right' || orientation === 'left';
    return {
      sensorPoints, rawFramePoints, frame: { width, height, orientation, isMirrored },
      upright: { width: quarterTurn ? height : width, height: quarterTurn ? width : height,
        bounds: { left: Math.min(...upright.map(({ x }) => x)), top: Math.min(...upright.map(({ y }) => y)),
          right: Math.max(...upright.map(({ x }) => x)), bottom: Math.max(...upright.map(({ y }) => y)) } },
    };
  } catch { return; }
}

function logTrackingFailureGeometry(geometry: TrackingFailureGeometry | undefined, sequence: number) {
  if (APP_ENVIRONMENT !== 'preview' || !geometry) return;
  try {
    const { frame, upright } = geometry;
    console.info('BRAILLE_WORD_TRACKING_GEOMETRY', JSON.stringify({
      schemaVersion: 1, sequence,
      sensorPoints: geometry.sensorPoints.map(({ x, y }) => ({ x, y })),
      rawFramePoints: geometry.rawFramePoints.map(({ x, y }) => ({ x, y })),
      frame: { width: frame.width, height: frame.height, orientation: frame.orientation, isMirrored: frame.isMirrored },
      upright: { width: upright.width, height: upright.height, bounds: {
        left: upright.bounds.left, top: upright.bounds.top, right: upright.bounds.right, bottom: upright.bounds.bottom,
      } },
    }));
  } catch { /* Geometry diagnostics must not prevent the existing retry path. */ }
}

function sameViewport(a: readonly SensorPoint[], b: readonly SensorPoint[]): boolean {
  const scale = Math.max(1, Math.max(...a.map((point) => point.x)) - Math.min(...a.map((point) => point.x)), Math.max(...a.map((point) => point.y)) - Math.min(...a.map((point) => point.y)));
  return a.length === b.length && a.every((point, index) => Math.abs(point.x - b[index]!.x) <= scale * 0.0001 && Math.abs(point.y - b[index]!.y) <= scale * 0.0001);
}

/** Sends only box coordinates across runtimes. Camera frames and pixels stay transient. */
export function useWordCameraTracking(options: Options) {
  const { active, sessionId, layout, camera, model, thresholds, acquireModel, pixelProcessor } = options;
  const enableGeometryDiagnostics = APP_ENVIRONMENT === 'preview';
  const callbacks = useRef(options);
  useLayoutEffect(() => { callbacks.current = options; }, [options]);
  const controls = useMemo(() => createSynchronizable<Controls>({ enabled: false, generation: 0, windows: [], target: null, confirming: false }), []);
  const busy = useMemo(() => createSynchronizable(false), []);
  const admission = useMemo(() => createSynchronizable({ time: -Infinity, tick: 0, window: 0, sequence: 0, received: 0, busyDrops: 0, throttled: 0, disabledDrops: 0, runnerDrops: 0 }), []);
  const metrics = useMemo(() => createSynchronizable({ completed: 0, errors: 0, totalMs: 0, maxMs: 0, latencyBuckets: [0, 0, 0, 0, 0, 0, 0] }), []);
  const channel = useMemo(() => createSynchronizable<Message>({ sequence: 0, generation: 0, timestampMs: 0, durationMs: 0, groups: [], error: false }), []);
  const modelBusy = useMemo(() => {
    const existing = modelBusyFlags.get(model);
    if (existing) return existing;
    const flag = createSynchronizable(false);
    modelBusyFlags.set(model, flag);
    return flag;
  }, [model]);
  const orientation = useOrientation('interface');
  const generation = useRef(0);
  const state = useRef(INITIAL_WORD_TRACKING_STATE);
  const capture = useRef<TrackedCapture | undefined>(undefined);
  // The shutter may run between periodic polls after a newer frame completed.
  // Reuse the same guarded channel consumer rather than accepting an older box.
  const refreshCompletedObservation = useRef<(() => void) | undefined>(undefined);
  const stopped = useRef(false);
  const mounted = useRef(true);
  const geometry = useRef(0);
  useLayoutEffect(() => { geometry.current += 1; }, [sessionId, layout.width, layout.height, camera, model, orientation]);
  const viewKey = useMemo(() => ({ active, sessionId, width: layout.width, height: layout.height, camera, model, thresholds, orientation }), [active, sessionId, layout.width, layout.height, camera, model, thresholds, orientation]);
  const [view, setView] = useState<{ key: object; display: WordTrackingGroup | undefined; status: TrackingStatus }>({ key: viewKey, display: undefined, status: 'searching' });
  const display = active && view.key === viewKey ? view.display : undefined;
  const status: TrackingStatus = active && view.key === viewKey ? view.status : 'searching';

  useEffect(() => {
    mounted.current = true;
    const release = acquireModel();
    return () => {
      mounted.current = false;
      // Atomic with frame admission: no frame may register ownership after
      // cleanup observes an idle lease and releases the interpreter.
      busy.lock();
      try { controls.setBlocking({ ...controls.getBlocking(), enabled: false }); }
      finally { busy.unlock(); }
      // Loader disposal must wait until the admitted native inference has settled.
      const finish = () => {
        if (busy.getBlocking()) setTimeout(finish, 20);
        else release();
      };
      finish();
    };
  }, [acquireModel, busy, controls]);

  useEffect(() => {
    if (APP_ENVIRONMENT !== 'preview' || !active) return;
    const startedAt = performance.now();
    const report = () => {
      const frames = admission.getBlocking();
      const timing = metrics.getBlocking();
      try {
        // Aggregate runtime measurements only. Never include scans, coordinates, images or paths.
        console.info('BRAILLE_WORD_TRACKING_METRICS', JSON.stringify({ schemaVersion: 1,
          elapsedMs: Math.round(performance.now() - startedAt), received: frames.received, admitted: frames.sequence,
          busyDrops: frames.busyDrops, throttled: frames.throttled, disabledDrops: frames.disabledDrops,
          // Kept at zero for the existing collector schema; no second runner is used.
          runnerDrops: frames.runnerDrops, completed: timing.completed, errors: timing.errors,
          meanMs: timing.completed ? Math.round(timing.totalMs / timing.completed) : 0,
          maxMs: Math.round(timing.maxMs), latencyUpperBoundsMs: [50, 100, 200, 400, 800, 1600, null], latencyBuckets: timing.latencyBuckets,
        }));
      } catch { /* Diagnostics must not affect camera lifecycle. */ }
    };
    const timer = setInterval(report, 10_000);
    return () => { clearInterval(timer); report(); };
  }, [active, admission, metrics]);

  useEffect(() => {
    let currentGeneration = ++generation.current;
    stopped.current = false;
    state.current = INITIAL_WORD_TRACKING_STATE;
    capture.current = undefined;
    callbacks.current.onStatus('searching');
    controls.setBlocking({ enabled: false, generation: currentGeneration, windows: [], target: null, confirming: false });
    if (!active || !layout.width || !layout.height) return;
    let lastSequence = channel.getBlocking().sequence;
    let lastStatus: TrackingStatus = 'searching';
    let failed = false;
    let viewport: readonly SensorPoint[] | undefined;
    const windows = getWordSearchWindows(layout.width, layout.height);
    const configure = () => {
      const now = performance.now();
      const currentViewport = snapshotWordGuideSensorPoints(camera.current, TRACKING_VIEWPORT, layout.width, layout.height);
      const target = getWordTrackingTarget(state.current, now);
      const pending = state.current.pending && state.current.pendingSeenMs !== undefined && now - state.current.pendingSeenMs <= WORD_TRACKING_CAPTURE_AGE_MS ? state.current.pending : undefined;
      // Confirm a new center candidate before resuming normal discovery/target alternation.
      const focus = pending ?? target?.raw;
      const targetContext = focus ? getWordTrackingContext(focus, layout.width, layout.height) : undefined;
      controls.setBlocking({
        enabled: true, generation: currentGeneration,
        windows: windows.map((window) => snapshotWordGuideSensorPoints(camera.current, window, layout.width, layout.height)),
        target: targetContext ? snapshotWordGuideSensorPoints(camera.current, targetContext, layout.width, layout.height) : null,
        confirming: pending !== undefined,
      });
      viewport = currentViewport;
    };
    const poll = () => {
      if (failed || stopped.current || generation.current !== currentGeneration || !camera.current) return;
      const now = performance.now();
      let failure: TrackingFailure | undefined;
      try {
        const currentViewport = snapshotWordGuideSensorPoints(camera.current, TRACKING_VIEWPORT, layout.width, layout.height);
        if (viewport && !sameViewport(viewport, currentViewport)) {
          // Preview crop/rotation may update after its layout/orientation event.
          // Discard earlier-frame geometry instead of remapping it into a new view.
          currentGeneration = ++generation.current;
          geometry.current += 1;
          state.current = INITIAL_WORD_TRACKING_STATE;
          capture.current = undefined;
        }
        const message = channel.getBlocking();
        if (message.generation === currentGeneration && message.sequence !== lastSequence) {
          lastSequence = message.sequence;
          if (message.error) {
            failure = message.failure;
            logTrackingFailureGeometry(message.failureGeometry, message.sequence);
            throw new Error('Frame tracking failed');
          }
          const groups: WordTrackingGroup[] = message.groups.map((group) => {
            const box = mapWordSensorBoxToPreview(group.points, camera.current!, layout.width, layout.height);
            const boxes = group.cells.map((points) => mapWordSensorBoxToPreview(points, camera.current!, layout.width, layout.height));
            return { box, boxes, score: group.score,
              complete: group.complete && isNormalizedWordBox(box) && boxes.every(isNormalizedWordBox) };
          });
          state.current = advanceWordTracking(state.current, groups, message.timestampMs);
          const observed = getWordTrackingTarget(state.current, now);
          if (observed && observed.lastSeenMs === message.timestampMs) {
            const index = groups.indexOf(observed.raw);
            if (index >= 0) capture.current = {
              points: message.groups[index]!.points.map((point) => ({ ...point })),
              timestampMs: message.timestampMs, generation: currentGeneration, geometryGeneration: geometry.current,
              viewportPoints: currentViewport.map((point) => ({ ...point })),
            };
          }
        }
        const target = getWordTrackingTarget(state.current, now);
        const fresh = target && now - target.lastSeenMs <= WORD_TRACKING_CAPTURE_AGE_MS;
        const nextStatus: TrackingStatus = target?.ready ? 'ready' : fresh ? 'detected' : 'searching';
        setView((previous) => previous.key === viewKey && previous.display === target?.display && previous.status === nextStatus
          ? previous : { key: viewKey, display: target?.display, status: nextStatus });
        if (nextStatus !== lastStatus) {
          lastStatus = nextStatus;
          callbacks.current.onStatus(nextStatus);
        }
        if (!target?.ready) capture.current = undefined;
        configure();
      } catch (error) {
        logTrackingFailure(failure ?? trackingFailure('preview-geometry', error));
        failed = true;
        controls.setBlocking({ enabled: false, generation: currentGeneration, windows: [], target: null, confirming: false });
        capture.current = undefined;
        setView({ key: viewKey, display: undefined, status: 'searching' });
        callbacks.current.onStatus('searching');
        callbacks.current.onError();
      }
    };
    refreshCompletedObservation.current = poll;
    try { configure(); } catch (error) {
      logTrackingFailure(trackingFailure('preview-geometry', error));
      failed = true;
      callbacks.current.onError();
    }
    const timer = setInterval(poll, 100);
    return () => {
      clearInterval(timer);
      if (refreshCompletedObservation.current === poll) refreshCompletedObservation.current = undefined;
      generation.current = currentGeneration + 1;
      controls.setBlocking({ enabled: false, generation: currentGeneration + 1, windows: [], target: null, confirming: false });
      capture.current = undefined;
    };
  }, [active, sessionId, layout.width, layout.height, camera, model, thresholds, channel, controls, viewKey]);

  const output = useFrameOutput({
    targetResolution: TRACKING_RESOLUTION,
    // Keep the YUV buffer in sensor orientation: CameraX 1.7.0-alpha02
    // composes physical rotation with sensorToBuffer in the wrong order.
    // FrameConverter rotates upright from frame.orientation; geometry matches it.
    enablePhysicalBufferRotation: false,
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    onFrame(frame) {
      'worklet';
      // VisionCamera invokes this on its dedicated ImageAnalysis thread with
      // KEEP_ONLY_LATEST. A second runtime hop strips captured helper metadata
      // in installed Worklets 0.10.1, so processing stays on this frame thread.
      let ownsModel = false;
      let ownsFrame = false;
      let admittedSequence = 0;
      let admittedGeneration = 0;
      let admittedAt = 0;
      let source: ReturnType<typeof HybridFrameConverter.convertFrameToImage> | undefined;
      let completed = false;
      let failed = false;
      let stage: TrackingFailureStage = 'admission';
      let transformPoints: readonly SensorPoint[] | undefined;
      try {
        const previous = admission.getBlocking();
        const now = performance.now();
        const received = previous.received + 1;
        let control: Controls;
        busy.lock();
        try {
          control = controls.getBlocking();
          if (!control.enabled || !control.windows.length) { admission.setBlocking({ ...previous, received, disabledDrops: previous.disabledDrops + 1 }); return; }
          if (busy.getBlocking()) { admission.setBlocking({ ...previous, received, busyDrops: previous.busyDrops + 1 }); return; }
          busy.setBlocking(true);
          ownsFrame = true;
        } finally { busy.unlock(); }
        if (now - previous.time < WORD_TRACKING_INTERVAL_MS) { admission.setBlocking({ ...previous, received, throttled: previous.throttled + 1 }); return; }
        admittedSequence = previous.sequence + 1;
        admittedGeneration = control.generation;
        admittedAt = now;
        // Do not hold this mutex during inference or JS lease polling would block.
        modelBusy.lock();
        try {
          if (modelBusy.getBlocking()) { admission.setBlocking({ ...previous, received, busyDrops: previous.busyDrops + 1 }); return; }
          modelBusy.setBlocking(true);
          ownsModel = true;
        } finally { modelBusy.unlock(); }
        const discovery = !control.target || (!control.confirming && previous.tick % 4 === 3);
        const points = discovery ? control.windows[previous.window % control.windows.length]! : control.target!;
        transformPoints = points;
        const sequence = admittedSequence;
        admission.setBlocking({ ...previous, received, time: now, tick: previous.tick + 1, window: previous.window + (discovery ? 1 : 0), sequence });
        if (!controls.getBlocking().enabled || controls.getBlocking().generation !== control.generation) return;
        stage = 'frame-transform';
        const context = mapWordSensorBoxToFrame(points, frame);
        stage = 'frame-conversion';
        source = HybridFrameConverter.convertFrameToImage(frame);
        stage = 'cell-detection';
        const result = detectWordImageCells(source, model, thresholds, context, pixelProcessor);
        if (result.status === 'error') throw new Error('Frame inference failed');
        stage = 'sensor-boxes';
        const groups = result.groups.map((group) => ({
          points: snapshotWordFrameBoxSensorPoints(frame, group.box),
          cells: group.boxes.map((box) => snapshotWordFrameBoxSensorPoints(frame, box)),
          score: group.score, complete: group.complete,
        }));
        stage = 'publish';
        channel.setBlocking({ sequence, generation: control.generation, timestampMs: now,
          durationMs: performance.now() - now, groups, error: false });
        completed = true;
      } catch (error) {
        completed = admittedSequence !== 0;
        failed = true;
        if (admittedSequence) {
          let failure: TrackingFailure = { stage, code: 'unknown' };
          try { failure = trackingFailure(stage, error, frame); } catch {}
          let failureGeometry: TrackingFailureGeometry | undefined;
          if (enableGeometryDiagnostics && stage === 'frame-transform' && transformPoints) {
            try { failureGeometry = snapshotTrackingFailureGeometry(transformPoints, frame); } catch {}
          }
          // Even diagnostic construction/publication must never escape the
          // callback and replace the original native processing failure.
          try {
            channel.setBlocking({ sequence: admittedSequence, generation: admittedGeneration, timestampMs: admittedAt,
              durationMs: performance.now() - admittedAt, groups: [], error: true, failure,
              ...(failureGeometry ? { failureGeometry } : {}) });
          } catch {}
        }
      } finally {
        try {
          if (completed) {
            const previousMetrics = metrics.getBlocking();
            const elapsed = Math.max(0, performance.now() - admittedAt);
            const bucket = elapsed <= 50 ? 0 : elapsed <= 100 ? 1 : elapsed <= 200 ? 2 : elapsed <= 400 ? 3 : elapsed <= 800 ? 4 : elapsed <= 1600 ? 5 : 6;
            const latencyBuckets = [...previousMetrics.latencyBuckets];
            latencyBuckets[bucket] = latencyBuckets[bucket]! + 1;
            metrics.setBlocking({ completed: previousMetrics.completed + 1, errors: previousMetrics.errors + (failed ? 1 : 0), totalMs: previousMetrics.totalMs + elapsed, maxMs: Math.max(previousMetrics.maxMs, elapsed), latencyBuckets });
          }
        } catch { /* Aggregate diagnostics must not prevent resource release. */ }
        try { source?.dispose(); } catch {}
        try { frame.dispose(); } catch {}
        if (ownsModel) {
          try { modelBusy.setBlocking(false); } catch {}
        }
        if (ownsFrame) {
          try { busy.setBlocking(false); } catch {}
        }
      }
    },
  });

  const snapshot = (): TrackedCapture | undefined => {
    if (!active || !mounted.current || stopped.current) return;
    refreshCompletedObservation.current?.();
    const current = capture.current;
    if (!current || current.generation !== generation.current || performance.now() - current.timestampMs > WORD_TRACKING_CAPTURE_AGE_MS) return;
    stopped.current = true;
    controls.setBlocking({ ...controls.getBlocking(), enabled: false });
    return { ...current, points: current.points.map((point) => ({ ...point })), viewportPoints: current.viewportPoints.map((point) => ({ ...point })) };
  };
  const isCurrent = (snapshot: TrackedCapture): boolean => {
    if (!mounted.current || snapshot.geometryGeneration !== geometry.current) return false;
    // Before the still image deactivates preview, also catch delayed native crop changes.
    if (!callbacks.current.active) return true;
    try {
      return sameViewport(snapshot.viewportPoints, snapshotWordGuideSensorPoints(camera.current, TRACKING_VIEWPORT, layout.width, layout.height));
    } catch { return false; }
  };
  return { output, display, status, snapshot, isCurrent };
}
