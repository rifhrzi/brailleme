import { useEffect, useLayoutEffect, useMemo, useRef, useState, type RefObject } from 'react';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { HybridFrameConverter, useAsyncRunner, useFrameOutput, useOrientation, type CameraRef } from 'react-native-vision-camera';
import { createSynchronizable } from 'react-native-worklets';
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
type Message = {
  sequence: number;
  generation: number;
  timestampMs: number;
  durationMs: number;
  groups: readonly SensorGroup[];
  error: boolean;
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
// A session can remount while its prior task is finishing. The native runner's
// single-task admission guard must be shared by every borrower of this interpreter.
const modelRunners = new WeakMap<TensorflowModel, ReturnType<typeof useAsyncRunner>>();

function sameViewport(a: readonly SensorPoint[], b: readonly SensorPoint[]): boolean {
  const scale = Math.max(1, Math.max(...a.map((point) => point.x)) - Math.min(...a.map((point) => point.x)), Math.max(...a.map((point) => point.y)) - Math.min(...a.map((point) => point.y)));
  return a.length === b.length && a.every((point, index) => Math.abs(point.x - b[index]!.x) <= scale * 0.0001 && Math.abs(point.y - b[index]!.y) <= scale * 0.0001);
}

/** Sends only box coordinates across runtimes. Camera frames and pixels stay transient. */
export function useWordCameraTracking(options: Options) {
  const { active, sessionId, layout, camera, model, thresholds, acquireModel, pixelProcessor } = options;
  const callbacks = useRef(options);
  useLayoutEffect(() => { callbacks.current = options; }, [options]);
  const controls = useMemo(() => createSynchronizable<Controls>({ enabled: false, generation: 0, windows: [], target: null, confirming: false }), []);
  const busy = useMemo(() => createSynchronizable(false), []);
  const admission = useMemo(() => createSynchronizable({ time: -Infinity, tick: 0, window: 0, sequence: 0, received: 0, busyDrops: 0, throttled: 0, disabledDrops: 0, runnerDrops: 0 }), []);
  const metrics = useMemo(() => createSynchronizable({ completed: 0, errors: 0, totalMs: 0, maxMs: 0, latencyBuckets: [0, 0, 0, 0, 0, 0, 0] }), []);
  const channel = useMemo(() => createSynchronizable<Message>({ sequence: 0, generation: 0, timestampMs: 0, durationMs: 0, groups: [], error: false }), []);
  const localRunner = useAsyncRunner();
  const runner = useMemo(() => {
    const existing = modelRunners.get(model);
    if (existing) return existing;
    modelRunners.set(model, localRunner);
    return localRunner;
  }, [model, localRunner]);
  const orientation = useOrientation('interface');
  const generation = useRef(0);
  const state = useRef(INITIAL_WORD_TRACKING_STATE);
  const capture = useRef<TrackedCapture | undefined>(undefined);
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
      controls.setBlocking({ ...controls.getBlocking(), enabled: false });
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
          if (message.error) throw new Error('Frame tracking failed');
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
      } catch {
        failed = true;
        controls.setBlocking({ enabled: false, generation: currentGeneration, windows: [], target: null, confirming: false });
        capture.current = undefined;
        setView({ key: viewKey, display: undefined, status: 'searching' });
        callbacks.current.onStatus('searching');
        callbacks.current.onError();
      }
    };
    try { configure(); } catch {
      failed = true;
      callbacks.current.onError();
    }
    const timer = setInterval(poll, 100);
    return () => {
      clearInterval(timer);
      generation.current = currentGeneration + 1;
      controls.setBlocking({ enabled: false, generation: currentGeneration + 1, windows: [], target: null, confirming: false });
      capture.current = undefined;
    };
  }, [active, sessionId, layout.width, layout.height, camera, model, thresholds, channel, controls, viewKey]);

  const output = useFrameOutput({
    targetResolution: TRACKING_RESOLUTION,
    enablePhysicalBufferRotation: true,
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    onFrame(frame) {
      'worklet';
      let handled = false;
      let admittedSequence = 0;
      let admittedGeneration = 0;
      let admittedAt = 0;
      try {
        const control = controls.getBlocking();
        const previous = admission.getBlocking();
        const now = performance.now();
        const received = previous.received + 1;
        if (!control.enabled || !control.windows.length) { admission.setBlocking({ ...previous, received, disabledDrops: previous.disabledDrops + 1 }); return; }
        if (busy.getBlocking()) { admission.setBlocking({ ...previous, received, busyDrops: previous.busyDrops + 1 }); return; }
        if (now - previous.time < WORD_TRACKING_INTERVAL_MS) { admission.setBlocking({ ...previous, received, throttled: previous.throttled + 1 }); return; }
        const discovery = !control.target || (!control.confirming && previous.tick % 4 === 3);
        const points = discovery ? control.windows[previous.window % control.windows.length]! : control.target!;
        const sequence = previous.sequence + 1;
        admittedSequence = sequence;
        admittedGeneration = control.generation;
        admittedAt = now;
        admission.setBlocking({ ...previous, received, time: now, tick: previous.tick + 1, window: previous.window + (discovery ? 1 : 0), sequence });
        busy.setBlocking(true);
        handled = runner.runAsync(() => {
          'worklet';
          let source: ReturnType<typeof HybridFrameConverter.convertFrameToImage> | undefined;
          let completed = false;
          let failed = false;
          try {
            if (!controls.getBlocking().enabled || controls.getBlocking().generation !== control.generation) return;
            const context = mapWordSensorBoxToFrame(points, frame);
            source = HybridFrameConverter.convertFrameToImage(frame);
            const result = detectWordImageCells(source, model, thresholds, context, pixelProcessor);
            if (result.status === 'error') throw new Error('Frame inference failed');
            const groups = result.groups.map((group) => ({
              points: snapshotWordFrameBoxSensorPoints(frame, group.box),
              cells: group.boxes.map((box) => snapshotWordFrameBoxSensorPoints(frame, box)),
              score: group.score, complete: group.complete,
            }));
            channel.setBlocking({ sequence, generation: control.generation, timestampMs: now,
              durationMs: performance.now() - now, groups, error: false });
            completed = true;
          } catch {
            completed = true;
            failed = true;
            channel.setBlocking({ sequence, generation: control.generation, timestampMs: now,
              durationMs: performance.now() - now, groups: [], error: true });
          } finally {
            if (completed) {
              const previousMetrics = metrics.getBlocking();
              const elapsed = Math.max(0, performance.now() - now);
              const bucket = elapsed <= 50 ? 0 : elapsed <= 100 ? 1 : elapsed <= 200 ? 2 : elapsed <= 400 ? 3 : elapsed <= 800 ? 4 : elapsed <= 1600 ? 5 : 6;
              const latencyBuckets = [...previousMetrics.latencyBuckets];
              latencyBuckets[bucket] = latencyBuckets[bucket]! + 1;
              metrics.setBlocking({ completed: previousMetrics.completed + 1, errors: previousMetrics.errors + (failed ? 1 : 0), totalMs: previousMetrics.totalMs + elapsed, maxMs: Math.max(previousMetrics.maxMs, elapsed), latencyBuckets });
            }
            try { source?.dispose(); } catch {}
            try { frame.dispose(); } catch {}
            busy.setBlocking(false);
          }
        });
        if (!handled) {
          busy.setBlocking(false);
          const dropped = admission.getBlocking();
          admission.setBlocking({ ...dropped, runnerDrops: dropped.runnerDrops + 1 });
        }
      } catch {
        if (!handled) busy.setBlocking(false);
        if (admittedSequence) channel.setBlocking({ sequence: admittedSequence, generation: admittedGeneration, timestampMs: admittedAt, durationMs: 0, groups: [], error: true });
      } finally {
        if (!handled) { try { frame.dispose(); } catch {} }
      }
    },
  });

  const snapshot = (): TrackedCapture | undefined => {
    const current = capture.current;
    if (!active || stopped.current || !current || current.generation !== generation.current || performance.now() - current.timestampMs > WORD_TRACKING_CAPTURE_AGE_MS) return;
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
