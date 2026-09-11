import {
  memo,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type Ref,
} from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { NitroImage, type Image } from 'react-native-nitro-image';
import {
  Camera,
  useCameraDevice,
  usePhotoOutput,
  type CameraRef,
  type Constraint,
  type Photo,
} from 'react-native-vision-camera';
import { createWorkletRuntime, runOnRuntimeAsync } from 'react-native-worklets';

import { APP_ENVIRONMENT } from '@/config/app-environment';
import { useWordCameraTracking, type TrackingStatus } from '@/features/live/use-word-camera-tracking';
import {
  mapImageBoxToPreview,
  recognizeTrackedWordImage,
} from '@/ml/word-image-processing';
import { mapSensorGuideToWordPhoto } from '@/ml/word-camera-geometry';
import { createWordPixelProcessor, getWordPhotoGeometry } from '../../../modules/braille-word-geometry/src';
import type { NormalizedBox } from '@/types/live-scan';
import {
  WORD_DECODER_VERSION,
  type WordModelThresholds,
} from '@/ml/word-model-contract';
import { colors, radii, typography } from '@/theme';
import type { WordScanEvent } from '@/types/word-scan';

export type WordPhotoCameraHandle = { capture(captureId: number): void };
export type WordPhotoPreviewProps = {
  readonly active: boolean;
  readonly sessionId: string;
  readonly torchEnabled: boolean;
  readonly cameraRef: Ref<WordPhotoCameraHandle>;
  readonly onEvent: (event: WordScanEvent) => void;
  readonly onAvailability: (ready: boolean, torchAvailable: boolean) => void;
  readonly onTracking: (status: TrackingStatus) => void;
};
type WordPhotoCameraProps = WordPhotoPreviewProps & {
  readonly model: TensorflowModel;
  readonly trackingModel: TensorflowModel;
  readonly thresholds: WordModelThresholds;
  readonly modelVersion: string;
  readonly acquireModel: () => () => void;
  readonly acquireTrackingModel: () => () => void;
};

const TARGET_RESOLUTION = { width: 1920, height: 1440 };
const PHOTO_CONSTRAINTS: Constraint[] = [{ previewStabilizationMode: 'off' }];
let photoRuntime: ReturnType<typeof createWorkletRuntime> | undefined;

function getPhotoRuntime() {
  // A previous session may still be processing when the app resumes. Sharing
  // this worker serializes access to the native TFLite interpreter across mounts.
  photoRuntime ??= createWorkletRuntime({ name: 'Braille word photo' });
  return photoRuntime;
}

function safeDispose(value: { dispose(): void } | undefined) {
  try {
    value?.dispose();
  } catch {
    /* Cleanup must also release other resources. */
  }
}

function logCameraError(stage: string, error: unknown) {
  try {
    // Camera-control errors only. Never pass Photo/Image objects or scan results.
    console.warn('BRAILLE_WORD_CAMERA_ERROR', JSON.stringify({
      stage,
      message: error instanceof Error ? error.message : 'Unknown camera error',
    }));
  } catch {
    /* Diagnostics must not interrupt camera recovery. */
  }
}

function WordPhotoCamera({
  active,
  sessionId,
  torchEnabled,
  cameraRef,
  onEvent,
  onAvailability,
  onTracking,
  model,
  trackingModel,
  thresholds,
  modelVersion,
  acquireModel,
  acquireTrackingModel,
}: WordPhotoCameraProps) {
  const device = useCameraDevice('back');
  const output = usePhotoOutput({
    targetResolution: TARGET_RESOLUTION,
    qualityPrioritization: 'quality',
    quality: 1,
  });
  const runtime = useMemo(() => getPhotoRuntime(), []);
  const pixelProcessor = useMemo(() => createWordPixelProcessor(), []);
  const mounted = useRef(true);
  const started = useRef(false);
  const previewStarted = useRef(false);
  const nativeCamera = useRef<CameraRef>(null);
  const activeRef = useRef(active);
  const busy = useRef(false);
  const captured = useRef(false);
  const currentCaptureId = useRef(0);
  const imageRef = useRef<Image | undefined>(undefined);
  const trackingReady = useRef(false);
  const callbacks = useRef({ onEvent, onAvailability: (ready: boolean, torch: boolean) => onAvailability(ready && trackingReady.current, torch) });
  const [image, setImage] = useState<Image>();
  const [result, setResult] = useState<WordScanEvent>();
  const [layout, setLayout] = useState({ width: 0, height: 0 });
  const [cameraError, setCameraError] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [torchFailed, setTorchFailed] = useState(false);
  const appliedTorch = useRef(false);
  const [photoSelection, setPhotoSelection] = useState<NormalizedBox>();
  const [trackingFailed, setTrackingFailed] = useState(false);
  const tracking = useWordCameraTracking({
    active: active && streaming && !image && !cameraError && !trackingFailed,
    sessionId, layout, camera: nativeCamera, model: trackingModel, thresholds,
    acquireModel: acquireTrackingModel, pixelProcessor,
    onStatus: (status) => {
      trackingReady.current = status === 'ready';
      if (!mounted.current || captured.current) return;
      onTracking(status);
      callbacks.current.onAvailability(activeRef.current && started.current && previewStarted.current && !cameraError && !trackingFailed,
        device?.hasTorch === true && !torchFailed && !cameraError);
    },
    onError: () => {
      if (!mounted.current || !activeRef.current || captured.current) return;
      trackingReady.current = false;
      setTrackingFailed(true);
      callbacks.current.onAvailability(false, device?.hasTorch === true && !torchFailed);
      callbacks.current.onEvent({ schemaVersion: 4, sessionId, captureId: 0,
        timestampMs: performance.now(), status: 'error', reason: 'tracking_error',
        cells: [], boxes: [], decoderVersion: WORD_DECODER_VERSION, modelVersion });
    },
  });
  const outputs = useMemo(() => [output, tracking.output], [output, tracking.output]);

  useEffect(() => {
    callbacks.current = { onEvent, onAvailability: (ready: boolean, torch: boolean) => onAvailability(ready && trackingReady.current, torch) };
  }, [onEvent, onAvailability]);
  useEffect(() => {
    activeRef.current = active;
  }, [active]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      activeRef.current = false;
      callbacks.current.onAvailability(false, false);
      // runSync must finish before its source image can be disposed.
      if (!busy.current) {
        safeDispose(imageRef.current);
        imageRef.current = undefined;
      }
    };
  }, []);

  useEffect(() => {
    callbacks.current.onAvailability(
      active &&
        started.current &&
        previewStarted.current &&
        layout.width > 0 &&
        layout.height > 0 &&
        !captured.current &&
        !cameraError,
      device?.hasTorch === true && !torchFailed && !cameraError,
    );
    if (!active || image || cameraError || streaming) return;
    const timer = setTimeout(() => {
      if (!mounted.current || !activeRef.current || (started.current && previewStarted.current)) return;
      logCameraError('startup', new Error(device ? 'Camera preview did not start within 15 seconds' : 'Back camera unavailable'));
      setCameraError(true);
      callbacks.current.onAvailability(false, false);
      callbacks.current.onEvent({
        schemaVersion: 4,
        sessionId,
        captureId: currentCaptureId.current,
        timestampMs: performance.now(),
        status: 'error',
        reason: 'capture_error',
        cells: [],
        boxes: [],
        decoderVersion: WORD_DECODER_VERSION,
        modelVersion,
      });
    }, 15_000);
    return () => clearTimeout(timer);
  }, [
    active,
    cameraError,
    device,
    layout.height,
    layout.width,
    modelVersion,
    sessionId,
    image,
    streaming,
    torchFailed,
  ]);

  useEffect(() => {
    // CameraX can reject controls before it is active. Do not send even "off"
    // at mount; a newly opened camera already has its torch off.
    const controller = nativeCamera.current?.controller;
    if (!active || !streaming || image || cameraError || !device?.hasTorch || torchFailed || !controller) return;
    if (appliedTorch.current === torchEnabled) return;
    let cancelled = false;
    appliedTorch.current = torchEnabled;
    void (async () => {
      try {
        await controller.setTorchMode(torchEnabled ? 'on' : 'off');
      } catch (error) {
        if (cancelled || !mounted.current || !activeRef.current) return;
        logCameraError('torch', error);
        setTorchFailed(true);
        // A failed optional control must not unmount an otherwise working camera.
        callbacks.current.onAvailability(started.current && previewStarted.current && !captured.current, false);
      }
    })();
    return () => { cancelled = true; };
  }, [active, cameraError, device?.hasTorch, image, streaming, torchEnabled, torchFailed]);

  const capture = useCallback(
    async (captureId: number) => {
      if (
        !mounted.current ||
        !activeRef.current ||
        busy.current ||
        captured.current
      )
        return;
      if (!started.current || !previewStarted.current || !layout.width || !layout.height) {
        callbacks.current.onAvailability(false, device?.hasTorch === true);
        callbacks.current.onEvent({
          schemaVersion: 4,
          sessionId,
          captureId,
          timestampMs: performance.now(),
          status: 'error',
          reason: 'capture_error',
          cells: [],
          boxes: [],
          decoderVersion: WORD_DECODER_VERSION,
          modelVersion,
        });
        return;
      }
      busy.current = true;
      const captureStartedAt = performance.now();
      captured.current = true;
      currentCaptureId.current = captureId;
      callbacks.current.onAvailability(false, device?.hasTorch === true);
      const base = {
        schemaVersion: 4 as const,
        sessionId,
        captureId,
        decoderVersion: WORD_DECODER_VERSION,
        modelVersion,
        cells: [],
        boxes: [],
      };
      const publish = (event: WordScanEvent) => {
        if (mounted.current && activeRef.current) {
          setResult(event);
          callbacks.current.onEvent(event);
          if (APP_ENVIRONMENT === 'preview' && ['success', 'rejected', 'error'].includes(event.status)) {
            try {
              console.info('BRAILLE_WORD_PHOTO_METRICS', JSON.stringify({ schemaVersion: 1,
                captureId, elapsedMs: Math.round(performance.now() - captureStartedAt), status: event.status }));
            } catch { /* Metrics do not affect the result. */ }
          }
        }
      };
      let photo: Photo | undefined;
      let source: Image | undefined;
      let releaseModel: (() => void) | undefined;
      try {
        const trackedCapture = tracking.snapshot();
        if (!trackedCapture) {
          publish({ ...base, timestampMs: performance.now(), status: 'error', reason: 'tracking_error' });
          return;
        }
        const sensorPoints = trackedCapture.points;
        let previewResolution: { width: number; height: number } | undefined;
        if (APP_ENVIRONMENT === 'preview') {
          try {
            const resolution = nativeCamera.current?.preview?.previewOutput?.currentResolution;
            if (resolution) previewResolution = { width: resolution.width, height: resolution.height };
          } catch {
            /* Optional diagnostic metadata must not prevent a capture. */
          }
        }
        publish({
          ...base,
          timestampMs: performance.now(),
          status: 'capturing',
        });
        photo = await output.capturePhoto(
          {
            flashMode: 'off',
            enableDistortionCorrection: false,
            enableVirtualDeviceFusion: false,
          },
          {},
        );
        if (!mounted.current || !activeRef.current) return;
        if (!tracking.isCurrent(trackedCapture)) throw new Error('Camera layout changed during shutter');
        const geometry = getWordPhotoGeometry(photo);
        source = await photo.toImageAsync();
        safeDispose(photo);
        photo = undefined;
        if (!mounted.current || !activeRef.current) return;
        if (!tracking.isCurrent(trackedCapture)) throw new Error('Camera layout changed during conversion');
        imageRef.current = source;
        setImage(source);
        publish({
          ...base,
          timestampMs: performance.now(),
          status: 'processing',
        });
        // Match actual preview and still sensor crops, including HAL rotation.
        const roi = mapSensorGuideToWordPhoto(sensorPoints, geometry, source.width, source.height);
        setPhotoSelection(roi);
        if (APP_ENVIRONMENT === 'preview') {
          try {
            // One geometry-only Logcat record per capture; never log pixels, text, Photo/Image objects or URIs.
            console.info('BRAILLE_WORD_CAPTURE_GEOMETRY', JSON.stringify({
              schemaVersion: 1,
              captureId,
              preview: { widthDp: layout.width, heightDp: layout.height, resolution: previewResolution ?? null, sensorPoints },
              photo: { width: geometry.width, height: geometry.height, sensorToBuffer: geometry.sensorToBuffer, cropRect: geometry.cropRect, rotationDegrees: geometry.rotationDegrees, isMirrored: geometry.isMirrored },
              upright: { width: source.width, height: source.height },
              roi,
            }));
          } catch {
            /* Diagnostic logging must not affect image processing. */
          }
        }
        releaseModel = acquireModel();
        const outcome = await runOnRuntimeAsync(
          runtime,
          (input: Image) => {
            'worklet';
            return recognizeTrackedWordImage(input, model, thresholds, roi, pixelProcessor);
          },
          source,
        );
        if (!tracking.isCurrent(trackedCapture)) throw new Error('Camera layout changed during inference');
        if (mounted.current && activeRef.current && outcome.boxes.length) {
          const left = Math.min(...outcome.boxes.map((box) => box.x));
          const top = Math.min(...outcome.boxes.map((box) => box.y));
          setPhotoSelection({ x: left, y: top,
            width: Math.max(...outcome.boxes.map((box) => box.x + box.width)) - left,
            height: Math.max(...outcome.boxes.map((box) => box.y + box.height)) - top });
        }
        publish({ ...base, ...outcome, timestampMs: performance.now() });
      } catch {
        publish({
          ...base,
          timestampMs: performance.now(),
          status: 'error',
          reason: source ? 'processing_error' : 'capture_error',
        });
      } finally {
        safeDispose(photo);
        try {
          releaseModel?.();
        } catch {
          /* Still release the image below. */
        }
        busy.current = false;
        if (
          !mounted.current ||
          !activeRef.current ||
          source !== imageRef.current
        ) {
          safeDispose(source);
          if (source === imageRef.current) imageRef.current = undefined;
        }
      }
    },
    [
      acquireModel,
      device?.hasTorch,
      tracking,
      layout,
      model,
      modelVersion,
      output,
      pixelProcessor,
      runtime,
      sessionId,
      thresholds,
    ],
  );

  useImperativeHandle(
    cameraRef,
    () => ({
      capture: (captureId) => {
        void capture(captureId);
      },
    }),
    [capture],
  );

  const scale =
    image && layout.width > 0
      ? Math.max(layout.width / image.width, layout.height / image.height)
      : 1;
  const offsetX = image ? (layout.width - image.width * scale) / 2 : 0;
  const offsetY = image ? (layout.height - image.height * scale) / 2 : 0;
  const guide = image
    ? result?.boxes.length && photoSelection && layout.width > 0 && layout.height > 0
      ? mapImageBoxToPreview(photoSelection, layout.width, layout.height, image.width, image.height)
      : undefined
    : tracking.display?.box;

  return (
    <View
      testID="word-photo-preview"
      style={styles.preview}
      onLayout={({ nativeEvent }) =>
        setLayout({
          width: nativeEvent.layout.width,
          height: nativeEvent.layout.height,
        })
      }
    >
      {device && !cameraError ? (
        <Camera
          ref={nativeCamera}
          device={device}
          isActive={active && !image}
          outputs={outputs}
          constraints={PHOTO_CONSTRAINTS}
          mirrorMode="off"
          orientationSource="interface"
          implementationMode="compatible"
          resizeMode="cover"
          enableNativeTapToFocusGesture
          onStarted={() => {
            if (!mounted.current || !activeRef.current) return;
            started.current = true;
            setStreaming(previewStarted.current);
            callbacks.current.onAvailability(
              activeRef.current &&
                previewStarted.current &&
                layout.width > 0 &&
                layout.height > 0 &&
                !captured.current,
              device.hasTorch && !torchFailed,
            );
          }}
          onPreviewStarted={() => {
            if (!mounted.current || !activeRef.current) return;
            previewStarted.current = true;
            setStreaming(started.current);
            callbacks.current.onAvailability(activeRef.current && started.current && layout.width > 0 && layout.height > 0 && !captured.current, device.hasTorch && !torchFailed);
          }}
          onPreviewStopped={() => {
            previewStarted.current = false;
            if (!mounted.current) return;
            setStreaming(false);
            callbacks.current.onAvailability(false, device.hasTorch && !torchFailed);
          }}
          onStopped={() => {
            started.current = false;
            previewStarted.current = false;
            appliedTorch.current = false;
            if (!mounted.current) return;
            setStreaming(false);
            callbacks.current.onAvailability(false, device.hasTorch && !torchFailed);
          }}
          onError={(error) => {
            if (!mounted.current || !activeRef.current || imageRef.current) return;
            logCameraError('session', error);
            started.current = false;
            previewStarted.current = false;
            setCameraError(true);
            callbacks.current.onAvailability(false, false);
            callbacks.current.onEvent({
              schemaVersion: 4,
              sessionId,
              captureId: currentCaptureId.current,
              timestampMs: performance.now(),
              status: 'error',
              reason: 'capture_error',
              cells: [],
              boxes: [],
              decoderVersion: WORD_DECODER_VERSION,
              modelVersion,
            });
          }}
          style={StyleSheet.absoluteFill}
        />
      ) : (
        <Text selectable style={styles.notice}>
          {cameraError ? 'Kamera belum dapat digunakan.' : 'Menyiapkan kamera.'}
        </Text>
      )}
      {image ? (
        <NitroImage
          image={image}
          resizeMode="cover"
          style={StyleSheet.absoluteFill}
        />
      ) : null}
      <View
        pointerEvents="none"
        accessible={false}
        importantForAccessibility="no-hide-descendants"
        style={StyleSheet.absoluteFill}
      >
        {guide && !cameraError && !trackingFailed ? (
          <View
            testID="word-guide"
            style={[
              styles.guide,
              {
                left: guide.x * layout.width,
                top: guide.y * layout.height,
                width: guide.width * layout.width,
                height: guide.height * layout.height,
              },
            ]}
          />
        ) : null}
        {(image && result ? result.boxes : tracking.display?.boxes ?? []).map((box, index) => (
              <View
                key={index}
                testID={`word-cell-${index}`}
                style={[
                  styles.cell,
                  {
                    left: image ? offsetX + box.x * image.width * scale : box.x * layout.width,
                    top: image ? offsetY + box.y * image.height * scale : box.y * layout.height,
                    width: box.width * (image ? image.width * scale : layout.width),
                    height: box.height * (image ? image.height * scale : layout.height),
                  },
                ]}
              />
            ))}
      </View>
    </View>
  );
}

export default memo(WordPhotoCamera);

const styles = StyleSheet.create({
  preview: {
    width: '100%',
    aspectRatio: 1,
    overflow: 'hidden',
    backgroundColor: colors.inverseSurface,
    borderRadius: radii.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  guide: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: colors.inverseOnSurface,
    borderRadius: radii.md,
  },
  cell: { position: 'absolute', borderWidth: 2, borderColor: colors.success },
  notice: {
    ...typography.body,
    color: colors.inverseOnSurface,
    textAlign: 'center',
  },
});
