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
import {
  getWordGuideBox,
  mapImageBoxToPreview,
  recognizeWordImage,
} from '@/ml/word-image-processing';
import { mapSensorGuideToWordPhoto, snapshotWordGuideSensorPoints } from '@/ml/word-camera-geometry';
import { getWordPhotoGeometry } from '../../../modules/braille-word-geometry/src';
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
  readonly guideAspectRatio: number;
  readonly cameraRef: Ref<WordPhotoCameraHandle>;
  readonly onEvent: (event: WordScanEvent) => void;
  readonly onAvailability: (ready: boolean, torchAvailable: boolean) => void;
};
type WordPhotoCameraProps = WordPhotoPreviewProps & {
  readonly model: TensorflowModel;
  readonly thresholds: WordModelThresholds;
  readonly modelVersion: string;
  readonly acquireModel: () => () => void;
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

function WordPhotoCamera({
  active,
  sessionId,
  torchEnabled,
  guideAspectRatio,
  cameraRef,
  onEvent,
  onAvailability,
  model,
  thresholds,
  modelVersion,
  acquireModel,
}: WordPhotoCameraProps) {
  const device = useCameraDevice('back');
  const output = usePhotoOutput({
    targetResolution: TARGET_RESOLUTION,
    qualityPrioritization: 'quality',
    quality: 1,
  });
  const outputs = useMemo(() => [output], [output]);
  const runtime = useMemo(() => getPhotoRuntime(), []);
  const mounted = useRef(true);
  const started = useRef(false);
  const previewStarted = useRef(false);
  const nativeCamera = useRef<CameraRef>(null);
  const activeRef = useRef(active);
  const busy = useRef(false);
  const captured = useRef(false);
  const imageRef = useRef<Image | undefined>(undefined);
  const callbacks = useRef({ onEvent, onAvailability });
  const [image, setImage] = useState<Image>();
  const [result, setResult] = useState<WordScanEvent>();
  const [layout, setLayout] = useState({ width: 0, height: 0 });
  const [cameraError, setCameraError] = useState(false);
  const [captureAspectRatio, setCaptureAspectRatio] = useState<number>();
  const [photoSelection, setPhotoSelection] = useState<NormalizedBox>();

  useEffect(() => {
    callbacks.current = { onEvent, onAvailability };
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
      device?.hasTorch === true,
    );
    if (device || !active) return;
    const timer = setTimeout(() => {
      if (!mounted.current) return;
      setCameraError(true);
      callbacks.current.onEvent({
        schemaVersion: 3,
        sessionId,
        captureId: 0,
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
  ]);

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
          schemaVersion: 3,
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
      captured.current = true;
      setCaptureAspectRatio(guideAspectRatio);
      callbacks.current.onAvailability(false, device?.hasTorch === true);
      const base = {
        schemaVersion: 3 as const,
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
        }
      };
      let photo: Photo | undefined;
      let source: Image | undefined;
      let releaseModel: (() => void) | undefined;
      try {
        const guideAtCapture = getWordGuideBox(
          layout.width,
          layout.height,
          guideAspectRatio,
        );
        const sensorPoints = snapshotWordGuideSensorPoints(nativeCamera.current, guideAtCapture, layout.width, layout.height);
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
        const geometry = getWordPhotoGeometry(photo);
        source = await photo.toImageAsync();
        safeDispose(photo);
        photo = undefined;
        if (!mounted.current || !activeRef.current) return;
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
              preview: { widthDp: layout.width, heightDp: layout.height, resolution: previewResolution ?? null, guide: guideAtCapture, sensorPoints },
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
            return recognizeWordImage(input, model, thresholds, roi);
          },
          source,
        );
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
      guideAspectRatio,
      layout,
      model,
      modelVersion,
      output,
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
  const guide = image && photoSelection && layout.width > 0 && layout.height > 0
    ? mapImageBoxToPreview(photoSelection, layout.width, layout.height, image.width, image.height)
    : layout.width > 0 && layout.height > 0
      ? getWordGuideBox(
          layout.width,
          layout.height,
          captureAspectRatio ?? guideAspectRatio,
        )
      : undefined;

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
          {...(device.hasTorch
            ? { torchMode: active && !image && torchEnabled ? 'on' as const : 'off' as const }
            : {})}
          onStarted={() => {
            started.current = true;
            callbacks.current.onAvailability(
              activeRef.current &&
                previewStarted.current &&
                layout.width > 0 &&
                layout.height > 0 &&
                !captured.current,
              device.hasTorch,
            );
          }}
          onPreviewStarted={() => {
            previewStarted.current = true;
            callbacks.current.onAvailability(activeRef.current && started.current && layout.width > 0 && layout.height > 0 && !captured.current, device.hasTorch);
          }}
          onPreviewStopped={() => {
            previewStarted.current = false;
            callbacks.current.onAvailability(false, device.hasTorch);
          }}
          onStopped={() => {
            started.current = false;
            previewStarted.current = false;
            callbacks.current.onAvailability(false, device.hasTorch);
          }}
          onError={() => {
            if (!mounted.current) return;
            setCameraError(true);
            callbacks.current.onAvailability(false, false);
            callbacks.current.onEvent({
              schemaVersion: 3,
              sessionId,
              captureId: 0,
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
        {guide ? (
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
        {image && result
          ? result.boxes.map((box, index) => (
              <View
                key={index}
                testID={`word-cell-${index}`}
                style={[
                  styles.cell,
                  {
                    left: offsetX + box.x * image.width * scale,
                    top: offsetY + box.y * image.height * scale,
                    width: box.width * image.width * scale,
                    height: box.height * image.height * scale,
                  },
                ]}
              />
            ))
          : null}
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
