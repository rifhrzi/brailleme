import { useFocusEffect } from 'expo-router';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AppState,
  StyleSheet,
  Text,
  View,
  type AppStateStatus,
} from 'react-native';
import type { TensorflowModel } from 'react-native-fast-tflite';
import type { Image } from 'react-native-nitro-image';
import {
  Camera,
  HybridFrameConverter,
  useCameraDevice,
  useFrameOutput,
} from 'react-native-vision-camera';
import { createSynchronizable } from 'react-native-worklets';

import {
  BRAILLE_DECODER_VERSION,
  BRAILLE_LABELS,
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_SCALE,
  BRAILLE_MODEL_INPUT_WIDTH,
  BRAILLE_MODEL_VERSION,
  INITIAL_PREDICTION_STATE,
  advancePredictionStabilizer,
  assertBrailleModelInputLength,
  convertRawPixelsToRgbTensor,
  decodeProbabilities,
  getFrameQualityIssue,
  type ModelPrediction,
  type PredictionStabilizerState,
} from '@/ml';
import type { SingleCellScannerPort } from '@/services';
import { colors, radii, spacing, typography } from '@/theme';
import type { QualityReason } from '@/types';

type WorkletMessage = {
  readonly sequence: number;
  readonly generation: number;
  readonly sessionId: string;
  readonly kind: 'idle' | 'quality' | 'inference' | 'error';
  readonly index: number;
  readonly confidence: number;
  readonly runnerUpConfidence: number;
  readonly meanLuminance: number;
  readonly contrast: number;
  readonly timestampMs: number;
  readonly message: string;
  readonly reason: QualityReason | '';
};

type LiveSingleCellCameraProps = {
  readonly active: boolean;
  readonly model: TensorflowModel;
  readonly resetToken?: number;
  readonly scanner: SingleCellScannerPort;
  readonly sessionId: string;
  readonly torchEnabled: boolean;
};

const TARGET_RESOLUTION = { width: 640, height: 480 };
const CAMERA_CONSTRAINTS = [{ fps: 30 }];
const INFERENCE_INTERVAL_MS = 200;
const RESULT_POLL_INTERVAL_MS = INFERENCE_INTERVAL_MS;
const CENTER_CROP_FRACTION = 0.55;
const DEVICE_SEARCH_NOTICE_DELAY_MS = 10_000;
// Pipeline ini memotong area target, bukan mendeteksi bounding box Braille.
// Jangan mengirim kotak semu yang tidak sesuai dengan koordinat frame aktual.
const NO_DETECTED_BOXES = Object.freeze([]);
const INITIAL_MESSAGE: WorkletMessage = Object.freeze({
  sequence: 0,
  generation: 0,
  sessionId: '',
  kind: 'idle',
  index: -1,
  confidence: 0,
  runnerUpConfidence: 0,
  meanLuminance: 0,
  contrast: 0,
  timestampMs: 0,
  message: '',
  reason: '',
});

function LiveSingleCellCamera({
  active,
  model,
  resetToken = 0,
  scanner,
  sessionId,
  torchEnabled,
}: LiveSingleCellCameraProps) {
  const device = useCameraDevice('back');
  const resultChannel = useMemo(
    () => createSynchronizable<WorkletMessage>(INITIAL_MESSAGE),
    [],
  );
  const workletActive = useMemo(() => createSynchronizable(false), []);
  const workletFailed = useMemo(() => createSynchronizable(false), []);
  const lastProcessedAt = useMemo(() => createSynchronizable(0), []);
  const lastMessageSequenceRef = useRef(0);
  const stabilizerRef = useRef<PredictionStabilizerState>(
    INITIAL_PREDICTION_STATE,
  );
  const cameraActiveRef = useRef(false);
  const fatalErrorRef = useRef(false);
  const mountedRef = useRef(true);
  const [isFocused, setIsFocused] = useState(false);
  const [appState, setAppState] = useState<AppStateStatus>(AppState.currentState);
  const [cameraStarted, setCameraStarted] = useState(false);
  const [processingFailed, setProcessingFailed] = useState(false);
  const [deviceSearchTimedOut, setDeviceSearchTimedOut] = useState(false);

  useEffect(() => {
    scanner.setTorchAvailable(device?.hasTorch === true);
    return () => scanner.setTorchAvailable(false);
  }, [device?.hasTorch, device?.id, scanner]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cameraActiveRef.current = false;
    };
  }, []);

  useEffect(() => {
    fatalErrorRef.current = false;
    workletFailed.setBlocking(false);
    lastMessageSequenceRef.current = 0;
    stabilizerRef.current = INITIAL_PREDICTION_STATE;
  }, [sessionId, workletFailed]);

  useEffect(() => {
    stabilizerRef.current = INITIAL_PREDICTION_STATE;
    lastMessageSequenceRef.current = 0;
    lastProcessedAt.setBlocking(0);
    if (!workletFailed.getBlocking()) {
      resultChannel.setBlocking({
        ...INITIAL_MESSAGE,
        generation: resetToken,
        sessionId,
      });
    }
    if (resetToken > 0 && cameraActiveRef.current) {
      scanner.publish(sessionId, {
        timestampMs: performance.now(),
        status: 'searching',
        boxes: NO_DETECTED_BOXES,
        decoderVersion: BRAILLE_DECODER_VERSION,
        modelVersion: BRAILLE_MODEL_VERSION,
      });
    }
  }, [
    lastProcessedAt,
    resetToken,
    resultChannel,
    scanner,
    sessionId,
    workletFailed,
  ]);

  useFocusEffect(
    useCallback(() => {
      setIsFocused(true);
      return () => {
        cameraActiveRef.current = false;
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
        setIsFocused(false);
      };
    }, []),
  );

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextState) => {
      setAppState(nextState);
      if (nextState !== 'active') {
        cameraActiveRef.current = false;
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
      }
    });
    return () => subscription.remove();
  }, []);

  const cameraActive =
    active && isFocused && appState === 'active' && !processingFailed;

  useEffect(() => {
    cameraActiveRef.current = cameraActive;
    workletActive.setBlocking(cameraActive);
    stabilizerRef.current = INITIAL_PREDICTION_STATE;
    if (cameraActive) {
      scanner.publish(sessionId, {
        timestampMs: performance.now(),
        status: 'searching',
        boxes: NO_DETECTED_BOXES,
        decoderVersion: BRAILLE_DECODER_VERSION,
        modelVersion: BRAILLE_MODEL_VERSION,
      });
    }
    return () => workletActive.setBlocking(false);
  }, [cameraActive, scanner, sessionId, workletActive]);

  const publishQualityIssue = useCallback(
    (
      originatingSessionId: string,
      reason: QualityReason,
      timestampMs: number,
      confidence?: number,
    ) => {
      stabilizerRef.current = INITIAL_PREDICTION_STATE;
      scanner.publish(originatingSessionId, {
        timestampMs,
        status: 'quality_issue',
        reason,
        ...(confidence === undefined ? {} : { confidence }),
        boxes: NO_DETECTED_BOXES,
        decoderVersion: BRAILLE_DECODER_VERSION,
        modelVersion: BRAILLE_MODEL_VERSION,
      });
    },
    [scanner],
  );

  const onProcessingError = useCallback(
    (originatingSessionId: string, _message: string) => {
      if (
        originatingSessionId !== sessionId ||
        !cameraActiveRef.current ||
        fatalErrorRef.current
      ) {
        return;
      }
      fatalErrorRef.current = true;
      if (__DEV__) {
        console.warn(`[BrailleMe scanner] ${_message}`);
      }
      cameraActiveRef.current = false;
      stabilizerRef.current = INITIAL_PREDICTION_STATE;
      scanner.publish(originatingSessionId, {
        timestampMs: performance.now(),
        status: 'error',
        boxes: NO_DETECTED_BOXES,
        decoderVersion: BRAILLE_DECODER_VERSION,
        modelVersion: BRAILLE_MODEL_VERSION,
      });
      if (mountedRef.current) {
        setProcessingFailed(true);
      }
    },
    [scanner, sessionId],
  );

  const onInference = useCallback(
    (
      originatingSessionId: string,
      index: number,
      confidence: number,
      runnerUpConfidence: number,
      meanLuminance: number,
      contrast: number,
      timestampMs: number,
    ) => {
      if (
        originatingSessionId !== sessionId ||
        !cameraActiveRef.current ||
        fatalErrorRef.current
      ) {
        return;
      }

      const label = BRAILLE_LABELS[index];
      const margin = confidence - runnerUpConfidence;
      if (!label || !Number.isFinite(confidence) || !Number.isFinite(margin)) {
        onProcessingError(originatingSessionId, 'Output model tidak valid.');
        return;
      }

      const qualityIssue = getFrameQualityIssue(meanLuminance, contrast);
      if (qualityIssue) {
        publishQualityIssue(
          originatingSessionId,
          qualityIssue,
          timestampMs,
          confidence,
        );
        return;
      }

      const prediction: ModelPrediction = {
        index,
        confidence,
        runnerUpConfidence,
        margin,
      };
      const outcome = advancePredictionStabilizer(
        stabilizerRef.current,
        prediction,
        timestampMs,
      );
      stabilizerRef.current = outcome.state;

      if (outcome.status === 'low_confidence') {
        publishQualityIssue(
          originatingSessionId,
          'low_confidence',
          timestampMs,
          confidence,
        );
        return;
      }

      const text = label.toUpperCase();
      scanner.publish(originatingSessionId, {
        timestampMs,
        status: outcome.status,
        text,
        confidence,
        boxes: NO_DETECTED_BOXES,
        decoderVersion: BRAILLE_DECODER_VERSION,
        modelVersion: BRAILLE_MODEL_VERSION,
      });
    },
    [onProcessingError, publishQualityIssue, scanner, sessionId],
  );

  useEffect(() => {
    if (!cameraActive) {
      lastMessageSequenceRef.current = resultChannel.getBlocking().sequence;
      return;
    }

    const interval = setInterval(() => {
      const message = resultChannel.getBlocking();
      if (!cameraActiveRef.current) {
        lastMessageSequenceRef.current = message.sequence;
        return;
      }
      if (message.sequence === lastMessageSequenceRef.current) {
        return;
      }

      lastMessageSequenceRef.current = message.sequence;
      if (message.sessionId !== sessionId) {
        return;
      }
      if (message.kind === 'error') {
        onProcessingError(message.sessionId, message.message);
      } else if (message.generation !== resetToken) {
        return;
      } else if (message.kind === 'quality' && message.reason) {
        publishQualityIssue(
          message.sessionId,
          message.reason,
          message.timestampMs,
        );
      } else if (message.kind === 'inference') {
        onInference(
          message.sessionId,
          message.index,
          message.confidence,
          message.runnerUpConfidence,
          message.meanLuminance,
          message.contrast,
          message.timestampMs,
        );
      }
    }, RESULT_POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [
    cameraActive,
    onInference,
    onProcessingError,
    publishQualityIssue,
    resetToken,
    resultChannel,
    sessionId,
  ]);

  const frameOutput = useFrameOutput({
    dropFramesWhileBusy: true,
    // Model menerima piksel upright. Biarkan pipeline kamera merotasi buffer
    // secara fisik agar orientasi portrait tidak masuk sebagai 90/270 derajat.
    enablePhysicalBufferRotation: true,
    pixelFormat: 'yuv',
    targetResolution: TARGET_RESOLUTION,
    onFrame(frame) {
      'worklet';

      let sourceImage: Image | undefined;
      let croppedImage: Image | undefined;
      let resizedImage: Image | undefined;
      let frameNeedsDispose = true;
      let stage = 'membatasi laju inferensi';
      const timestampMs = performance.now();

      try {
        if (!workletActive.getBlocking()) {
          return;
        }
        if (workletFailed.getBlocking()) {
          return;
        }
        const previousTimestampMs = lastProcessedAt.getBlocking();
        if (timestampMs - previousTimestampMs < INFERENCE_INTERVAL_MS) {
          return;
        }
        lastProcessedAt.setBlocking(timestampMs);

        stage = 'mengonversi frame kamera';
        sourceImage = HybridFrameConverter.convertFrameToImage(frame);
        stage = 'melepas frame kamera';
        frame.dispose();
        frameNeedsDispose = false;

        stage = 'menghitung crop tengah';
        const cropSize = Math.max(
          1,
          Math.floor(
            Math.min(sourceImage.width, sourceImage.height) *
              CENTER_CROP_FRACTION,
          ),
        );
        const startX = Math.floor((sourceImage.width - cropSize) / 2);
        const startY = Math.floor((sourceImage.height - cropSize) / 2);

        stage = 'memotong gambar';
        croppedImage = sourceImage.crop(
          startX,
          startY,
          startX + cropSize,
          startY + cropSize,
        );
        stage = 'mengubah ukuran gambar';
        resizedImage = croppedImage.resize(
          BRAILLE_MODEL_INPUT_WIDTH,
          BRAILLE_MODEL_INPUT_HEIGHT,
        );

        stage = 'membaca piksel gambar';
        const rawPixels = resizedImage.toRawPixelData(false);
        stage = 'melakukan preprocessing';
        const preprocessed = convertRawPixelsToRgbTensor(
          rawPixels,
          BRAILLE_MODEL_INPUT_SCALE,
        );
        stage = 'memeriksa kualitas frame';
        const qualityIssue = getFrameQualityIssue(
          preprocessed.meanLuminance,
          preprocessed.contrast,
        );
        if (qualityIssue) {
          resultChannel.setBlocking({
            sequence: timestampMs,
            generation: resetToken,
            sessionId,
            kind: 'quality',
            index: -1,
            confidence: 0,
            runnerUpConfidence: 0,
            meanLuminance: preprocessed.meanLuminance,
            contrast: preprocessed.contrast,
            timestampMs,
            message: '',
            reason: qualityIssue,
          });
          return;
        }
        stage = 'menjalankan model';
        assertBrailleModelInputLength(preprocessed.tensor);
        const [outputBuffer] = model.runSync([preprocessed.tensor.buffer]);

        stage = 'memeriksa output model';
        if (!outputBuffer) {
          throw new Error('Model tidak menghasilkan output.');
        }
        const prediction = decodeProbabilities(new Float32Array(outputBuffer));
        if (!prediction) {
          throw new Error('Model menghasilkan probabilitas tidak valid.');
        }

        resultChannel.setBlocking({
          sequence: timestampMs,
          generation: resetToken,
          sessionId,
          kind: 'inference',
          index: prediction.index,
          confidence: prediction.confidence,
          runnerUpConfidence: prediction.runnerUpConfidence,
          meanLuminance: preprocessed.meanLuminance,
          contrast: preprocessed.contrast,
          timestampMs,
          message: '',
          reason: '',
        });
      } catch (error) {
        workletFailed.setBlocking(true);
        resultChannel.setBlocking({
          sequence: timestampMs,
          generation: resetToken,
          sessionId,
          kind: 'error',
          index: -1,
          confidence: 0,
          runnerUpConfidence: 0,
          meanLuminance: 0,
          contrast: 0,
          timestampMs,
          message: `Gagal saat ${stage}: ${String(error)}`,
          reason: '',
        });
      } finally {
        // Setiap resource native dibersihkan secara independen. Jika satu
        // dispose gagal, resource berikutnya tetap harus dilepas.
        try {
          resizedImage?.dispose();
        } catch {}
        try {
          croppedImage?.dispose();
        } catch {}
        try {
          sourceImage?.dispose();
        } catch {}
        if (frameNeedsDispose) {
          try {
            frame.dispose();
            frameNeedsDispose = false;
          } catch {}
        }
      }
    },
  });
  const cameraOutputs = useMemo(() => [frameOutput], [frameOutput]);

  useEffect(() => {
    if (device || !cameraActive) {
      return;
    }
    const resetTimeout = setTimeout(() => {
      setDeviceSearchTimedOut(false);
    }, 0);
    const noticeTimeout = setTimeout(() => {
      setDeviceSearchTimedOut(true);
    }, DEVICE_SEARCH_NOTICE_DELAY_MS);
    return () => {
      clearTimeout(resetTimeout);
      clearTimeout(noticeTimeout);
    };
  }, [cameraActive, device]);

  if (!device) {
    return (
      <View style={styles.previewFallback}>
        <Text accessibilityRole="alert" selectable style={styles.previewText}>
          {deviceSearchTimedOut && cameraActive
            ? 'Kamera belakang masih disiapkan. Tunggu sebentar atau akhiri pemindaian lalu coba lagi.'
            : 'Menyiapkan kamera belakang.'}
        </Text>
      </View>
    );
  }

  return (
    <View
      accessibilityElementsHidden
      accessible={false}
      importantForAccessibility="no-hide-descendants"
      style={styles.preview}
      testID="live-camera-preview"
    >
      <Camera
        {...(cameraStarted && device.hasTorch
          ? { torchMode: torchEnabled ? ('on' as const) : ('off' as const) }
          : {})}
        device={device}
        constraints={CAMERA_CONSTRAINTS}
        isActive={cameraActive}
        onError={(error) => onProcessingError(sessionId, error.message)}
        onStarted={() => {
          setCameraStarted(true);
          setDeviceSearchTimedOut(false);
        }}
        onStopped={() => setCameraStarted(false)}
        outputs={cameraOutputs}
        resizeMode="cover"
        style={StyleSheet.absoluteFill}
      />
      <View style={styles.targetArea}>
        <Text style={styles.targetLabel}>SATU SEL</Text>
      </View>
    </View>
  );
}

export default memo(LiveSingleCellCamera);

const styles = StyleSheet.create({
  preview: {
    width: '100%',
    maxWidth: 360,
    aspectRatio: 1,
    alignSelf: 'center',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    backgroundColor: colors.primaryContainer,
    borderCurve: 'continuous',
    borderRadius: radii.lg,
  },
  previewFallback: {
    minHeight: 220,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primaryContainer,
    borderCurve: 'continuous',
    borderRadius: radii.lg,
    padding: spacing.xl,
  },
  previewText: {
    ...typography.body,
    color: colors.text,
    textAlign: 'center',
  },
  targetArea: {
    width: '55%',
    aspectRatio: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
    backgroundColor: 'transparent',
    borderColor: colors.primary,
    borderRadius: radii.md,
    borderStyle: 'dashed',
    borderWidth: 3,
    padding: spacing.sm,
  },
  targetLabel: {
    ...typography.labelCaps,
    backgroundColor: 'rgba(16, 42, 67, 0.92)',
    borderRadius: radii.sm,
    color: colors.onPrimary,
    fontWeight: '700',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
});
