import { useFocusEffect } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AppState,
  Linking,
  StyleSheet,
  Switch,
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
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { createSynchronizable } from 'react-native-worklets';

import {
  AccessibleButton,
  Screen,
  SectionCard,
  StatusAnnouncer,
} from '@/components';
import {
  BRAILLE_LABELS,
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_SCALE,
  BRAILLE_MODEL_INPUT_WIDTH,
  INITIAL_PREDICTION_STATE,
  advancePredictionStabilizer,
  convertRawPixelsToRgbTensor,
  decodeProbabilities,
  useBrailleModel,
  type ModelPrediction,
  type PredictionStabilizerState,
} from '@/ml';
import {
  colors,
  minimumTouchTarget,
  radii,
  spacing,
  typography,
} from '@/theme';

type ProbeStatus =
  | 'searching'
  | 'quality_issue'
  | 'candidate'
  | 'stable'
  | 'error';

type ProbePresentation = {
  readonly status: ProbeStatus;
  readonly title: string;
  readonly guidance?: string;
};

type StableResult = {
  readonly letter: string;
  readonly confidence: number;
  readonly key: string;
};

type Diagnostic = {
  /** Optional so an in-place development refresh cannot crash on older state. */
  readonly candidate?: string;
  readonly confidence: number;
  readonly margin: number;
  readonly meanLuminance: number;
  readonly contrast: number;
};

type ProbeWorkletMessage = {
  readonly sequence: number;
  readonly kind: 'idle' | 'inference' | 'error';
  readonly index: number;
  readonly confidence: number;
  readonly runnerUpConfidence: number;
  readonly meanLuminance: number;
  readonly contrast: number;
  readonly timestampMs: number;
  readonly message: string;
};

const PROBE_TARGET_RESOLUTION = { width: 640, height: 480 };
const PROBE_CONSTRAINTS = [{ fps: 30 }];
const PROBE_INTERVAL_MS = 200;
const CENTER_CROP_FRACTION = 0.55;
const INITIAL_WORKLET_MESSAGE: ProbeWorkletMessage = {
  sequence: 0,
  kind: 'idle',
  index: -1,
  confidence: 0,
  runnerUpConfidence: 0,
  meanLuminance: 0,
  contrast: 0,
  timestampMs: 0,
  message: '',
};

const SEARCHING_PRESENTATION: ProbePresentation = {
  status: 'searching',
  title: 'Mencari satu sel Braille',
  guidance: 'Tempatkan satu huruf Braille di dalam kotak, lalu tahan kamera.',
};

function SingleCellCamera({
  active,
  model,
  onInference,
  onProcessingError,
  torchEnabled,
}: {
  readonly active: boolean;
  readonly model: TensorflowModel;
  readonly onInference: (
    index: number,
    confidence: number,
    runnerUpConfidence: number,
    meanLuminance: number,
    contrast: number,
    timestampMs: number,
  ) => void;
  readonly onProcessingError: (message: string) => void;
  readonly torchEnabled: boolean;
}) {
  const device = useCameraDevice('back');
  const resultChannel = useMemo(
    () => createSynchronizable<ProbeWorkletMessage>(INITIAL_WORKLET_MESSAGE),
    [],
  );
  const lastProcessedAt = useMemo(() => createSynchronizable(0), []);
  const lastMessageSequenceRef = useRef(0);

  useEffect(() => {
    if (!active) {
      lastMessageSequenceRef.current = resultChannel.getBlocking().sequence;
      return;
    }

    const interval = setInterval(() => {
      const message = resultChannel.getBlocking();

      if (message.sequence === lastMessageSequenceRef.current) {
        return;
      }

      lastMessageSequenceRef.current = message.sequence;
      if (message.kind === 'error') {
        onProcessingError(message.message);
        return;
      }
      if (message.kind === 'inference') {
        onInference(
          message.index,
          message.confidence,
          message.runnerUpConfidence,
          message.meanLuminance,
          message.contrast,
          message.timestampMs,
        );
      }
    }, PROBE_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [active, onInference, onProcessingError, resultChannel]);

  // Tetap satu-hop di thread frame: nested worker mengubah worklet impor menjadi
  // object pada kombinasi Worklets/VisionCamera ini. Throttle menjaga biaya CPU.
  const frameOutput = useFrameOutput({
    dropFramesWhileBusy: true,
    enablePhysicalBufferRotation: true,
    pixelFormat: 'yuv',
    targetResolution: PROBE_TARGET_RESOLUTION,
    onFrame(frame) {
      'worklet';

      let sourceImage: Image | undefined;
      let croppedImage: Image | undefined;
      let resizedImage: Image | undefined;
      let frameNeedsDispose = true;
      let stage = 'membatasi laju inferensi';
      const timestampMs = performance.now();

      try {
        const previousTimestampMs = lastProcessedAt.getBlocking();
        if (timestampMs - previousTimestampMs < PROBE_INTERVAL_MS) {
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
        stage = 'menjalankan model';
        const [outputBuffer] = model.runSync([preprocessed.tensor.buffer]);

        stage = 'memeriksa output model';
        if (!outputBuffer) {
          throw new Error('Model tidak menghasilkan output.');
        }

        const prediction = decodeProbabilities(new Float32Array(outputBuffer));
        if (!prediction) {
          throw new Error('Model menghasilkan probabilitas tidak valid.');
        }

        stage = 'mengirim hasil inferensi';
        resultChannel.setBlocking({
          sequence: timestampMs,
          kind: 'inference',
          index: prediction.index,
          confidence: prediction.confidence,
          runnerUpConfidence: prediction.runnerUpConfidence,
          meanLuminance: preprocessed.meanLuminance,
          contrast: preprocessed.contrast,
          timestampMs,
          message: '',
        });
      } catch (error) {
        resultChannel.setBlocking({
          sequence: timestampMs,
          kind: 'error',
          index: -1,
          confidence: 0,
          runnerUpConfidence: 0,
          meanLuminance: 0,
          contrast: 0,
          timestampMs,
          message: `Gagal saat ${stage}: ${String(error)}`,
        });
      } finally {
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

  if (!device) {
    return (
      <View style={styles.previewFallback}>
        <Text style={styles.previewFallbackText}>Kamera belakang tidak ditemukan.</Text>
      </View>
    );
  }

  return (
    <View
      accessibilityElementsHidden
      accessible={false}
      importantForAccessibility="no-hide-descendants"
      style={styles.preview}
    >
      <Camera
        constraints={PROBE_CONSTRAINTS}
        device={device}
        isActive={active}
        onError={(error) => onProcessingError(error.message)}
        outputs={cameraOutputs}
        resizeMode="cover"
        style={StyleSheet.absoluteFill}
        torchMode={torchEnabled ? 'on' : 'off'}
      />
      <View style={styles.targetArea}>
        <Text style={styles.targetLabel}>SATU SEL</Text>
      </View>
    </View>
  );
}

function ReadyModelProbe({ model }: { readonly model: TensorflowModel }) {
  const permission = useCameraPermission();
  const device = useCameraDevice('back');
  const stabilizerRef = useRef<PredictionStabilizerState>(
    INITIAL_PREDICTION_STATE,
  );
  const resultSequenceRef = useRef(0);
  const lastStableLetterRef = useRef('');
  const cameraActiveRef = useRef(false);
  const [isFocused, setIsFocused] = useState(false);
  const [appState, setAppState] = useState<AppStateStatus>(AppState.currentState);
  const [paused, setPaused] = useState(false);
  const [torchEnabled, setTorchEnabled] = useState(false);
  const [presentation, setPresentation] =
    useState<ProbePresentation>(SEARCHING_PRESENTATION);
  const [stableResult, setStableResult] = useState<StableResult | null>(null);
  const [diagnostic, setDiagnostic] = useState<Diagnostic | null>(null);
  const [permissionActionMessage, setPermissionActionMessage] = useState('');

  useFocusEffect(
    useCallback(() => {
      setIsFocused(true);
      return () => {
        cameraActiveRef.current = false;
        setIsFocused(false);
        setTorchEnabled(false);
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
      };
    }, []),
  );

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextState) => {
      setAppState(nextState);
      if (nextState !== 'active') {
        cameraActiveRef.current = false;
        setTorchEnabled(false);
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
      }
    });
    return () => subscription.remove();
  }, []);

  const cameraActive =
    permission.hasPermission &&
    isFocused &&
    appState === 'active' &&
    !paused;

  useEffect(() => {
    cameraActiveRef.current = cameraActive;
  }, [cameraActive]);

  const onProcessingError = useCallback((message: string) => {
    if (!cameraActiveRef.current) {
      return;
    }

    stabilizerRef.current = INITIAL_PREDICTION_STATE;
    setPresentation({
      status: 'error',
      title: 'Uji model mengalami masalah',
      guidance: message,
    });
  }, []);

  const onInference = useCallback(
    (
      index: number,
      confidence: number,
      runnerUpConfidence: number,
      meanLuminance: number,
      contrast: number,
      timestampMs: number,
    ) => {
      if (!cameraActiveRef.current) {
        return;
      }

      const label = BRAILLE_LABELS[index];
      const margin = confidence - runnerUpConfidence;

      if (!label || !Number.isFinite(confidence) || !Number.isFinite(margin)) {
        onProcessingError('Output model tidak valid.');
        return;
      }

      setDiagnostic({
        candidate: label,
        confidence,
        margin,
        meanLuminance,
        contrast,
      });

      if (meanLuminance < 0.08) {
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
        setPresentation({
          status: 'quality_issue',
          title: 'Gambar terlalu gelap',
          guidance: 'Tambah cahaya atau aktifkan senter.',
        });
        return;
      }
      if (meanLuminance > 0.96) {
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
        setPresentation({
          status: 'quality_issue',
          title: 'Cahaya terlalu kuat',
          guidance: 'Kurangi pantulan atau ubah sudut kamera.',
        });
        return;
      }
      if (contrast < 0.035) {
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
        setPresentation({
          status: 'quality_issue',
          title: 'Pola belum terlihat jelas',
          guidance: 'Dekatkan satu sel Braille dan tunggu fokus kamera.',
        });
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
        setPresentation({
          status: 'quality_issue',
          title: 'Belum dapat dipastikan',
          guidance: 'Tahan satu sel di tengah kotak dan perbaiki fokus.',
        });
        return;
      }

      if (outcome.status === 'candidate') {
        setPresentation({
          status: 'candidate',
          title: `Memeriksa kemungkinan huruf ${label.toUpperCase()}`,
        });
        return;
      }

      if (lastStableLetterRef.current !== label) {
        resultSequenceRef.current += 1;
        lastStableLetterRef.current = label;
      }
      setStableResult({
        letter: label,
        confidence,
        key: `${resultSequenceRef.current}:${label}`,
      });
      setPresentation({
        status: 'stable',
        title: `Kandidat stabil: ${label.toUpperCase()}`,
        guidance: 'Verifikasi hasil terhadap sumber Braille sebelum digunakan.',
      });
    },
    [onProcessingError],
  );

  const clearResult = useCallback(() => {
    stabilizerRef.current = INITIAL_PREDICTION_STATE;
    lastStableLetterRef.current = '';
    setStableResult(null);
    setDiagnostic(null);
    setPresentation(SEARCHING_PRESENTATION);
  }, []);

  const requestCameraPermission = useCallback(async () => {
    setPermissionActionMessage('');
    try {
      const granted = await permission.requestPermission();
      if (!granted) {
        setPermissionActionMessage('Izin kamera belum diberikan.');
      }
    } catch {
      setPermissionActionMessage('Izin kamera tidak dapat diminta.');
    }
  }, [permission]);

  const openCameraSettings = useCallback(async () => {
    setPermissionActionMessage('');
    try {
      await Linking.openSettings();
      setPermissionActionMessage('Pengaturan aplikasi dibuka.');
    } catch {
      setPermissionActionMessage('Pengaturan aplikasi tidak dapat dibuka.');
    }
  }, []);

  const togglePause = useCallback(() => {
    setPaused((current) => {
      const next = !current;
      if (next) {
        cameraActiveRef.current = false;
        setTorchEnabled(false);
        stabilizerRef.current = INITIAL_PREDICTION_STATE;
        setPresentation({ status: 'searching', title: 'Uji model dijeda' });
      } else {
        setPresentation(SEARCHING_PRESENTATION);
      }
      return next;
    });
  }, []);

  if (!permission.hasPermission) {
    return (
      <Screen testID="model-probe-screen">
        <SectionCard
          description="Izin hanya digunakan saat layar uji satu sel dibuka. Gambar tidak disimpan."
          title="Izin kamera diperlukan"
        >
          <Text selectable style={styles.body}>
            Berikan akses kamera agar model dapat menerima crop satu sel Braille.
          </Text>
          {permission.canRequestPermission ? (
            <AccessibleButton
              label="Izinkan kamera"
              onPress={() => void requestCameraPermission()}
              variant="action"
            />
          ) : (
            <AccessibleButton
              label="Buka pengaturan aplikasi"
              onPress={() => void openCameraSettings()}
              variant="secondary"
            />
          )}
          {permissionActionMessage ? (
            <Text
              accessibilityLiveRegion="polite"
              accessibilityRole="alert"
              selectable
              style={styles.supporting}
            >
              {permissionActionMessage}
            </Text>
          ) : null}
        </SectionCard>
      </Screen>
    );
  }

  if (!device) {
    return (
      <Screen testID="model-probe-screen">
        <SectionCard title="Kamera belakang tidak tersedia">
          <Text accessibilityRole="alert" selectable style={styles.body}>
            Sambungkan perangkat dengan kamera belakang untuk menjalankan uji
            satu sel.
          </Text>
        </SectionCard>
      </Screen>
    );
  }

  return (
    <Screen testID="model-probe-screen">
      <StatusAnnouncer
        guidance={presentation.guidance}
        stableKey={stableResult?.key}
        status={presentation.status}
        title={presentation.title}
      />

      <Text selectable style={styles.warningText}>
        Eksperimen: arahkan tepat satu sel Braille timbul. Layar ini bukan
        pemindai kata atau satu baris.
      </Text>

      <SingleCellCamera
        active={cameraActive}
        model={model}
        onInference={onInference}
        onProcessingError={onProcessingError}
        torchEnabled={torchEnabled && Boolean(device?.hasTorch)}
      />

      <View
        accessibilityLabel={`${presentation.title}. ${presentation.guidance ?? ''}`}
        accessibilityRole="summary"
        accessible
        style={styles.statusPanel}
      >
        <Text accessible={false} style={styles.statusTitle}>
          {presentation.title}
        </Text>
        {presentation.guidance ? (
          <Text accessible={false} style={styles.body}>
            {presentation.guidance}
          </Text>
        ) : null}
      </View>

      <SectionCard
        description="Kandidat dipertahankan sampai Anda menekan Bersihkan."
        title="Hasil terakhir"
      >
        <Text
          accessibilityLabel={
            stableResult
              ? `Kandidat huruf ${stableResult.letter.toUpperCase()}`
              : 'Belum ada kandidat stabil'
          }
          selectable
          style={styles.result}
        >
          {stableResult?.letter.toUpperCase() ?? '—'}
        </Text>
        <Text selectable style={styles.supporting}>
          {stableResult
            ? `Softmax ${(stableResult.confidence * 100).toFixed(2)}% — diagnostik, belum terkalibrasi.`
            : 'Belum ada hasil yang memenuhi gate sementara.'}
        </Text>
      </SectionCard>

      <SectionCard title="Kontrol">
        <View style={styles.actions}>
          <AccessibleButton
            label={paused ? 'Lanjutkan uji' : 'Jeda uji'}
            onPress={togglePause}
            variant="secondary"
          />
          <AccessibleButton
            disabled={!stableResult && !diagnostic}
            label="Bersihkan"
            onPress={clearResult}
            variant="secondary"
          />
          <View style={styles.switchRow}>
            <Text accessible={false} style={styles.body}>
              Senter
            </Text>
            <Switch
              accessibilityLabel="Senter"
              accessibilityRole="switch"
              accessibilityState={{
                checked: torchEnabled,
                disabled: !device?.hasTorch || paused,
              }}
              disabled={!device?.hasTorch || paused}
              hitSlop={8}
              onValueChange={setTorchEnabled}
              style={styles.switch}
              thumbColor={
                torchEnabled ? colors.primary : colors.surfaceContainerLowest
              }
              trackColor={{
                false: colors.outline,
                true: colors.primary,
              }}
              value={torchEnabled}
            />
          </View>
        </View>
      </SectionCard>

      {diagnostic ? (
        <SectionCard
          description="Angka ini untuk pengembangan, bukan jaminan benar."
          title="Diagnostik frame"
        >
          <Text selectable style={styles.supporting}>
            Top-1 {diagnostic.candidate?.toUpperCase() ?? '—'} · confidence{' '}
            {(diagnostic.confidence * 100).toFixed(2)}% · margin{' '}
            {(diagnostic.margin * 100).toFixed(2)}% · terang{' '}
            {diagnostic.meanLuminance.toFixed(3)} · kontras{' '}
            {diagnostic.contrast.toFixed(3)}
          </Text>
        </SectionCard>
      ) : null}

      <SectionCard title="Batas model">
        <Text selectable style={styles.body}>
          Model hanya memilih huruf A–Z dan tidak memiliki kelas spasi,
          latar, atau “tidak dikenal”. Karena itu hasil stabil tetap dapat salah
          dan harus diverifikasi manusia.
        </Text>
      </SectionCard>
    </Screen>
  );
}

export default function ModelProbeScreen() {
  const modelState = useBrailleModel();

  if (modelState.status === 'loading') {
    return (
      <Screen testID="model-probe-loading">
        <SectionCard title="Menyiapkan model">
          <Text accessibilityLiveRegion="polite" selectable style={styles.body}>
            Memuat model dan menjalankan pemeriksaan tensor…
          </Text>
        </SectionCard>
      </Screen>
    );
  }

  if (modelState.status === 'error') {
    return (
      <Screen testID="model-probe-error">
        <SectionCard title="Model tidak dapat digunakan">
          <Text accessibilityRole="alert" selectable style={styles.body}>
            {modelState.error.message}
          </Text>
        </SectionCard>
      </Screen>
    );
  }

  return <ReadyModelProbe model={modelState.model} />;
}

const styles = StyleSheet.create({
  body: {
    ...typography.body,
    color: colors.text,
  },
  warningText: {
    ...typography.body,
    color: colors.secondary,
    fontWeight: '700',
  },
  preview: {
    height: 360,
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
  previewFallbackText: {
    ...typography.body,
    color: colors.canvas,
    textAlign: 'center',
  },
  targetArea: {
    width: '55%',
    aspectRatio: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
    borderColor: colors.canvas,
    borderRadius: radii.md,
    borderStyle: 'dashed',
    borderWidth: 3,
    padding: spacing.sm,
  },
  targetLabel: {
    ...typography.labelCaps,
    color: colors.canvas,
    fontWeight: '700',
  },
  statusPanel: {
    gap: spacing.xs,
    backgroundColor: colors.surfaceContainerLow,
    borderLeftColor: colors.primary,
    borderLeftWidth: 4,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  statusTitle: {
    ...typography.heading,
    color: colors.text,
  },
  result: {
    ...typography.displayLarge,
    color: colors.text,
    textAlign: 'center',
  },
  supporting: {
    ...typography.supporting,
    color: colors.textSecondary,
  },
  actions: {
    gap: spacing.sm,
  },
  switchRow: {
    minHeight: minimumTouchTarget,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.lg,
  },
  switch: {
    minHeight: minimumTouchTarget,
    minWidth: 56,
  },
});
