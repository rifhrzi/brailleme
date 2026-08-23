import { Redirect, useRouter } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { StyleSheet, Text } from 'react-native';

import {
  AccessibleButton,
  ReadinessRow,
  Screen,
  SectionCard,
  SettingRow,
} from '@/components';
import { INTERNAL_FEATURE_FLAGS } from '@/config';
import {
  cameraPermissionPort,
  readinessPort,
  scannerPort,
  ttsPort,
  type CameraPermissionPort,
  type ReadinessPort,
  type ScannerPort,
  type TtsPort,
} from '@/services';
import { useAlphaPreferences } from '@/storage';
import { colors, typography } from '@/theme';
import type {
  CameraPermissionSnapshot,
  ReadinessSnapshot,
  ReadinessStatus,
  ScannerAvailability,
  TtsAvailability,
} from '@/types';

type DebugScreenViewProps = {
  cameraPermission?: CameraPermissionPort;
  readiness?: ReadinessPort;
  scanner?: ScannerPort;
  tts?: TtsPort;
};

type DiagnosticsSnapshot = {
  readiness: ReadinessSnapshot;
  scanner: ScannerAvailability;
  permission: CameraPermissionSnapshot;
  tts: TtsAvailability;
};

const readinessTone: Record<
  ReadinessStatus,
  'success' | 'warning' | 'error' | 'neutral'
> = {
  ready: 'success',
  needs_attention: 'warning',
  unavailable: 'error',
};

const readinessValue: Record<ReadinessStatus, string> = {
  ready: 'Siap',
  needs_attention: 'Perlu tindakan',
  unavailable: 'Belum tersedia',
};

export function DebugScreenView({
  cameraPermission = cameraPermissionPort,
  readiness = readinessPort,
  scanner = scannerPort,
  tts = ttsPort,
}: DebugScreenViewProps) {
  const router = useRouter();
  const preferences = useAlphaPreferences();
  const [diagnostics, setDiagnostics] = useState<DiagnosticsSnapshot | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [diagnosticError, setDiagnosticError] = useState('');
  const requestSequenceRef = useRef(0);

  const readDiagnostics = useCallback(async () => {
    if (!preferences.debugMenuEnabled) {
      return null;
    }

    const [nextReadiness, nextScanner, nextPermission, nextTts] =
      await Promise.all([
        readiness.check(),
        scanner.getAvailability(),
        cameraPermission.getStatus(),
        tts.getAvailability(),
      ]);

    return {
      readiness: nextReadiness,
      scanner: nextScanner,
      permission: nextPermission,
      tts: nextTts,
    } satisfies DiagnosticsSnapshot;
  }, [
    cameraPermission,
    preferences.debugMenuEnabled,
    readiness,
    scanner,
    tts,
  ]);

  const refreshDiagnostics = useCallback(async () => {
    const requestSequence = ++requestSequenceRef.current;
    setLoading(true);
    setDiagnosticError('');

    try {
      const nextDiagnostics = await readDiagnostics();
      if (requestSequenceRef.current !== requestSequence) {
        return;
      }
      setDiagnostics(nextDiagnostics);
    } catch {
      if (requestSequenceRef.current === requestSequence) {
        setDiagnostics(null);
        setDiagnosticError('Diagnostik tidak dapat diperiksa. Coba lagi.');
      }
    } finally {
      if (requestSequenceRef.current === requestSequence) {
        setLoading(false);
      }
    }
  }, [readDiagnostics]);

  useEffect(() => {
    const requestSequence = ++requestSequenceRef.current;

    void readDiagnostics()
      .then((nextDiagnostics) => {
        if (requestSequenceRef.current === requestSequence) {
          setDiagnostics(nextDiagnostics);
        }
      })
      .catch(() => {
        if (requestSequenceRef.current === requestSequence) {
          setDiagnostics(null);
          setDiagnosticError('Diagnostik tidak dapat diperiksa. Coba lagi.');
        }
      })
      .finally(() => {
        if (requestSequenceRef.current === requestSequence) {
          setLoading(false);
        }
      });

    return () => {
      requestSequenceRef.current += 1;
    };
  }, [readDiagnostics]);

  if (!preferences.debugMenuEnabled) {
    return <Redirect href="/settings" />;
  }

  return (
    <Screen testID="debug-screen">
      <Text selectable style={styles.intro}>
        Informasi pada layar ini ditujukan untuk pengembangan dan pengujian.
        Menu diagnostik tidak mengaktifkan kamera, pengenalan Braille, atau suara.
      </Text>

      <SectionCard title="Kesiapan komponen">
        {diagnostics ? (
          <>
            <ReadinessRow
              detail={diagnostics.readiness.model.message}
              label="Model pemindai"
              tone={readinessTone[diagnostics.readiness.model.status]}
              value={readinessValue[diagnostics.readiness.model.status]}
            />
            <ReadinessRow
              detail={diagnostics.readiness.camera.message}
              label="Kamera"
              tone={readinessTone[diagnostics.readiness.camera.status]}
              value={readinessValue[diagnostics.readiness.camera.status]}
            />
            <ReadinessRow
              detail={diagnostics.readiness.tts.message}
              label="TTS lokal"
              tone={readinessTone[diagnostics.readiness.tts.status]}
              value={readinessValue[diagnostics.readiness.tts.status]}
            />
          </>
        ) : (
          <Text accessibilityLiveRegion="polite" selectable style={styles.body}>
            Memeriksa diagnostik.
          </Text>
        )}
      </SectionCard>

      {diagnosticError ? (
        <Text
          accessibilityLiveRegion="polite"
          accessibilityRole="alert"
          selectable
          style={styles.error}
        >
          {diagnosticError}
        </Text>
      ) : null}

      <SectionCard title="Kontrak runtime">
        <SettingRow
          description="Profil internal yang dikunci untuk Technical Alpha."
          label="Profil Braille"
          value={preferences.profile}
        />
        <SettingRow
          description="Auto-speak tetap merupakan flag internal dan tidak tersedia pada UI pengguna."
          label="Mode suara"
          value={`${preferences.speechMode}; auto-speak ${INTERNAL_FEATURE_FLAGS.autoSpeakEnabled ? 'aktif' : 'nonaktif'}`}
        />
        <SettingRow
          label="ScannerPort"
          value={
            diagnostics
              ? diagnostics.scanner.available
                ? 'available'
                : `unavailable: ${diagnostics.scanner.reason}`
              : 'memeriksa'
          }
        />
        <SettingRow
          label="CameraPermissionPort"
          value={
            diagnostics
              ? `${diagnostics.permission.status}; canAskAgain=${String(diagnostics.permission.canAskAgain)}`
              : 'memeriksa'
          }
        />
        <SettingRow
          label="TtsPort"
          value={
            diagnostics
              ? `${diagnostics.tts.status}; localVoice=${String(diagnostics.tts.localVoiceAvailable)}`
              : 'memeriksa'
          }
        />
      </SectionCard>

      <SectionCard title="Scope build">
        <Text selectable style={styles.body}>
          Build ini menyediakan kamera dan classifier satu sel A sampai Z pada
          layar Pindai Satu Sel. Segmentasi kata atau baris belum tersedia.
        </Text>
        <Text selectable style={styles.body}>
          Mode Dokumen, editor, riwayat, penyimpanan hasil, salin, bagikan, dan
          ekspor tidak aktif pada build ini.
        </Text>
      </SectionCard>

      <SectionCard
        description="Klasifikasi kamera untuk satu huruf A sampai Z. Bukan penerjemah kata atau baris."
        title="Eksperimen model"
      >
        <AccessibleButton
          accessibilityHint="Membuka eksperimen kamera satu sel Braille"
          label="Buka uji model satu sel"
          onPress={() => router.push('/model-probe')}
          variant="secondary"
        />
      </SectionCard>

      <AccessibleButton
        accessibilityHint="Memeriksa ulang status seluruh adapter runtime"
        disabled={loading}
        label={loading ? 'Memeriksa diagnostik' : 'Muat ulang diagnostik'}
        onPress={() => void refreshDiagnostics()}
        variant="secondary"
      />
    </Screen>
  );
}

export default function DebugScreen() {
  return <DebugScreenView />;
}

const styles = StyleSheet.create({
  intro: {
    ...typography.bodyLarge,
    color: colors.textSecondary,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
  error: {
    ...typography.body,
    color: colors.error,
  },
});
