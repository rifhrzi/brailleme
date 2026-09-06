import {
  lazy,
  Suspense,
  useCallback,
  useState,
  type ReactNode,
} from 'react';
import { StyleSheet, Text, View } from 'react-native';
import {
  Redirect,
  useFocusEffect,
  useRouter,
  type ErrorBoundaryProps,
} from 'expo-router';

import {
  AccessibleButton,
  PrimaryPage,
  Screen,
  SectionCard,
} from '@/components';
import {
  INITIAL_WORD_DRAFT_STATE,
  type WordDraftState,
} from '@/features/live/word-draft';
import { useAppPreferences } from '@/storage';
import { colors, radii, spacing, typography } from '@/theme';

// VisionCamera dan TFLite baru dievaluasi setelah pengguna membuka scanner.
// Ini mempertahankan cold-start rute utama yang stabil.
const loadLiveRuntimeScreen = () =>
  import('@/features/live/live-runtime-screen');
let LiveRuntimeScreen = lazy(loadLiveRuntimeScreen);

function LoadingScanner() {
  return (
    <Screen testID="live-route-loading">
      <SectionCard title="Membuka pemindai">
        <Text accessibilityLiveRegion="polite" selectable style={typography.body}>
          Menyiapkan komponen kamera.
        </Text>
      </SectionCard>
    </Screen>
  );
}

export function ErrorBoundary({ retry }: ErrorBoundaryProps) {
  const router = useRouter();
  const retryImport = () => {
    // React.lazy menyimpan Promise yang rejected. Buat wrapper baru agar retry
    // benar-benar mencoba import ulang, bukan langsung melempar error lama.
    LiveRuntimeScreen = lazy(loadLiveRuntimeScreen);
    retry();
  };

  return (
    <Screen testID="live-route-error">
      <SectionCard title="Pemindai tidak dapat dibuka">
        <Text accessibilityRole="alert" selectable style={typography.body}>
          Tidak ada gambar atau hasil yang dibuat. Coba buka kembali pemindai.
        </Text>
        <AccessibleButton
          label="Coba lagi"
          onPress={retryImport}
          variant="secondary"
        />
        <AccessibleButton
          label="Kembali ke Beranda"
          onPress={() => router.replace('/')}
          variant="secondary"
        />
      </SectionCard>
    </Screen>
  );
}

export type LiveRuntimeRenderProps = {
  readonly onWordDraftChange: (nextDraft: WordDraftState) => void;
  readonly wordDraft: WordDraftState;
};

type LiveRouteViewProps = {
  renderRuntime?: ((props: LiveRuntimeRenderProps) => ReactNode) | undefined;
};

function renderLiveRuntime({
  onWordDraftChange,
  wordDraft,
}: LiveRuntimeRenderProps) {
  return (
    <Suspense fallback={<LoadingScanner />}>
      <LiveRuntimeScreen
        onWordDraftChange={onWordDraftChange}
        wordDraft={wordDraft}
      />
    </Suspense>
  );
}

export function LiveRouteView({
  renderRuntime = renderLiveRuntime,
}: LiveRouteViewProps) {
  const preferences = useAppPreferences();
  const [activationConfirmed, setActivationConfirmed] = useState(false);
  const [wordDraft, setWordDraft] = useState<WordDraftState>(
    INITIAL_WORD_DRAFT_STATE,
  );

  useFocusEffect(
    useCallback(
      () => () => {
        setActivationConfirmed(false);
        setWordDraft(INITIAL_WORD_DRAFT_STATE);
      },
      [],
    ),
  );

  if (!preferences.onboardingCompleted) {
    return <Redirect href="/onboarding" />;
  }

  if (!activationConfirmed) {
    return (
      <Screen testID="live-activation-gate">
        <View style={styles.hero}>
          <Text accessibilityRole="header" selectable style={styles.title}>
            Pindai satu kata
          </Text>
          <Text selectable style={styles.intro}>
            Kamera hanya aktif setelah Anda menyetujuinya pada setiap kunjungan.
            Satu foto digunakan untuk membaca seluruh kata dari kiri ke kanan.
          </Text>
        </View>
        <SectionCard title="Sebelum kamera aktif">
          <Text selectable style={styles.body}>
            Kamera dan pemrosesan offline dimulai setelah Anda menekan tombol di
            bawah. Tempatkan satu kata utuh dalam kotak lalu tekan Pindai kata.
            Maksimal 12 huruf A sampai Z dalam satu baris. Foto dan hasil tidak
            disimpan atau dikirim.
          </Text>
          <View style={styles.warning}>
            <Text selectable style={styles.warningText}>
              Hasil deteksi untuk setiap huruf A sampai Z dapat salah. Sistem
              pengenalan belum lulus validasi fisik independen; periksa setiap
              huruf dan kata bersama sumber Braille sebelum digunakan.
            </Text>
          </View>
          <AccessibleButton
            accessibilityHint="Membuka pratinjau kamera untuk memindai satu foto kata"
            label="Aktifkan kamera"
            onPress={() => setActivationConfirmed(true)}
            variant="action"
          />
        </SectionCard>
      </Screen>
    );
  }

  return renderRuntime({
    onWordDraftChange: setWordDraft,
    wordDraft,
  });
}

const styles = StyleSheet.create({
  hero: {
    gap: spacing.sm,
  },
  title: {
    ...typography.displayLargeMobile,
    color: colors.text,
  },
  intro: {
    ...typography.bodyLarge,
    color: colors.textSecondary,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
  warning: {
    backgroundColor: colors.warningContainer,
    borderColor: colors.secondary,
    borderCurve: 'continuous',
    borderRadius: radii.md,
    borderWidth: 1,
    padding: spacing.md,
  },
  warningText: {
    ...typography.body,
    color: colors.onSecondaryContainer,
  },
});

export default function LiveRoute() {
  return (
    <PrimaryPage page="scan">
      <LiveRouteView />
    </PrimaryPage>
  );
}
