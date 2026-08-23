import { lazy, Suspense, useState, type ReactNode } from 'react';
import { Text } from 'react-native';
import {
  Redirect,
  useRouter,
  type ErrorBoundaryProps,
} from 'expo-router';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { useAlphaPreferences } from '@/storage';
import { typography } from '@/theme';

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

type LiveRouteViewProps = {
  renderRuntime?: (() => ReactNode) | undefined;
};

function renderLiveRuntime() {
  return (
    <Suspense fallback={<LoadingScanner />}>
      <LiveRuntimeScreen />
    </Suspense>
  );
}

export function LiveRouteView({
  renderRuntime = renderLiveRuntime,
}: LiveRouteViewProps) {
  const preferences = useAlphaPreferences();
  const [activationConfirmed, setActivationConfirmed] = useState(false);

  if (!preferences.onboardingCompleted) {
    return <Redirect href="/onboarding" />;
  }

  if (!activationConfirmed) {
    return (
      <Screen testID="live-activation-gate">
        <SectionCard title="Aktifkan kamera untuk memindai?">
          <Text selectable style={typography.body}>
            Kamera dan pemrosesan model baru dimulai setelah Anda menekan tombol
            di bawah. Pemrosesan berlangsung langsung di perangkat; gambar dan
            hasil tidak disimpan atau dikirim.
          </Text>
          <Text selectable style={typography.body}>
            Pemindai ini hanya membuat kandidat satu huruf Braille A sampai Z
            dan hasilnya dapat salah. Model belum lulus validasi fisik
            independen; jangan gunakan hasil tanpa verifikasi manusia untuk
            keputusan penting.
          </Text>
          <AccessibleButton
            accessibilityHint="Membuka kamera dan memulai pemrosesan satu sel di perangkat"
            label="Aktifkan kamera dan mulai"
            onPress={() => setActivationConfirmed(true)}
            variant="action"
          />
        </SectionCard>
      </Screen>
    );
  }

  return renderRuntime();
}

export default function LiveRoute() {
  return <LiveRouteView />;
}
