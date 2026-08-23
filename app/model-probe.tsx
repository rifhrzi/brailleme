import { lazy, Suspense } from 'react';
import { Text } from 'react-native';
import {
  Redirect,
  useRouter,
  type ErrorBoundaryProps,
} from 'expo-router';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { useAlphaPreferences } from '@/storage';
import { typography } from '@/theme';

// Modul kamera dan TFLite sengaja baru dievaluasi ketika eksperimen dibuka.
// Ini menjaga alur utama tetap ringan dan mengisolasi native runtime debug.
const loadModelProbeScreen = () =>
  import('@/features/model-probe/model-probe-screen');
let ModelProbeScreen = lazy(loadModelProbeScreen);

function LoadingProbe() {
  return (
    <Screen testID="model-probe-route-loading">
      <SectionCard title="Menyiapkan uji model">
        <Text accessibilityLiveRegion="polite" style={typography.body}>
          Memuat kamera dan model eksperimen.
        </Text>
      </SectionCard>
    </Screen>
  );
}

export function ErrorBoundary({ retry }: ErrorBoundaryProps) {
  const router = useRouter();
  const retryImport = () => {
    ModelProbeScreen = lazy(loadModelProbeScreen);
    retry();
  };

  return (
    <Screen testID="model-probe-route-error">
      <SectionCard title="Uji model tidak dapat dibuka">
        <Text accessibilityRole="alert" selectable style={typography.body}>
          Komponen pengujian mengalami masalah. Tidak ada hasil yang dibuat.
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

export default function ModelProbeRoute() {
  const preferences = useAlphaPreferences();

  if (!preferences.onboardingCompleted) {
    return <Redirect href="/onboarding" />;
  }
  if (!__DEV__) {
    return <Redirect href="/" />;
  }

  return (
    <Suspense fallback={<LoadingProbe />}>
      <ModelProbeScreen />
    </Suspense>
  );
}
