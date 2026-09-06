import { useRouter } from 'expo-router';
import { useCallback } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import WordPhotoCamera, {
  type WordPhotoPreviewProps,
} from '@/features/live/word-photo-camera';
import WordScanScreen from '@/features/live/word-scan-screen';
import type { WordDraftState } from '@/features/live/word-draft';
import { useWordModel } from '@/ml/word-model-loader';
import { cameraPermissionPort } from '@/services';
import { spacing, typography } from '@/theme';

type LiveRuntimeScreenProps = {
  readonly onWordDraftChange?:
    ((nextDraft: WordDraftState) => void) | undefined;
  readonly wordDraft?: WordDraftState | undefined;
};

export default function LiveRuntimeScreen({
  onWordDraftChange,
  wordDraft,
}: LiveRuntimeScreenProps = {}) {
  const router = useRouter();
  const modelState = useWordModel();
  const leaveLive = useCallback(() => {
    if (router.canGoBack()) {
      router.back();
      return;
    }
    router.replace('/');
  }, [router]);
  const renderScannerPreview = useCallback(
    (props: WordPhotoPreviewProps) =>
      modelState.status === 'ready' ? (
        <WordPhotoCamera
          {...props}
          key={props.sessionId}
          model={modelState.model}
          thresholds={modelState.thresholds}
          modelVersion={modelState.modelVersion}
          acquireModel={modelState.acquireModel}
        />
      ) : null,
    [modelState],
  );

  if (modelState.status === 'loading') {
    return (
      <Screen testID="live-model-loading">
        <SectionCard title="Menyiapkan pemindai kata">
          <Text
            accessibilityLiveRegion="polite"
            selectable
            style={typography.body}
          >
            Menyiapkan pemindai sebelum kamera dinyalakan.
          </Text>
        </SectionCard>
      </Screen>
    );
  }

  if (modelState.status === 'error') {
    return (
      <Screen testID="live-model-error">
        <SectionCard title="Pemindai belum siap">
          <Text accessibilityRole="alert" selectable style={typography.body}>
            Pemindai belum dapat disiapkan. Coba lagi. Jika masalah berlanjut,
            buka Bantuan.
          </Text>
          <View accessible={false} style={styles.actions}>
            <AccessibleButton
              accessibilityHint="Menyiapkan ulang pemindai"
              label="Coba lagi"
              onPress={modelState.retry}
            />
            <AccessibleButton
              accessibilityHint="Membuka petunjuk penanganan masalah"
              label="Bantuan"
              onPress={() => router.push('/help')}
              variant="secondary"
            />
            <AccessibleButton
              accessibilityHint="Kembali ke layar sebelumnya"
              label="Kembali"
              onPress={leaveLive}
              variant="secondary"
            />
          </View>
        </SectionCard>
      </Screen>
    );
  }

  return (
    <WordScanScreen
      cameraPermission={cameraPermissionPort}
      onWordDraftChange={onWordDraftChange}
      renderCamera={renderScannerPreview}
      wordDraft={wordDraft}
    />
  );
}

const styles = StyleSheet.create({
  actions: {
    gap: spacing.sm,
  },
});
