import { useLocalSearchParams, useRouter } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { updateAlphaPreferences } from '@/storage';
import { colors, spacing, typography } from '@/theme';

export default function OnboardingScreen() {
  const router = useRouter();
  const { review } = useLocalSearchParams<{ review?: string }>();
  const isReview = review === '1';

  const completeOnboarding = () => {
    updateAlphaPreferences({ onboardingCompleted: true });
    if (isReview && router.canGoBack()) {
      router.back();
      return;
    }
    router.replace('/');
  };

  return (
    <View style={styles.container} testID="onboarding-screen">
      <Screen
        contentContainerStyle={styles.content}
        testID="onboarding-content"
      >
        <Text selectable style={styles.intro}>
          Tiga hal penting sebelum mulai.
        </Text>

        <SectionCard title="Kemampuan saat ini">
          <Text selectable style={styles.body}>
            Pemindai kamera satu sel tersedia untuk kandidat huruf A sampai Z.
            Ia belum dapat membaca kata, spasi, atau satu baris Braille.
          </Text>
        </SectionCard>

        <SectionCard title="Privasi dan offline">
          <Text selectable style={styles.body}>
            Gambar dan teks hasil tidak disimpan atau dikirim ke cloud. Hanya
            preferensi aplikasi yang tersimpan di perangkat.
          </Text>
        </SectionCard>

        <SectionCard title="Batas dan verifikasi">
          <Text selectable style={styles.body}>
            Kandidat model dapat salah dan bukan hasil terverifikasi. Periksa
            bersama manusia sebelum memakai huruf untuk keputusan penting.
          </Text>
        </SectionCard>
      </Screen>

      <SafeAreaView edges={['bottom']} style={styles.footer}>
        <AccessibleButton
          label={isReview ? 'Selesai' : 'Buka Beranda'}
          onPress={completeOnboarding}
          variant="action"
        />
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.canvas,
  },
  content: {
    paddingBottom: spacing.lg,
  },
  intro: {
    ...typography.bodyLarge,
    color: colors.text,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
  footer: {
    backgroundColor: colors.surfaceBright,
    borderTopColor: colors.outlineVariant,
    borderTopWidth: 1,
    paddingBottom: spacing.lg,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
  },
});
