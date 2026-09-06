import { useLocalSearchParams, useRouter } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { updateAppPreferences } from '@/storage';
import { colors, spacing, typography } from '@/theme';

export default function OnboardingScreen() {
  const router = useRouter();
  const { review } = useLocalSearchParams<{ review?: string }>();
  const isReview = review === '1';

  const completeOnboarding = () => {
    updateAppPreferences({ onboardingCompleted: true });
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

        <SectionCard title="Pindai satu kata" variant="plain">
          <Text selectable style={styles.body}>
            Tempatkan satu kata Braille di dalam bingkai, dengan kamera tepat di
            atasnya dan cahaya miring dari ujung atas kertas. Tekan Pindai kata.
            Aplikasi membaca huruf dari kiri ke kanan. Gunakan satu baris A sampai Z,
            maksimal 12 huruf, lalu periksa hasil sebelum memilih Bacakan.
          </Text>
        </SectionCard>

        <SectionCard title="Privasi dan offline" variant="plain">
          <Text selectable style={styles.body}>
            Gambar, huruf, dan kata hasil pemindaian tidak disimpan atau dikirim
            ke cloud. Hanya preferensi aplikasi yang tersimpan di perangkat.
          </Text>
        </SectionCard>

        <SectionCard title="Periksa setiap hasil" variant="plain">
          <Text selectable style={styles.body}>
            Setiap huruf dapat salah. Cocokkan kata dengan sumber Braille atau
            minta bantuan orang yang memahaminya sebelum memakai hasil untuk
            keputusan penting.
          </Text>
        </SectionCard>
      </Screen>

      <SafeAreaView edges={['bottom']} style={styles.footer}>
        <View style={styles.footerContent} testID="onboarding-footer-content">
          <AccessibleButton
            label={isReview ? 'Selesai' : 'Buka Beranda'}
            onPress={completeOnboarding}
            variant="action"
          />
        </View>
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
  footerContent: {
    width: '100%',
    maxWidth: 760,
    alignSelf: 'center',
  },
});
