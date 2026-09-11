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

        <SectionCard title="Pindai kata dan angka" variant="plain">
          <Text selectable style={styles.body}>
            Arahkan kamera tepat di atas satu kata atau angka Braille, dengan
            cahaya miring dari ujung atas kertas. Bingkai akan mengikuti posisinya.
            Sertakan tanda kapital, angka, dan tanda baca. Tahan kamera, lalu tekan Pindai.
            Gunakan satu baris, maksimal 12 karakter, lalu periksa hasil sebelum
            memilih Bacakan.
          </Text>
        </SectionCard>

        <SectionCard title="Bisa digunakan tanpa internet" variant="plain">
          <Text selectable style={styles.body}>
            Foto dan hasil pemindaian diproses di perangkat, tanpa disimpan atau
            dikirim. Hanya pengaturan aplikasi yang tersimpan.
          </Text>
        </SectionCard>

        <SectionCard title="Periksa setiap hasil" variant="plain">
          <Text selectable style={styles.body}>
            Cocokkan huruf, angka, dan tanda baca dengan sumber Braille. Jika
            hasilnya belum sesuai, perbaiki posisi atau cahaya lalu pindai ulang.
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
