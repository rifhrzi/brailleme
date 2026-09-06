import { Redirect, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { scanReadinessPort } from '@/services/scan-readiness-port';
import { useAppPreferences } from '@/storage';
import {
  colors,
  minimumTouchTarget,
  radii,
  spacing,
  typography,
} from '@/theme';
import type { ScanReadiness } from '@/types';

function GuideRow({ number, text }: { number: number; text: string }) {
  return (
    <View
      accessibilityLabel={`${number}. ${text}`}
      accessible
      style={styles.guideRow}
    >
      <View accessible={false} style={styles.guideNumber}>
        <Text
          accessible={false}
          maxFontSizeMultiplier={1.25}
          style={styles.guideNumberText}
        >
          {number}
        </Text>
      </View>
      <Text accessible={false} style={styles.guideText}>
        {text}
      </Text>
    </View>
  );
}

const BRAILLE_B_PATTERN = [true, false, true, false, false, false] as const;

export default function HomeScreen() {
  const router = useRouter();
  const preferences = useAppPreferences();
  const [scanReadiness, setScanReadiness] =
    useState<ScanReadiness | null>(null);

  useEffect(() => {
    let active = true;

    void scanReadinessPort
      .getAvailability()
      .then((availability) => {
        if (active) {
          setScanReadiness(availability);
        }
      })
      .catch(() => {
        if (active) {
          setScanReadiness({
            available: false,
            reason: 'scanner_not_available',
            message: 'Status pemindai tidak dapat diperiksa.',
          });
        }
      });

    return () => {
      active = false;
    };
  }, []);

  if (!preferences.onboardingCompleted) {
    return <Redirect href="/onboarding" />;
  }

  const scannerReady = scanReadiness?.available === true;
  const statusTitle = scannerReady
    ? 'Siap memindai kata'
    : scanReadiness === null
      ? 'Memeriksa pemindai'
      : 'Pemindai belum siap';
  const statusDescription = scannerReady
    ? 'Tempatkan satu kata di dalam bingkai, lalu tekan Pindai kata.'
    : scanReadiness === null
      ? 'Sedang memeriksa komponen pemindai di perangkat ini.'
      : scanReadiness.message;

  return (
    <Screen testID="home-screen">
      <View style={styles.brandRow}>
        <View
          accessibilityElementsHidden
          accessible={false}
          style={styles.brandMark}
        >
          {BRAILLE_B_PATTERN.map((filled, index) => (
            <View
              key={index}
              style={[styles.brandDot, !filled && styles.brandDotOutline]}
            />
          ))}
        </View>
        <View style={styles.brandCopy}>
          <Text accessible={false} style={styles.brandName}>
            BrailleMe
          </Text>
          <Text
            accessibilityLabel="Bekerja offline di perangkat"
            style={styles.offlineLabel}
          >
            OFFLINE DI PERANGKAT
          </Text>
        </View>
      </View>

      <View style={styles.hero}>
        <Text accessibilityRole="header" selectable style={styles.title}>
          Pindai satu kata Braille
        </Text>
        <Text selectable style={styles.intro}>
          Ambil foto satu kata. Aplikasi membaca huruf dari kiri ke kanan,
          lalu menampilkan kata untuk diperiksa dan dibacakan. Semua pemrosesan tetap offline.
        </Text>
      </View>

      <View style={styles.scanPanel}>
        <View
          accessibilityLabel={`${statusTitle}. ${statusDescription}`}
          accessibilityLiveRegion="polite"
          accessible
          style={styles.statusRow}
        >
          <View
            accessible={false}
            style={[
              styles.statusIcon,
              scannerReady
                ? styles.statusReady
                : scanReadiness === null
                  ? styles.statusChecking
                  : styles.statusUnavailable,
            ]}
          >
            <Text
              accessible={false}
              maxFontSizeMultiplier={1.25}
              style={styles.statusIconText}
            >
              {scannerReady ? 'OK' : scanReadiness === null ? '...' : '!'}
            </Text>
          </View>
          <View accessible={false} style={styles.statusCopy}>
            <Text accessible={false} style={styles.statusTitle}>
              {statusTitle}
            </Text>
            <Text accessible={false} style={styles.statusDescription}>
              {statusDescription}
            </Text>
          </View>
        </View>
        <AccessibleButton
          accessibilityHint="Membuka kamera untuk memindai satu blok kata Braille"
          disabled={scanReadiness === null}
          label={
            scanReadiness === null
              ? 'Memeriksa pemindai'
              : scannerReady
                ? 'Buka pemindai kata'
                : 'Lihat status pemindai'
          }
          onPress={() => router.replace('/live')}
          style={styles.primaryAction}
          variant={scannerReady ? 'action' : 'secondary'}
        />
        <Text selectable style={styles.disclaimer}>
          Satu baris, huruf A–Z, maksimal 12 huruf. Periksa hasil terhadap sumber Braille.
        </Text>
      </View>

      <SectionCard title="Sebelum memindai" variant="plain">
        <View style={styles.guideList}>
          <GuideRow number={1} text="Letakkan seluruh kata di dalam bingkai." />
          <GuideRow
            number={2}
            text="Tahan kamera, lalu tekan Pindai kata."
          />
          <GuideRow
            number={3}
            text="Periksa hasil, lalu pilih Bacakan atau Pindai ulang."
          />
        </View>
      </SectionCard>

      <SectionCard title="Bantuan dan privasi" variant="plain">
        <View style={styles.links}>
          <AccessibleButton
            accessibilityHint="Membuka panduan dan batas penggunaan"
            label="Bantuan"
            onPress={() => router.push('/help')}
            variant="navigation"
          />
          <AccessibleButton
            accessibilityHint="Membuka kebijakan privasi lengkap"
            label="Privasi"
            onPress={() => router.push('/privacy')}
            variant="navigation"
          />
        </View>
        <Text selectable style={styles.privacyCopy}>
          Gambar dan hasil pemindaian tidak disimpan. Hanya preferensi aplikasi
          yang tersimpan pada perangkat.
        </Text>
      </SectionCard>
    </Screen>
  );
}

const styles = StyleSheet.create({
  brandRow: {
    minHeight: minimumTouchTarget,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  brandMark: {
    width: 25,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 5,
  },
  brandDot: {
    width: 10,
    height: 10,
    backgroundColor: colors.primary,
    borderColor: colors.primary,
    borderRadius: radii.pill,
    borderWidth: 2,
  },
  brandDotOutline: {
    backgroundColor: 'transparent',
  },
  brandCopy: {
    flex: 1,
    gap: 2,
  },
  brandName: {
    ...typography.headlineMedium,
    color: colors.text,
    fontFamily: typography.title.fontFamily,
  },
  offlineLabel: {
    ...typography.labelCaps,
    color: colors.primary,
  },
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
  scanPanel: {
    gap: spacing.md,
    backgroundColor: colors.surfaceContainerLowest,
    borderColor: colors.outlineVariant,
    borderCurve: 'continuous',
    borderRadius: radii.xl,
    borderWidth: 1,
    padding: spacing.lg,
  },
  statusRow: {
    minHeight: minimumTouchTarget,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.md,
  },
  statusIcon: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radii.pill,
  },
  statusReady: {
    backgroundColor: colors.success,
  },
  statusChecking: {
    backgroundColor: colors.primary,
  },
  statusUnavailable: {
    backgroundColor: colors.warning,
  },
  statusIconText: {
    ...typography.label,
    color: colors.onPrimary,
  },
  statusCopy: {
    flex: 1,
    gap: spacing.xxs,
  },
  statusTitle: {
    ...typography.label,
    color: colors.text,
  },
  statusDescription: {
    ...typography.supporting,
    color: colors.textSecondary,
  },
  primaryAction: {
    minHeight: 60,
  },
  disclaimer: {
    ...typography.supporting,
    color: colors.secondary,
    fontFamily: typography.label.fontFamily,
  },
  guideList: {
    borderBottomColor: colors.outlineVariant,
    borderTopColor: colors.outlineVariant,
    borderBottomWidth: 1,
    borderTopWidth: 1,
  },
  guideRow: {
    minHeight: 64,
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    borderBottomColor: colors.outlineVariant,
    borderBottomWidth: StyleSheet.hairlineWidth,
    paddingVertical: spacing.sm,
  },
  guideNumber: {
    width: 32,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primaryContainer,
    borderRadius: radii.pill,
  },
  guideNumberText: {
    ...typography.label,
    color: colors.onPrimaryContainer,
  },
  guideText: {
    ...typography.body,
    flex: 1,
    color: colors.text,
  },
  links: {
    gap: spacing.sm,
  },
  privacyCopy: {
    ...typography.supporting,
    color: colors.textSecondary,
  },
});
