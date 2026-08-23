import { Redirect, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { scannerPort } from '@/services';
import { useAlphaPreferences } from '@/storage';
import { colors, radii, spacing, typography } from '@/theme';
import type { ScannerAvailability } from '@/types';

export default function HomeScreen() {
  const router = useRouter();
  const preferences = useAlphaPreferences();
  const [scannerAvailability, setScannerAvailability] =
    useState<ScannerAvailability | null>(null);

  useEffect(() => {
    let active = true;

    void scannerPort
      .getAvailability()
      .then((availability) => {
        if (active) {
          setScannerAvailability(availability);
        }
      })
      .catch(() => {
        if (active) {
          setScannerAvailability({
            available: false,
            reason: 'native_scanner_not_installed',
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

  return (
    <Screen testID="home-screen">
      <View style={styles.hero}>
        <Text accessible={false} style={styles.eyebrow}>
          PRIBADI · OFFLINE
        </Text>
        <Text accessibilityRole="header" selectable style={styles.title}>
          Periksa Braille dengan lebih tenang
        </Text>
        <Text selectable style={styles.intro}>
          Navigasi, status, hasil teks, dan kontrol suara dirancang sederhana
          serta mudah digunakan dengan pembaca layar.
        </Text>
      </View>

      <SectionCard
        description={
          scannerAvailability === null
            ? 'Sedang memeriksa komponen pemindai di perangkat ini.'
            : scannerAvailability.available
              ? 'Fitur pemindai satu sel dikonfigurasi. Kamera dan model diperiksa ketika layar pemindai dibuka. Hasil masih eksperimental dan wajib diverifikasi.'
              : 'Komponen pemindai satu sel tidak dapat dimuat. Buka status untuk melihat penjelasan.'
        }
        title={
          scannerAvailability?.available
            ? 'Pemindai satu sel dikonfigurasi'
            : scannerAvailability === null
              ? 'Memeriksa pemindai'
              : 'Pemindai belum tersedia'
        }
      >
        <View
          accessibilityLabel={
            scannerAvailability?.available
              ? 'Status fitur pemindai satu sel eksperimental dikonfigurasi; kamera dan model diperiksa saat dibuka'
              : scannerAvailability === null
                ? 'Status sedang diperiksa'
                : 'Status belum tersedia'
          }
          accessible
          style={styles.statusRow}
        >
          <View
            accessible={false}
            style={[
              styles.statusDot,
              scannerAvailability?.available
                ? styles.statusReady
                : styles.statusUnavailable,
            ]}
          />
          <Text accessible={false} style={styles.statusLabel}>
            {scannerAvailability?.available
              ? 'Siap diperiksa, eksperimental'
              : scannerAvailability === null
                ? 'Memeriksa'
                : 'Belum tersedia'}
          </Text>
        </View>
        <AccessibleButton
          accessibilityHint="Membuka layar pemindai dan rincian statusnya"
          disabled={scannerAvailability === null}
          label={
            scannerAvailability === null
              ? 'Memeriksa pemindai'
              : scannerAvailability.available
                ? 'Pindai satu sel'
                : 'Lihat status pemindai'
          }
          onPress={() => router.push('/live')}
          style={styles.primaryAction}
          variant={scannerAvailability?.available ? 'action' : 'secondary'}
        />
      </SectionCard>

      <View style={styles.navigationGroup}>
        <Text accessibilityRole="header" style={styles.sectionTitle}>
          Akses cepat
        </Text>
        <View style={styles.actions}>
          <AccessibleButton
            accessibilityHint="Mengatur suara, haptik, dan opsi aplikasi"
            label="Pengaturan"
            onPress={() => router.push('/settings')}
            variant="navigation"
          />
          <AccessibleButton
            accessibilityHint="Membuka panduan dan batas penggunaan"
            label="Bantuan"
            onPress={() => router.push('/help')}
            variant="navigation"
          />
        </View>
      </View>

      <SectionCard title="Privasi" variant="plain">
        <Text selectable style={styles.body}>
          Gambar dan teks hasil tidak disimpan. Hanya preferensi aplikasi yang
          tersimpan pada perangkat.
        </Text>
        <AccessibleButton
          accessibilityHint="Membuka kebijakan privasi lengkap"
          label="Kebijakan privasi"
          onPress={() => router.push('/privacy')}
          variant="secondary"
        />
      </SectionCard>
    </Screen>
  );
}

const styles = StyleSheet.create({
  hero: {
    gap: spacing.sm,
    paddingBottom: spacing.xs,
  },
  eyebrow: {
    ...typography.labelCaps,
    color: colors.primary,
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
  actions: {
    gap: spacing.sm,
  },
  primaryAction: {
    minHeight: 56,
  },
  navigationGroup: {
    gap: spacing.sm,
  },
  sectionTitle: {
    ...typography.headlineMedium,
    color: colors.text,
  },
  statusRow: {
    minHeight: 32,
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: spacing.xs,
    backgroundColor: colors.surfaceContainerLow,
    borderRadius: radii.pill,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: radii.pill,
  },
  statusReady: {
    backgroundColor: colors.success,
  },
  statusUnavailable: {
    backgroundColor: colors.warning,
  },
  statusLabel: {
    ...typography.supporting,
    color: colors.text,
    fontFamily: typography.label.fontFamily,
  },
});
