import Slider from '@react-native-community/slider';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Alert, AppState, StyleSheet, Switch, Text, View } from 'react-native';

import { AccessibleButton, Screen, SectionCard, SettingRow } from '@/components';
import { ttsPort } from '@/services';
import {
  resetAppPreferences,
  updateAppPreferences,
  useAppPreferences,
} from '@/storage';
import {
  colors,
  minimumTouchTarget,
  radii,
  spacing,
  typography,
} from '@/theme';
import type { TtsAvailability } from '@/types';

const RATE_MIN = 0.5;
const RATE_MAX = 2;
const RATE_STEP = 0.1;
const VOLUME_MIN = 0;
const VOLUME_MAX = 1;
const VOLUME_STEP = 0.1;

function clampStep(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, Math.round(value * 10) / 10));
}

export default function SettingsScreen() {
  const router = useRouter();
  const preferences = useAppPreferences();
  const [voiceAvailability, setVoiceAvailability] =
    useState<TtsAvailability | null>(null);
  const [actionStatus, setActionStatus] = useState('');

  useEffect(() => {
    let active = true;
    let latestAvailabilityRequest = 0;

    const refreshVoiceAvailability = async () => {
      const request = ++latestAvailabilityRequest;

      try {
        const availability = await ttsPort.getAvailability();
        if (active && request === latestAvailabilityRequest) {
          setVoiceAvailability(availability);
        }
      } catch {
        if (active && request === latestAvailabilityRequest) {
          setVoiceAvailability({
            status: 'unavailable',
            localVoiceAvailable: false,
            message: 'Status suara lokal tidak dapat diperiksa.',
          });
        }
      }
    };

    void refreshVoiceAvailability();
    const subscription = AppState.addEventListener('change', (nextState) => {
      if (nextState === 'active') {
        ttsPort.invalidateAvailabilityCache?.();
        void refreshVoiceAvailability();
      }
    });

    return () => {
      active = false;
      latestAvailabilityRequest += 1;
      subscription.remove();
    };
  }, []);

  const changeRate = (next: number) => {
    const speechRate = clampStep(next, RATE_MIN, RATE_MAX);
    updateAppPreferences({
      speechRate,
    });
    setActionStatus(
      `Kecepatan suara ${speechRate.toFixed(1).replace('.', ',')} kali.`,
    );
  };

  const changeVolume = (next: number) => {
    const speechVolume = clampStep(next, VOLUME_MIN, VOLUME_MAX);
    updateAppPreferences({
      speechVolume,
    });
    setActionStatus(`Volume suara ${Math.round(speechVolume * 100)} persen.`);
  };

  const openTtsSettings = async () => {
    try {
      const result = await ttsPort.openTtsSettings();
      setActionStatus(
        result.ok
          ? 'Pengaturan suara Android dibuka.'
          : result.message,
      );
    } catch {
      setActionStatus('Pengaturan suara Android tidak dapat dibuka.');
    }
  };

  const requestResetPreferences = () => {
    Alert.alert(
      'Reset semua preferensi?',
      'Pengenalan, suara, dan getaran akan kembali ke nilai awal.',
      [
        { text: 'Batal', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            resetAppPreferences();
            setActionStatus('Preferensi dikembalikan ke nilai awal.');
          },
        },
      ],
    );
  };

  return (
    <Screen testID="settings-screen">
      <Text accessibilityRole="header" selectable style={styles.title}>
        Pengaturan
      </Text>
      <Text selectable style={styles.intro}>
        Atur cara BrailleMe membacakan hasil dan memberi getaran. Preferensi
        disimpan hanya pada perangkat.
      </Text>

      <SectionCard title="Suara" variant="plain">
        <SettingRow
          description="Hasil hanya dibaca setelah pengguna menekan Bacakan."
          label="Mode membaca"
          testID="speech-mode"
          value="Tekan untuk membaca"
        />
        <SettingRow
          description={
            voiceAvailability
              ? voiceAvailability.status === 'available' &&
                voiceAvailability.localVoiceAvailable
                ? 'Suara lokal siap digunakan.'
                : 'Periksa atau pasang suara lokal melalui pengaturan Android.'
              : 'Memeriksa ketersediaan suara lokal.'
          }
          label="Suara lokal"
          value={
            voiceAvailability?.status === 'available' &&
            voiceAvailability.localVoiceAvailable
              ? 'Tersedia secara lokal'
              : voiceAvailability
                ? 'Belum tersedia'
                : 'Memeriksa'
          }
        >
          <AccessibleButton
            accessibilityHint="Membuka pengaturan pembaca teks Android"
            accessibilityLabel="Buka pengaturan suara Android"
            label="Pengaturan suara Android"
            onPress={() => void openTtsSettings()}
            variant="secondary"
          />
        </SettingRow>
        <SettingRow
          description="Rentang 0,5 sampai 2 kali."
          label="Kecepatan suara"
          value={`${preferences.speechRate.toFixed(1).replace('.', ',')} kali`}
        >
          <View style={styles.adjustableGroup}>
            <Slider
              accessibilityLabel="Kecepatan suara"
              accessibilityRole="adjustable"
              accessibilityValue={{
                min: RATE_MIN,
                max: RATE_MAX,
                now: preferences.speechRate,
                text: `${preferences.speechRate.toFixed(1)} kali`,
              }}
              maximumTrackTintColor={colors.outline}
              maximumValue={RATE_MAX}
              minimumTrackTintColor={colors.primary}
              minimumValue={RATE_MIN}
              onSlidingComplete={changeRate}
              step={RATE_STEP}
              style={styles.slider}
              thumbTintColor={colors.primary}
              value={preferences.speechRate}
            />
            <View style={styles.stepButtons}>
              <AccessibleButton
                accessibilityLabel="Kurangi kecepatan"
                accessibilityHint="Mengurangi kecepatan sebesar 0,1 kali"
                disabled={preferences.speechRate <= RATE_MIN}
                label="Kurangi"
                onPress={() => changeRate(preferences.speechRate - RATE_STEP)}
                style={styles.stepButton}
                variant="secondary"
              />
              <AccessibleButton
                accessibilityLabel="Tambah kecepatan"
                accessibilityHint="Menambah kecepatan sebesar 0,1 kali"
                disabled={preferences.speechRate >= RATE_MAX}
                label="Tambah"
                onPress={() => changeRate(preferences.speechRate + RATE_STEP)}
                style={styles.stepButton}
                variant="secondary"
              />
            </View>
          </View>
        </SettingRow>
        <SettingRow
          description="Rentang 0 sampai 100 persen."
          label="Volume suara"
          value={`${Math.round(preferences.speechVolume * 100)} persen`}
        >
          <View style={styles.adjustableGroup}>
            <Slider
              accessibilityLabel="Volume suara"
              accessibilityRole="adjustable"
              accessibilityValue={{
                min: 0,
                max: 100,
                now: Math.round(preferences.speechVolume * 100),
                text: `${Math.round(preferences.speechVolume * 100)} persen`,
              }}
              maximumTrackTintColor={colors.outline}
              maximumValue={VOLUME_MAX}
              minimumTrackTintColor={colors.primary}
              minimumValue={VOLUME_MIN}
              onSlidingComplete={changeVolume}
              step={VOLUME_STEP}
              style={styles.slider}
              thumbTintColor={colors.primary}
              value={preferences.speechVolume}
            />
            <View style={styles.stepButtons}>
              <AccessibleButton
                accessibilityLabel="Kurangi volume"
                accessibilityHint="Mengurangi volume sebesar 10 persen"
                disabled={preferences.speechVolume <= VOLUME_MIN}
                label="Kurangi"
                onPress={() =>
                  changeVolume(preferences.speechVolume - VOLUME_STEP)
                }
                style={styles.stepButton}
                variant="secondary"
              />
              <AccessibleButton
                accessibilityLabel="Tambah volume"
                accessibilityHint="Menambah volume sebesar 10 persen"
                disabled={preferences.speechVolume >= VOLUME_MAX}
                label="Tambah"
                onPress={() =>
                  changeVolume(preferences.speechVolume + VOLUME_STEP)
                }
                style={styles.stepButton}
                variant="secondary"
              />
            </View>
          </View>
        </SettingRow>
        <SettingRow
          description="Mematikan suara aplikasi tanpa mengubah getaran."
          label="Bisukan suara"
          value={preferences.muted ? 'Aktif' : 'Tidak aktif'}
        >
          <Switch
            accessibilityLabel="Bisukan suara"
            accessibilityRole="switch"
            accessibilityState={{ checked: preferences.muted }}
            hitSlop={8}
            onValueChange={(muted) => {
              updateAppPreferences({ muted });
            }}
            style={styles.switch}
            testID="mute-switch"
            thumbColor={preferences.muted ? colors.primary : colors.surfaceContainerLowest}
            trackColor={{
              false: colors.outline,
              true: colors.primary,
            }}
            value={preferences.muted}
          />
        </SettingRow>
      </SectionCard>

      <SectionCard title="Getaran" variant="plain">
        <SettingRow
          description="Dapat dimatikan tanpa menghilangkan informasi visual atau teks."
          label="Umpan balik getaran"
          value={preferences.hapticsEnabled ? 'Aktif' : 'Tidak aktif'}
        >
          <Switch
            accessibilityLabel="Umpan balik getaran"
            accessibilityRole="switch"
            accessibilityState={{ checked: preferences.hapticsEnabled }}
            hitSlop={8}
            onValueChange={(hapticsEnabled) => {
              updateAppPreferences({ hapticsEnabled });
            }}
            style={styles.switch}
            testID="haptics-switch"
            thumbColor={
              preferences.hapticsEnabled
                ? colors.primary
                : colors.surfaceContainerLowest
            }
            trackColor={{
              false: colors.outline,
              true: colors.primary,
            }}
            value={preferences.hapticsEnabled}
          />
        </SettingRow>
      </SectionCard>

      <SectionCard title="Aksesibilitas" variant="plain">
        <View
          accessibilityLabel="Navigasi dengan TalkBack. Gunakan empat tab di bagian bawah dengan urutan Beranda, Pindai, Penerjemah, lalu Setelan."
          accessible
          style={styles.accessibilityNote}
        >
          <Text accessible={false} style={styles.accessibilityTitle}>
            Navigasi dengan TalkBack
          </Text>
          <Text accessible={false} style={styles.accessibilityCopy}>
            Gunakan empat tab di bagian bawah untuk berpindah halaman. Urutan
            fokusnya adalah Beranda, Pindai, Penerjemah, lalu Setelan.
          </Text>
        </View>
      </SectionCard>

      <SectionCard title="Bantuan dan privasi" variant="plain">
        <View style={styles.helpActions}>
          <AccessibleButton
            accessibilityHint="Membuka panduan pemindaian dan batas penggunaan"
            label="Buka bantuan"
            onPress={() => router.push('/help')}
            variant="navigation"
          />
          <AccessibleButton
            accessibilityHint="Membuka kebijakan privasi BrailleMe"
            label="Buka privasi"
            onPress={() => router.push('/privacy')}
            variant="navigation"
          />
        </View>
      </SectionCard>

      {actionStatus ? (
        <Text
          accessibilityLiveRegion="polite"
          selectable
          style={styles.status}
        >
          {actionStatus}
        </Text>
      ) : null}

      <AccessibleButton
        accessibilityHint="Mengembalikan pengenalan, suara, dan getaran ke nilai awal"
        label="Kembalikan pengaturan awal"
        onPress={requestResetPreferences}
        variant="secondary"
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: {
    ...typography.displayLargeMobile,
    color: colors.text,
  },
  intro: {
    ...typography.bodyLarge,
    color: colors.textSecondary,
  },
  adjustableGroup: {
    alignSelf: 'stretch',
    gap: spacing.sm,
    minWidth: minimumTouchTarget,
  },
  slider: {
    alignSelf: 'stretch',
    height: minimumTouchTarget,
    minWidth: minimumTouchTarget,
  },
  switch: {
    alignSelf: 'flex-start',
    minHeight: minimumTouchTarget,
    minWidth: 56,
  },
  stepButtons: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  stepButton: {
    flex: 1,
    minWidth: 0,
    paddingHorizontal: spacing.md,
  },
  accessibilityNote: {
    gap: spacing.xs,
    backgroundColor: colors.primaryContainer,
    borderCurve: 'continuous',
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  accessibilityTitle: {
    ...typography.label,
    color: colors.onPrimaryContainer,
  },
  accessibilityCopy: {
    ...typography.body,
    color: colors.onPrimaryContainer,
  },
  helpActions: {
    gap: spacing.sm,
  },
  status: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surfaceContainerLow,
    borderLeftColor: colors.primary,
    borderLeftWidth: 4,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
});
