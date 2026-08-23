import { useRouter } from 'expo-router';
import { StyleSheet, Text } from 'react-native';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { colors, typography } from '@/theme';

export default function HelpScreen() {
  const router = useRouter();

  return (
    <Screen testID="help-screen">
      <Text selectable style={styles.intro}>
        Temukan status fitur, cara memeriksa hasil, dan langkah jika ada masalah.
      </Text>

      <SectionCard title="Mulai dari Pindai Satu Sel">
        <Text selectable style={styles.body}>
          Buka Pindai Satu Sel, izinkan kamera, lalu tempatkan tepat satu huruf
          Braille timbul di tengah kotak. Tahan posisi sampai kandidat stabil.
          Model ini belum dapat membaca kata, spasi, atau satu baris.
        </Text>
      </SectionCard>

      <SectionCard title="Risiko penggunaan">
        <Text selectable style={styles.body}>
          Kandidat stabil tetap dapat salah. Jangan gunakan kandidat tanpa
          verifikasi manusia untuk konteks medis, hukum, finansial, ujian,
          keselamatan, atau keputusan penting lainnya.
        </Text>
        <Text selectable style={styles.body}>
          Periksa hasil bersama orang yang memahami sumbernya atau pemeriksa
          Braille yang tepercaya.
        </Text>
      </SectionCard>

      <SectionCard title="Jika kamera atau suara bermasalah">
        <Text selectable style={styles.body}>
          Periksa izin kamera di pengaturan aplikasi. Untuk suara, buka
          Pengaturan lalu pilih Pengaturan suara Android dan pastikan suara
          Bahasa Indonesia tersedia secara lokal.
        </Text>
      </SectionCard>

      <Text selectable style={styles.privacy}>
        Privasi: gambar dan teks hasil tidak disimpan; hanya preferensi yang
        tersimpan pada perangkat.
      </Text>

      <AccessibleButton
        accessibilityHint="Membuka kebijakan privasi lengkap di dalam aplikasi"
        label="Baca kebijakan privasi"
        onPress={() => router.push('/privacy')}
        variant="secondary"
      />

      <AccessibleButton
        accessibilityHint="Membuka kembali penjelasan awal aplikasi"
        label="Buka kembali pengenalan"
        onPress={() => router.push({ pathname: '/onboarding', params: { review: '1' } })}
        variant="secondary"
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  intro: {
    ...typography.bodyLarge,
    color: colors.text,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
  privacy: {
    ...typography.supporting,
    color: colors.textSecondary,
  },
});
