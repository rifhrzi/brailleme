import { useRouter } from 'expo-router';
import { StyleSheet, Text } from 'react-native';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { colors, typography } from '@/theme';

export default function HelpScreen() {
  const router = useRouter();

  return (
    <Screen testID="help-screen">
      <Text selectable style={styles.intro}>
        Ikuti langkah berikut untuk memindai satu kata dan memulihkan pemindai
        jika ada masalah.
      </Text>

      <SectionCard title="Pindai satu blok kata" variant="plain">
        <Text selectable style={styles.body}>
          Buka Pindai, izinkan kamera, lalu tempatkan seluruh kata di dalam
          bingkai. Sisakan ruang di kedua ujung kata dan tahan kamera saat
          menekan Pindai kata. Aplikasi membaca sel dari kiri ke kanan.
        </Text>
        <Text selectable style={styles.body}>
          Posisikan kamera tepat di atas kata. Arahkan cahaya miring dari ujung
          atas kertas agar bayangan titik terlihat.
        </Text>
        <Text selectable style={styles.body}>
          Periksa hasil, lalu pilih Bacakan. Jika belum sesuai atau gambar kurang
          jelas, pilih Pindai ulang. Pemindai menerima satu kata A–Z dalam satu
          baris, maksimal 12 huruf. Angka, tanda baca, singkatan Braille, dan
          beberapa kata sekaligus belum didukung.
        </Text>
      </SectionCard>

      <SectionCard title="Periksa sebelum digunakan" variant="plain">
        <Text selectable style={styles.body}>
          Kata hasil pemindaian tetap dapat salah. Jangan
          gunakan hasil tanpa verifikasi manusia untuk konteks medis, hukum,
          finansial, ujian, keselamatan, atau keputusan penting lainnya.
        </Text>
        <Text selectable style={styles.body}>
          Periksa hasil bersama orang yang memahami sumbernya atau pemeriksa
          Braille yang tepercaya.
        </Text>
      </SectionCard>

      <SectionCard title="Jika kamera atau suara bermasalah" variant="plain">
        <Text selectable style={styles.body}>
          Periksa izin kamera di pengaturan aplikasi. Untuk suara, buka
          Pengaturan lalu pilih Pengaturan suara Android dan pastikan suara
          Bahasa Indonesia tersedia secara lokal.
        </Text>
      </SectionCard>

      <Text selectable style={styles.privacy}>
        Privasi: gambar, huruf, dan kata hasil pemindaian tidak disimpan; hanya
        preferensi aplikasi yang tersimpan pada perangkat.
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
