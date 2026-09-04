import { useRouter } from 'expo-router';
import { StyleSheet, Text } from 'react-native';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { colors, typography } from '@/theme';

export default function HelpScreen() {
  const router = useRouter();

  return (
    <Screen testID="help-screen">
      <Text selectable style={styles.intro}>
        Ikuti langkah berikut untuk menyusun satu kata dan memulihkan pemindai
        jika ada masalah.
      </Text>

      <SectionCard title="Susun kata, satu sel per huruf" variant="plain">
        <Text selectable style={styles.body}>
          Buka Pindai, izinkan kamera, lalu tempatkan sel Braille pertama di
          tengah kotak. Tahan posisi sampai huruf stabil, periksa hasilnya, dan
          pilih Tambah huruf. Ulangi untuk sel berikutnya sampai satu kata
          selesai.
        </Text>
        <Text selectable style={styles.body}>
          Gunakan Hapus huruf terakhir untuk koreksi. Pemindai menyusun kata
          secara berurutan; aplikasi tidak membaca seluruh baris sekaligus dan
          tidak menebak spasi.
        </Text>
      </SectionCard>

      <SectionCard title="Periksa sebelum digunakan" variant="plain">
        <Text selectable style={styles.body}>
          Huruf yang stabil dan kata yang tersusun tetap dapat salah. Jangan
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
