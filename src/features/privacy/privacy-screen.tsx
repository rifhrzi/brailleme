import { StyleSheet, Text } from 'react-native';

import { Screen, SectionCard } from '@/components';
import { colors, typography } from '@/theme';

export default function PrivacyScreen() {
  return (
    <Screen testID="privacy-screen">
      <Text accessibilityRole="header" selectable style={styles.title}>
        Kebijakan Privasi BrailleMe
      </Text>
      <Text selectable style={styles.supporting}>
        Berlaku mulai 22 Agustus 2026
      </Text>

      <SectionCard title="Pemrosesan tetap di perangkat">
        <Text selectable style={styles.body}>
          BrailleMe memproses frame kamera, crop satu sel, dan kandidat huruf
          langsung di perangkat. Aplikasi tidak menyimpan atau mengunggah gambar
          kamera maupun teks hasil pemindaian.
        </Text>
      </SectionCard>

      <SectionCard title="Data yang disimpan">
        <Text selectable style={styles.body}>
          Hanya preferensi aplikasi—seperti status pengenalan, kecepatan dan
          volume suara, mute, serta haptik—yang disimpan secara lokal. Preferensi
          dapat dihapus melalui Reset di Pengaturan atau dengan menghapus aplikasi.
        </Text>
      </SectionCard>

      <SectionCard title="Izin dan layanan perangkat">
        <Text selectable style={styles.body}>
          Izin kamera digunakan hanya ketika pemindai dibuka. Getaran dipakai
          untuk umpan balik haptik. Suara dibacakan melalui mesin text-to-speech
          Android yang terpasang; BrailleMe hanya memilih voice Bahasa Indonesia
          yang ditandai tersedia secara lokal.
        </Text>
      </SectionCard>

      <SectionCard title="Jaringan, akun, dan pelacakan">
        <Text selectable style={styles.body}>
          Build standalone BrailleMe tidak meminta izin internet. Aplikasi tidak
          memiliki akun, iklan, analytics, atau telemetry, serta tidak menjual
          maupun membagikan data pengguna.
        </Text>
      </SectionCard>

      <SectionCard title="Kontak dan perubahan">
        <Text selectable style={styles.body}>
          Kontak pengembang dan URL kebijakan publik dicantumkan pada halaman
          BrailleMe di Google Play. Perubahan material pada cara data diproses
          harus memperbarui kebijakan ini dan deklarasi Data safety sebelum rilis.
        </Text>
      </SectionCard>
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: {
    ...typography.headlineMedium,
    color: colors.text,
  },
  supporting: {
    ...typography.supporting,
    color: colors.textSecondary,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
});
