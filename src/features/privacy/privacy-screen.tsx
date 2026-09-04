import { StyleSheet, Text } from 'react-native';

import { Screen, SectionCard } from '@/components';
import { colors, typography } from '@/theme';

export default function PrivacyScreen() {
  return (
    <Screen testID="privacy-screen">
      <Text selectable style={styles.supporting}>
        Kebijakan Privasi BrailleMe, berlaku mulai 27 Agustus 2026.
      </Text>

      <SectionCard title="Pemrosesan tetap di perangkat" variant="plain">
        <Text selectable style={styles.body}>
          BrailleMe memproses gambar kamera, potongan satu sel, hasil huruf, dan
          kata sementara langsung di perangkat. Aplikasi tidak menyimpan atau
          mengunggah gambar kamera maupun teks hasil pemindaian.
        </Text>
      </SectionCard>

      <SectionCard title="Data yang disimpan" variant="plain">
        <Text selectable style={styles.body}>
          Hanya preferensi aplikasi, seperti status pengenalan, kecepatan dan
          volume suara, status bisu, serta getaran, yang disimpan secara lokal.
          Preferensi dapat dihapus melalui Kembalikan pengaturan awal di halaman
          Pengaturan atau dengan menghapus aplikasi.
        </Text>
      </SectionCard>

      <SectionCard title="Izin dan layanan perangkat" variant="plain">
        <Text selectable style={styles.body}>
          Izin kamera digunakan hanya ketika pemindai dibuka. Getaran dipakai
          sebagai umpan balik. Suara dibacakan melalui mesin pembaca teks Android
          yang terpasang; BrailleMe hanya memilih suara Bahasa Indonesia yang
          tersedia secara lokal.
        </Text>
      </SectionCard>

      <SectionCard title="Jaringan, akun, dan pelacakan" variant="plain">
        <Text selectable style={styles.body}>
          BrailleMe yang dipasang dari paket rilis tidak meminta izin internet.
          Aplikasi tidak memiliki akun, iklan, analitik, atau telemetri, serta
          tidak menjual maupun membagikan data pengguna.
        </Text>
      </SectionCard>

      <SectionCard title="Kontak dan perubahan" variant="plain">
        <Text selectable style={styles.body}>
          Kontak pengembang dan URL kebijakan publik dicantumkan pada halaman
          BrailleMe di Google Play. Jika cara data diproses berubah, kebijakan ini
          dan deklarasi Data safety akan diperbarui.
        </Text>
      </SectionCard>
    </Screen>
  );
}

const styles = StyleSheet.create({
  supporting: {
    ...typography.supporting,
    color: colors.textSecondary,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
});
