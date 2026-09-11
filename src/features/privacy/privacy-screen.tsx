import { StyleSheet, Text } from 'react-native';

import { Screen, SectionCard } from '@/components';
import { colors, typography } from '@/theme';

export default function PrivacyScreen() {
  return (
    <Screen testID="privacy-screen">
      <Text selectable style={styles.supporting}>
        Kebijakan Privasi BrailleMe, berlaku mulai 11 September 2026.
      </Text>

      <SectionCard title="Pemrosesan tetap di perangkat" variant="plain">
        <Text selectable style={styles.body}>
          BrailleMe membaca foto kata, angka, dan tanda baca Braille langsung di
          perangkat. Foto dan hasil hanya digunakan sementara selama
          pemindaian. Aplikasi tidak menyimpan atau mengunggah gambar kamera
          maupun teks hasil pemindaian.
        </Text>
      </SectionCard>

      <SectionCard title="Ketikan dan titik Braille" variant="plain">
        <Text selectable style={styles.body}>
          Penerjemah bekerja tanpa internet atau izin kamera. Ketikan dan
          susunan titik hanya disimpan sementara dalam memori selama aplikasi
          berjalan, termasuk saat berpindah tab. Isinya tidak disimpan sebagai
          riwayat dan tidak dikirim ke layanan lain. Gunakan Hapus semua untuk
          mengosongkan masukan pada mode yang sedang dibuka.
        </Text>
      </SectionCard>

      <SectionCard title="Data yang disimpan" variant="plain">
        <Text selectable style={styles.body}>
          Hanya pengaturan aplikasi, seperti kecepatan, volume suara, dan getaran,
          yang disimpan di perangkat. Pengaturan dapat dihapus melalui
          Kembalikan pengaturan awal di halaman
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
          Pemindaian dapat digunakan tanpa internet. Aplikasi tidak memiliki
          akun, iklan, atau pelacakan penggunaan, serta
          tidak menjual maupun membagikan data pengguna.
        </Text>
      </SectionCard>

      <SectionCard title="Perubahan kebijakan" variant="plain">
        <Text selectable style={styles.body}>
          Jika cara data diproses berubah, penjelasan pada halaman ini akan
          diperbarui. Anda dapat membacanya kembali melalui Pengaturan, lalu
          pilih Buka privasi.
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
