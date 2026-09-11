import { useRouter } from 'expo-router';
import { StyleSheet, Text } from 'react-native';

import { AccessibleButton, Screen, SectionCard } from '@/components';
import { colors, typography } from '@/theme';

export default function HelpScreen() {
  const router = useRouter();

  return (
    <Screen testID="help-screen">
      <Text selectable style={styles.intro}>
        Ikuti panduan ini untuk memindai dan menerjemahkan Braille.
      </Text>

      <SectionCard title="Bingkai mengikuti Braille" variant="plain">
        <Text selectable style={styles.body}>
          Buka Pindai dan izinkan kamera. Arahkan satu kata atau rangkaian angka
          ke tengah kamera. Bingkai mengikuti posisinya dan kotak kecil menandai
          setiap sel. Sisakan ruang di kedua ujung, tunggu petunjuk untuk menahan
          kamera, lalu tekan Pindai.
        </Text>
        <Text selectable style={styles.body}>
          Posisikan kamera tepat di atas kata. Arahkan cahaya miring dari ujung
          atas kertas agar bayangan titik terlihat.
        </Text>
        <Text selectable style={styles.body}>
          Periksa hasil, lalu pilih Bacakan. Jika belum sesuai atau gambar kurang
          jelas, pilih Pindai ulang. Gunakan satu kata atau angka bulat dalam satu
          baris, maksimal 12 karakter. Sertakan tanda kapital, angka, dan tanda baca.
          Huruf tanpa tanda kapital dibaca kecil; nol awal angka tetap ditampilkan.
        </Text>
      </SectionCard>

      <SectionCard title="Kata, angka, dan tanda baca" variant="plain">
        <Text selectable style={styles.body}>
          Pindai huruf kecil, huruf kapital, atau angka bulat. Tanda baca yang
          dapat dibaca: titik, koma, tanda tanya, tanda seru, titik dua, titik
          koma, tanda hubung, dan tanda kurung.
        </Text>
        <Text selectable style={styles.body}>
          Sertakan kata bersama tanda bacanya, misalnya Baca?, ABC!, atau (001).
          Jika memakai tanda kurung, sertakan kurung buka dan tutup. Pindai satu
          rangkaian tanpa spasi; desimal, angka negatif, singkatan Braille, dan
          gabungan huruf dengan angka belum dapat dibaca.
        </Text>
      </SectionCard>

      <SectionCard title="Penerjemah dua arah" variant="plain">
        <Text selectable style={styles.body}>
          Buka Penerjemah, lalu pilih Teks ke Braille untuk melihat titik Braille
          dari ketikan Anda. Spasi dan baris baru tetap ditampilkan. Gunakan huruf,
          angka bulat yang terpisah dari huruf, dan tanda baca umum.
        </Text>
        <Text selectable style={styles.body}>
          Pilih Braille ke Teks untuk menyusun titik sendiri. Tambah sel, pilih
          selnya, lalu tekan tombol titik 1 sampai 6. Titik 1, 2, 3 ada di kiri;
          titik 4, 5, 6 ada di kanan. Tombol Spasi dan Baris baru memisahkan teks.
          Hasil berubah langsung. Lengkapi bagian yang ditandai sebelum
          memindahkan hasil ke mode lain.
        </Text>
        <Text selectable style={styles.body}>
          Batasnya 200 karakter teks dan 400 sel Braille. Ketikan dan susunan
          titik tetap ada saat berpindah tab selama aplikasi berjalan.
        </Text>
      </SectionCard>

      <SectionCard title="Periksa sebelum digunakan" variant="plain">
        <Text selectable style={styles.body}>
          Cocokkan hasil dengan sumber Braille, terutama huruf kapital, angka,
          dan tanda baca. Untuk informasi penting, periksa bersama orang yang
          memahami Braille. Pilih Pindai ulang jika hasil belum sesuai.
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
