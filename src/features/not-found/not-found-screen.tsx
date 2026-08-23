import { useRouter } from 'expo-router';
import { StyleSheet, Text } from 'react-native';

import { AccessibleButton, Screen } from '@/components';
import { colors, typography } from '@/theme';

export default function NotFoundScreen() {
  const router = useRouter();

  return (
    <Screen testID="not-found-screen">
      <Text selectable style={styles.body}>
        Halaman yang diminta tidak tersedia di BrailleMe.
      </Text>
      <AccessibleButton label="Kembali ke Beranda" onPress={() => router.replace('/')} />
    </Screen>
  );
}

const styles = StyleSheet.create({
  body: {
    ...typography.bodyLarge,
    color: colors.text,
  },
});
