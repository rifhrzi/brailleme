import { HankenGrotesk_600SemiBold } from '@expo-google-fonts/hanken-grotesk/600SemiBold';
import { HankenGrotesk_700Bold } from '@expo-google-fonts/hanken-grotesk/700Bold';
import { Inter_400Regular } from '@expo-google-fonts/inter/400Regular';
import { Inter_500Medium } from '@expo-google-fonts/inter/500Medium';
import { Inter_600SemiBold } from '@expo-google-fonts/inter/600SemiBold';
import { JetBrainsMono_500Medium } from '@expo-google-fonts/jetbrains-mono/500Medium';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { SymbolView } from 'expo-symbols';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { Pressable, StyleSheet } from 'react-native';

import { colors, fonts } from '@/theme';
import { useReduceMotionEnabled } from '@/accessibility/use-accessibility-preferences';

void SplashScreen.preventAutoHideAsync();

function LocalizedHeaderBackButton({ onPress }: { onPress: () => void }) {
  const [focused, setFocused] = useState(false);

  return (
    <Pressable
      accessibilityLabel="Kembali"
      accessibilityRole="button"
      android_ripple={{ color: colors.surfaceContainerHigh, borderless: true }}
      hitSlop={8}
      onBlur={() => setFocused(false)}
      onFocus={() => setFocused(true)}
      onPress={onPress}
      style={({ pressed }) => [
        styles.backButton,
        focused && styles.backButtonFocused,
        pressed && styles.backButtonPressed,
      ]}
    >
      <SymbolView
        accessible={false}
        name={{ android: 'arrow_back', ios: 'chevron.left' }}
        size={28}
        tintColor={colors.text}
      />
    </Pressable>
  );
}

export default function RootLayout() {
  const reduceMotionEnabled = useReduceMotionEnabled();
  const [fontsLoaded, fontError] = useFonts({
    HankenGrotesk_600SemiBold,
    HankenGrotesk_700Bold,
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    JetBrainsMono_500Medium,
  });

  useEffect(() => {
    if (fontsLoaded || fontError) {
      void SplashScreen.hideAsync();
    }
  }, [fontError, fontsLoaded]);

  if (!fontsLoaded && !fontError) {
    return null;
  }

  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={({ navigation }) => ({
          animation: reduceMotionEnabled ? 'none' : 'simple_push',
          contentStyle: { backgroundColor: colors.canvas },
          headerBackButtonDisplayMode: 'minimal',
          headerLeft: (props) =>
            props.canGoBack ? (
              <LocalizedHeaderBackButton onPress={navigation.goBack} />
            ) : null,
          headerShadowVisible: false,
          headerStyle: { backgroundColor: colors.canvas },
          headerTintColor: colors.text,
          headerTitleStyle: {
            fontFamily: fonts.headingBold,
            fontSize: 24,
            fontWeight: '700',
          },
        })}
      >
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="onboarding" options={{ title: 'Tentang BrailleMe' }} />
        <Stack.Screen name="help" options={{ title: 'Bantuan' }} />
        <Stack.Screen name="privacy" options={{ title: 'Privasi' }} />
        <Stack.Screen name="+not-found" options={{ title: 'Halaman tidak ditemukan' }} />
      </Stack>
    </>
  );
}

const styles = StyleSheet.create({
  backButton: {
    minHeight: 48,
    minWidth: 48,
    alignItems: 'center',
    justifyContent: 'center',
    borderColor: 'transparent',
    borderRadius: 24,
    borderWidth: 2,
  },
  backButtonFocused: {
    borderColor: colors.primary,
  },
  backButtonPressed: {
    backgroundColor: colors.surfaceContainerHigh,
  },
});
