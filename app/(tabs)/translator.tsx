import { Redirect } from 'expo-router';

import { PrimaryPage } from '@/components';
import { TranslatorScreen } from '@/features/translator/translator-screen';
import { useAppPreferences } from '@/storage';

export default function TranslatorRoute() {
  const preferences = useAppPreferences();

  if (!preferences.onboardingCompleted) {
    return <Redirect href="/onboarding" />;
  }

  return (
    <PrimaryPage page="translator">
      <TranslatorScreen />
    </PrimaryPage>
  );
}
