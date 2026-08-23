import { Redirect } from 'expo-router';

import DebugScreen from '@/features/debug/debug-screen';

export default function DebugRoute() {
  if (!__DEV__) {
    return <Redirect href="/settings" />;
  }

  return <DebugScreen />;
}
