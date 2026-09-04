const localStorageValues = new Map<string, string>();

const localStorageMock = {
  get length() {
    return localStorageValues.size;
  },
  clear: jest.fn(() => {
    localStorageValues.clear();
  }),
  getItem: jest.fn((key: string) => localStorageValues.get(key) ?? null),
  key: jest.fn((index: number) => [...localStorageValues.keys()][index] ?? null),
  removeItem: jest.fn((key: string) => {
    localStorageValues.delete(key);
  }),
  setItem: jest.fn((key: string, value: string) => {
    localStorageValues.set(key, String(value));
  }),
};

Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: localStorageMock,
});

jest.mock('expo-sqlite/localStorage/install', () => ({}));

jest.mock('expo-intent-launcher', () => ({
  ActivityAction: {
    TTS_SETTINGS: 'android.settings.TTS_SETTINGS',
  },
  startActivityAsync: jest.fn(async () => undefined),
}));

jest.mock('@react-native-community/slider', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  const { View } = jest.requireActual<typeof import('react-native')>(
    'react-native',
  );

  function MockSlider(props: Record<string, unknown>) {
    return React.createElement(View, { ...props, accessible: true });
  }

  return {
    __esModule: true,
    default: MockSlider,
  };
});

jest.mock('expo-router', () => {
  const React = jest.requireActual<typeof import('react')>('react');
  const { View } = jest.requireActual<typeof import('react-native')>(
    'react-native',
  );
  const router = {
    back: jest.fn(),
    canGoBack: jest.fn(() => true),
    push: jest.fn(),
    replace: jest.fn(),
  };
  let localSearchParams: Record<string, string | undefined> = {};
  let routeFocused = true;
  const focusRecords = new Set<{
    active: boolean;
    cleanup: (() => void) | undefined;
    effect: () => void | (() => void);
  }>();

  function Redirect({ href }: { href: unknown }) {
    return React.createElement(View, {
      accessibilityLabel: `Alihkan ke ${String(href)}`,
      accessibilityValue: { text: String(href) },
      testID: 'expo-router-redirect',
    });
  }

  function Stack({ children }: { children?: React.ReactNode }) {
    return React.createElement(React.Fragment, null, children);
  }

  Stack.Screen = function Screen() {
    return null;
  };

  function Tabs({
    children,
    screenOptions,
  }: {
    children?: React.ReactNode;
    screenOptions?: {
      animation?: string;
      tabBarStyle?: React.ComponentProps<typeof View>['style'];
    };
  }) {
    return React.createElement(
      View,
      {
        accessibilityValue: { text: screenOptions?.animation },
        style: screenOptions?.tabBarStyle,
        testID: 'expo-router-tabs',
      },
      children,
    );
  }

  Tabs.Screen = function Screen({
    name,
    options,
  }: {
    name: string;
    options?: {
      tabBarAccessibilityLabel?: string;
      tabBarButtonTestID?: string;
      title?: string;
    };
  }) {
    return React.createElement(View, {
      accessibilityLabel: options?.tabBarAccessibilityLabel ?? options?.title,
      accessibilityRole: 'tab',
      accessibilityValue: { text: name },
      testID: options?.tabBarButtonTestID,
    });
  };

  function useFocusEffect(effect: () => void | (() => void)) {
    React.useEffect(() => {
      const record = {
        active: false,
        cleanup: undefined as (() => void) | undefined,
        effect,
      };
      if (routeFocused) {
        record.cleanup = effect() || undefined;
        record.active = true;
      }
      focusRecords.add(record);
      return () => {
        if (record.active) {
          record.cleanup?.();
        }
        focusRecords.delete(record);
      };
    }, [effect]);
  }

  return {
    Redirect,
    Stack,
    Tabs,
    useFocusEffect,
    useLocalSearchParams: () => localSearchParams,
    useRouter: () => router,
    __router: router,
    __resetRouterMock: () => {
      router.back.mockReset();
      router.canGoBack.mockReset().mockReturnValue(true);
      router.push.mockReset();
      router.replace.mockReset();
      localSearchParams = {};
      routeFocused = true;
    },
    __setRouteFocused: (focused: boolean) => {
      routeFocused = focused;
      for (const record of focusRecords) {
        if (focused && !record.active) {
          record.cleanup = record.effect() || undefined;
          record.active = true;
        } else if (!focused && record.active) {
          record.cleanup?.();
          record.cleanup = undefined;
          record.active = false;
        }
      }
    },
    __setLocalSearchParams: (
      nextParams: Record<string, string | undefined>,
    ) => {
      localSearchParams = nextParams;
    },
  };
});

beforeEach(() => {
  localStorageMock.clear();

  const routerModule = jest.requireMock<{
    __resetRouterMock(): void;
  }>('expo-router');
  routerModule.__resetRouterMock();
});
