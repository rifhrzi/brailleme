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

  return {
    Redirect,
    Stack,
    useFocusEffect(effect: () => void | (() => void)) {
      React.useEffect(effect, [effect]);
    },
    useLocalSearchParams: () => localSearchParams,
    useRouter: () => router,
    __router: router,
    __resetRouterMock: () => {
      router.back.mockReset();
      router.canGoBack.mockReset().mockReturnValue(true);
      router.push.mockReset();
      router.replace.mockReset();
      localSearchParams = {};
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
