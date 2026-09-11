import { act, render, screen, waitFor } from '@testing-library/react-native';
import { AccessibilityInfo, Dimensions } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import PrimaryTabsLayout from '../../../app/(tabs)/_layout';

describe('navbar utama', () => {
  let originalWindow: ReturnType<typeof Dimensions.get>;

  beforeEach(async () => {
    originalWindow = Dimensions.get('window');
    await act(() => {
      Dimensions.set({ window: { width: 360, height: 800, scale: 1, fontScale: 1 } });
    });
  });

  afterEach(async () => {
    await act(() => { Dimensions.set({ window: originalWindow }); });
  });

  const renderTabs = () =>
    render(
      <SafeAreaProvider
        initialMetrics={{
          frame: { x: 0, y: 0, width: 360, height: 800 },
          insets: { top: 24, right: 0, bottom: 16, left: 0 },
        }}
      >
        <PrimaryTabsLayout />
      </SafeAreaProvider>,
    );

  it('menampilkan tepat empat tab berlabel dalam urutan yang konsisten', async () => {
    await renderTabs();

    const tabs = ['Beranda', 'Pindai', 'Penerjemah', 'Setelan'].map((label) =>
      screen.getByLabelText(label),
    );
    expect(screen.getAllByTestId(/^tab-/)).toHaveLength(4);
    expect(tabs.map((tab) => tab.props.accessibilityLabel)).toEqual([
      'Beranda',
      'Pindai',
      'Penerjemah',
      'Setelan',
    ]);
    expect(screen.queryByRole('tab', { name: 'Privasi' })).toBeNull();
    expect(screen.queryByRole('tab', { name: 'Diagnostik' })).toBeNull();
    expect(screen.getByTestId('primary-tabs-safe-area')).toHaveStyle({
      paddingTop: 24,
    });
    expect(screen.getByTestId('expo-router-tabs')).toHaveStyle({ height: 80 });
  });

  it('memberi ruang label empat tab pada layar kecil dengan font 200 persen', async () => {
    const previousWindow = Dimensions.get('window');
    Dimensions.set({ window: { width: 320, height: 640, scale: 1, fontScale: 2 } });
    try {
      await renderTabs();
      expect(screen.getAllByTestId(/^tab-/)).toHaveLength(4);
      expect(screen.getByLabelText('Penerjemah')).toHaveAccessibilityValue({ text: 'translator' });
      expect(screen.getByTestId('expo-router-tabs')).toHaveStyle({ height: 174 });
    } finally {
      await act(() => { Dimensions.set({ window: previousWindow }); });
    }
  });

  it('mematikan animasi perpindahan ketika reduced motion aktif', async () => {
    const reduceMotionSpy = jest
      .spyOn(AccessibilityInfo, 'isReduceMotionEnabled')
      .mockResolvedValueOnce(true);

    try {
      await renderTabs();
      await waitFor(() => {
        expect(screen.getByTestId('expo-router-tabs')).toHaveAccessibilityValue({
          text: 'none',
        });
      });
    } finally {
      reduceMotionSpy.mockRestore();
    }
  });
});
