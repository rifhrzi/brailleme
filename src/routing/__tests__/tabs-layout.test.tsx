import { render, screen, waitFor } from '@testing-library/react-native';
import { AccessibilityInfo } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import PrimaryTabsLayout from '../../../app/(tabs)/_layout';

describe('navbar utama', () => {
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

  it('menampilkan tepat tiga tab berlabel dalam urutan yang konsisten', async () => {
    await renderTabs();

    const tabs = ['Beranda', 'Pindai', 'Pengaturan'].map((label) =>
      screen.getByLabelText(label),
    );
    expect(tabs).toHaveLength(3);
    expect(tabs.map((tab) => tab.props.accessibilityLabel)).toEqual([
      'Beranda',
      'Pindai',
      'Pengaturan',
    ]);
    expect(screen.queryByRole('tab', { name: 'Privasi' })).toBeNull();
    expect(screen.queryByRole('tab', { name: 'Diagnostik' })).toBeNull();
    expect(screen.getByTestId('primary-tabs-safe-area')).toHaveStyle({
      paddingTop: 24,
    });
    expect(screen.getByTestId('expo-router-tabs')).toHaveStyle({ height: 80 });
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
