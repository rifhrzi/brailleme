import { render, screen } from '@testing-library/react-native';
import { Text } from 'react-native';

import { PrimaryPage } from '@/components/primary-page';
import { colors } from '@/theme';

describe('PrimaryPage', () => {
  it('hanya membungkus konten tanpa navigasi atau fokus tambahan', async () => {
    await render(
      <PrimaryPage page="scan">
        <Text>Konten pindai</Text>
      </PrimaryPage>,
    );

    expect(screen.getByTestId('primary-page-scan')).toHaveStyle({
      flex: 1,
      backgroundColor: colors.canvas,
    });
    expect(screen.getByText('Konten pindai')).toBeOnTheScreen();
    expect(screen.queryByRole('adjustable')).toBeNull();
    expect(screen.queryByLabelText('Navigasi halaman')).toBeNull();
  });
});
