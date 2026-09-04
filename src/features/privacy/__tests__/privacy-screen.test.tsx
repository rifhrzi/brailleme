import { render, screen } from '@testing-library/react-native';

import PrivacyScreen from '../privacy-screen';

describe('PrivacyScreen', () => {
  it('menjelaskan pemrosesan lokal dan penghapusan preferensi', async () => {
    await render(<PrivacyScreen />);

    expect(screen.getByText(/Kebijakan Privasi BrailleMe/i)).toBeOnTheScreen();
    expect(screen.getByText(/tidak menyimpan atau mengunggah gambar/i)).toBeOnTheScreen();
    expect(
      screen.getByText(/Kembalikan pengaturan awal di halaman Pengaturan/i),
    ).toBeOnTheScreen();
    expect(screen.getByText(/tidak meminta izin internet/i)).toBeOnTheScreen();
  });
});
