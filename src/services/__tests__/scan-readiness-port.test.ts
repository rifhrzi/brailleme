import { RuntimeScanReadinessPort } from '@/services/scan-readiness-port';
import type { ScannerPort } from '@/services/ports';
import type { ScannerAvailability } from '@/types';

function scannerProbe(availability: ScannerAvailability) {
  return {
    getAvailability: jest.fn(async () => availability),
  } satisfies Pick<ScannerPort, 'getAvailability'>;
}

describe('RuntimeScanReadinessPort', () => {
  it('fail-closed pada production ketika model belum terverifikasi', async () => {
    const scanner = scannerProbe({ available: true });
    const port = new RuntimeScanReadinessPort(scanner, true, false);

    await expect(port.getAvailability()).resolves.toEqual({
      available: false,
      reason: 'model_not_available',
      message: 'Pemindai belum tersedia pada build ini.',
    });
    expect(scanner.getAvailability).not.toHaveBeenCalled();
  });

  it('tidak siap ketika runtime scanner tidak tersedia', async () => {
    const scanner = scannerProbe({
      available: false,
      reason: 'native_scanner_not_installed',
      message: 'Native scanner tidak ditemukan.',
    });
    const port = new RuntimeScanReadinessPort(scanner, true, true);

    await expect(port.getAvailability()).resolves.toEqual({
      available: false,
      reason: 'scanner_not_available',
      message: 'Komponen pemindai belum tersedia pada perangkat ini.',
    });
  });

  it.each([
    ['production terverifikasi', true, true],
    ['preview kandidat', false, false],
  ])('siap untuk %s ketika scanner tersedia', async (_, production, verified) => {
    const scanner = scannerProbe({ available: true });
    const port = new RuntimeScanReadinessPort(
      scanner,
      production,
      verified,
    );

    await expect(port.getAvailability()).resolves.toEqual({ available: true });
    expect(scanner.getAvailability).toHaveBeenCalledTimes(1);
  });
});
