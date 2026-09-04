import { IS_PRODUCTION_BUILD } from '@/config';
import { BRAILLE_MODEL_RELEASE_VERIFIED } from '@/ml/braille-model-contract';
import { isBrailleModelRuntimeAllowed } from '@/ml/braille-model-runtime-policy';
import type { ScanReadinessPort, ScannerPort } from '@/services/ports';
import { scannerPort } from '@/services/single-cell-scanner-port';
import type { ScanReadiness } from '@/types';

const READY: ScanReadiness = Object.freeze({ available: true });
const MODEL_NOT_AVAILABLE: ScanReadiness = Object.freeze({
  available: false,
  reason: 'model_not_available',
  message: 'Pemindai belum tersedia pada build ini.',
});
const SCANNER_NOT_AVAILABLE: ScanReadiness = Object.freeze({
  available: false,
  reason: 'scanner_not_available',
  message: 'Komponen pemindai belum tersedia pada perangkat ini.',
});

type ScannerAvailabilityProbe = Pick<ScannerPort, 'getAvailability'>;

export class RuntimeScanReadinessPort implements ScanReadinessPort {
  constructor(
    private readonly scanner: ScannerAvailabilityProbe = scannerPort,
    private readonly isProductionBuild = IS_PRODUCTION_BUILD,
    private readonly releaseVerified = BRAILLE_MODEL_RELEASE_VERIFIED,
  ) {}

  async getAvailability(): Promise<ScanReadiness> {
    if (
      !isBrailleModelRuntimeAllowed(
        this.isProductionBuild,
        this.releaseVerified,
      )
    ) {
      return MODEL_NOT_AVAILABLE;
    }

    try {
      const scanner = await this.scanner.getAvailability();
      return scanner.available ? READY : SCANNER_NOT_AVAILABLE;
    } catch {
      return SCANNER_NOT_AVAILABLE;
    }
  }
}

export const scanReadinessPort: ScanReadinessPort =
  new RuntimeScanReadinessPort();
