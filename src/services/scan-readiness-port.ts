import { IS_PRODUCTION_BUILD } from '@/config';
import { WORD_MODEL_BUNDLED, WORD_MODEL_RELEASE_VERIFIED } from '@/ml/word-model-assets';
import { isBrailleModelRuntimeAllowed } from '@/ml/braille-model-runtime-policy';
import type { ScanReadinessPort, ScannerPort } from '@/services/ports';
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
// Keep the home route lightweight: actual native/model loading happens on Pindai.
const PHOTO_SCANNER_PROBE: ScannerAvailabilityProbe = {
  getAvailability: async () => ({ available: true }),
};

export class RuntimeScanReadinessPort implements ScanReadinessPort {
  constructor(
    private readonly scanner: ScannerAvailabilityProbe = PHOTO_SCANNER_PROBE,
    private readonly isProductionBuild = IS_PRODUCTION_BUILD,
    private readonly releaseVerified = WORD_MODEL_RELEASE_VERIFIED,
    private readonly modelBundled = WORD_MODEL_BUNDLED,
  ) {}

  async getAvailability(): Promise<ScanReadiness> {
    if (
      !this.modelBundled || !isBrailleModelRuntimeAllowed(
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
