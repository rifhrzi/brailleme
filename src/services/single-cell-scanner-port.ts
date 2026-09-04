import type {
  LiveScanListener,
  ScannerCapabilities,
  ScannerCapabilitiesListener,
  ScannerPort,
} from '@/services/ports';
import {
  isValidLiveScanEvent,
  type LiveScanEvent,
  type PortResult,
  type ScannerAvailability,
  type ScannerStartResult,
} from '@/types';

export type SingleCellScanPayload = Omit<
  LiveScanEvent,
  'schemaVersion' | 'sessionId'
>;

type ScannerState = 'idle' | 'running' | 'paused';

const AVAILABLE: ScannerAvailability = Object.freeze({ available: true });

function failed(message: string): PortResult {
  return { ok: false, code: 'failed', message };
}

/**
 * Pengendali sesi untuk pipeline kamera satu-sel.
 *
 * Frame dan tensor tetap berada di komponen native/worklet. Port ini hanya
 * meneruskan DTO kecil yang sudah melalui quality gate ke layar Live.
 */
export class SingleCellScannerPort implements ScannerPort {
  private readonly listeners = new Set<LiveScanListener>();
  private readonly capabilityListeners = new Set<ScannerCapabilitiesListener>();
  private state: ScannerState = 'idle';
  private sessionId = '';
  private sessionSequence = 0;
  private torchAvailable = false;

  async getAvailability(): Promise<ScannerAvailability> {
    return AVAILABLE;
  }

  subscribe(listener: LiveScanListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  getCapabilities(): ScannerCapabilities {
    return { torchAvailable: this.torchAvailable };
  }

  subscribeCapabilities(listener: ScannerCapabilitiesListener): () => void {
    this.capabilityListeners.add(listener);
    return () => {
      this.capabilityListeners.delete(listener);
    };
  }

  /** Dipanggil surface kamera ketika perangkat aktif sudah diketahui. */
  setTorchAvailable(available: boolean): void {
    if (this.torchAvailable === available) {
      return;
    }
    this.torchAvailable = available;
    const capabilities = this.getCapabilities();
    for (const listener of this.capabilityListeners) {
      try {
        listener(capabilities);
      } catch {
        // Consumer capability tidak boleh memutus lifecycle scanner.
      }
    }
  }

  async start(): Promise<ScannerStartResult> {
    if (this.state !== 'idle' && this.sessionId) {
      return { ok: true, sessionId: this.sessionId };
    }

    this.sessionSequence += 1;
    this.sessionId = `single-cell-${Date.now()}-${this.sessionSequence}`;
    this.state = 'running';
    return { ok: true, sessionId: this.sessionId };
  }

  async pause(): Promise<PortResult> {
    if (this.state !== 'running') {
      return failed('Pemindai hanya dapat dijeda ketika sedang berjalan.');
    }
    this.state = 'paused';
    return { ok: true };
  }

  async resume(): Promise<PortResult> {
    if (this.state !== 'paused') {
      return failed('Pemindai hanya dapat dilanjutkan setelah dijeda.');
    }
    this.state = 'running';
    return { ok: true };
  }

  async stop(): Promise<PortResult> {
    this.state = 'idle';
    this.sessionId = '';
    this.setTorchAvailable(false);
    return { ok: true };
  }

  async setTorch(_enabled: boolean): Promise<PortResult> {
    if (this.state === 'idle') {
      return failed('Senter hanya dapat diubah saat pemindai aktif.');
    }
    if (!this.torchAvailable) {
      return failed('Senter tidak tersedia pada kamera aktif.');
    }
    return { ok: true };
  }

  /** Menerbitkan event hanya untuk sesi yang masih aktif dan tidak dijeda. */
  publish(expectedSessionId: string, payload: SingleCellScanPayload): boolean {
    if (
      this.state !== 'running' ||
      !this.sessionId ||
      expectedSessionId !== this.sessionId
    ) {
      return false;
    }

    const event: LiveScanEvent = {
      ...payload,
      schemaVersion: 2,
      sessionId: expectedSessionId,
    };
    // Pertahankan validasi di batas port selain validasi defensif milik UI.
    if (!isValidLiveScanEvent(event)) {
      return false;
    }
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch {
        // Satu consumer yang bermasalah tidak boleh memutus consumer lain.
      }
    }
    return true;
  }
}

export const scannerPort = new SingleCellScannerPort();
