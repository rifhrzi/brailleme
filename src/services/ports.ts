import type {
  CameraPermissionSnapshot,
  LiveScanEvent,
  PortResult,
  ScanReadiness,
  ScannerStartResult,
  ScannerAvailability,
  TtsAvailability,
  TtsSpeakOptions,
} from "@/types";

export type LiveScanListener = (event: LiveScanEvent) => void;

export type ScannerCapabilities = {
  readonly torchAvailable: boolean;
};

export type ScannerCapabilitiesListener = (
  capabilities: ScannerCapabilities,
) => void;

export type TtsPlaybackEvent =
  | { readonly status: 'speaking'; readonly text: string }
  | {
      readonly status: 'idle';
      readonly reason: 'done' | 'stopped';
      readonly text?: string;
    }
  | { readonly status: 'error'; readonly message: string; readonly text?: string };

export type TtsPlaybackListener = (event: TtsPlaybackEvent) => void;

/** Batas UI ke implementasi scanner. Frame kamera tidak boleh melewati port ini. */
export interface ScannerPort {
  getAvailability(): Promise<ScannerAvailability>;
  subscribe(listener: LiveScanListener): () => void;
  start(): Promise<ScannerStartResult>;
  pause(): Promise<PortResult>;
  resume(): Promise<PortResult>;
  stop(): Promise<PortResult>;
  setTorch(enabled: boolean): Promise<PortResult>;
  /** Opsional untuk adapter yang dapat melaporkan kemampuan perangkat aktif. */
  getCapabilities?(): ScannerCapabilities;
  subscribeCapabilities?(listener: ScannerCapabilitiesListener): () => void;
}

/** Kesiapan ringan jalur pemindaian tanpa memuat runtime kamera atau TFLite. */
export interface ScanReadinessPort {
  getAvailability(): Promise<ScanReadiness>;
}

/** Izin kamera dipisahkan agar UI tidak bergantung pada library kamera. */
export interface CameraPermissionPort {
  getStatus(): Promise<CameraPermissionSnapshot>;
  request(): Promise<CameraPermissionSnapshot>;
  openAppSettings(): Promise<PortResult>;
}

/** Batas TTS lokal; implementasi native wajib memverifikasi voice offline. */
export interface TtsPort {
  getAvailability(): Promise<TtsAvailability>;
  /** Membatalkan snapshot voice agar pemeriksaan berikutnya membaca status OS terbaru. */
  invalidateAvailabilityCache?(): void;
  speak(text: string, options: TtsSpeakOptions): Promise<PortResult>;
  stop(): Promise<PortResult>;
  repeat(): Promise<PortResult>;
  openTtsSettings(): Promise<PortResult>;
  /** Opsional untuk adapter yang dapat melaporkan lifecycle utterance native. */
  subscribePlayback?(listener: TtsPlaybackListener): () => void;
}
