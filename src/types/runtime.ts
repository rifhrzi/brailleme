export type PortFailureCode = "unavailable" | "failed";

export type PortResult =
  | { readonly ok: true }
  | {
      readonly ok: false;
      readonly code: PortFailureCode;
      readonly message: string;
    };

/** Start scanner mengembalikan identitas sesi untuk menolak event terlambat. */
export type ScannerStartResult =
  | { readonly ok: true; readonly sessionId: string }
  | {
      readonly ok: false;
      readonly code: PortFailureCode;
      readonly message: string;
    };

export type ScannerAvailability =
  | { readonly available: true }
  | {
      readonly available: false;
      readonly reason: "native_scanner_not_installed";
      readonly message: string;
    };

export type ScanReadiness =
  | { readonly available: true }
  | {
      readonly available: false;
      readonly reason: 'model_not_available' | 'scanner_not_available';
      readonly message: string;
    };

export type CameraPermissionStatus =
  | "granted"
  | "undetermined"
  | "denied"
  | "blocked"
  | "unavailable";

export interface CameraPermissionSnapshot {
  readonly status: CameraPermissionStatus;
  readonly canAskAgain: boolean;
}

export type TtsAvailabilityStatus = "available" | "unavailable";

export interface TtsAvailability {
  readonly status: TtsAvailabilityStatus;
  readonly localVoiceAvailable: boolean;
  readonly message: string;
}

export interface TtsSpeakOptions {
  readonly rate: number;
  readonly volume: number;
}
