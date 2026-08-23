/** Status wire contract yang dikunci oleh PRD Technical Alpha. */
export type ScanStatus =
  | "searching"
  | "quality_issue"
  | "candidate"
  | "stable"
  | "unsupported"
  | "error";

/** Alasan quality gate yang dikunci oleh PRD Technical Alpha. */
export type QualityReason =
  | "blur"
  | "dark"
  | "glare"
  | "too_far"
  | "tilted"
  | "no_grid"
  | "low_confidence";

/**
 * Kotak overlay dalam koordinat ternormalisasi 0..1.
 *
 * Tipe ini sengaja tidak membawa crop, bitmap, tensor, atau data frame lain.
 */
export interface NormalizedBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** DTO kecil dari batas scanner native ke React. */
export interface LiveScanEvent {
  readonly schemaVersion: 1;
  readonly sessionId: string;
  readonly timestampMs: number;
  readonly status: ScanStatus;
  readonly reason?: QualityReason;
  readonly text?: string;
  readonly confidence?: number;
  readonly boxes: readonly NormalizedBox[];
  readonly decoderVersion: string;
  readonly modelVersion?: string;
}

const SCAN_STATUSES: readonly ScanStatus[] = [
  'searching',
  'quality_issue',
  'candidate',
  'stable',
  'unsupported',
  'error',
];

const QUALITY_REASONS: readonly QualityReason[] = [
  'blur',
  'dark',
  'glare',
  'too_far',
  'tilted',
  'no_grid',
  'low_confidence',
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isNormalizedBox(value: unknown): value is NormalizedBox {
  if (!isRecord(value)) {
    return false;
  }

  const coordinates = [value.x, value.y, value.width, value.height];
  if (
    !coordinates.every(
      (coordinate) =>
        typeof coordinate === 'number' &&
        Number.isFinite(coordinate) &&
        coordinate >= 0 &&
        coordinate <= 1,
    )
  ) {
    return false;
  }

  return (
    (value.x as number) + (value.width as number) <= 1.000_001 &&
    (value.y as number) + (value.height as number) <= 1.000_001
  );
}

/** Validasi runtime untuk event yang masuk dari batas native. */
export function isValidLiveScanEvent(value: unknown): value is LiveScanEvent {
  if (!isRecord(value)) {
    return false;
  }

  if (
    value.schemaVersion !== 1 ||
    typeof value.sessionId !== 'string' ||
    value.sessionId.trim().length === 0 ||
    typeof value.timestampMs !== 'number' ||
    !Number.isFinite(value.timestampMs) ||
    value.timestampMs < 0 ||
    typeof value.status !== 'string' ||
    !SCAN_STATUSES.includes(value.status as ScanStatus) ||
    !Array.isArray(value.boxes) ||
    !value.boxes.every(isNormalizedBox) ||
    typeof value.decoderVersion !== 'string' ||
    value.decoderVersion.trim().length === 0
  ) {
    return false;
  }

  if (
    value.reason !== undefined &&
    (typeof value.reason !== 'string' ||
      !QUALITY_REASONS.includes(value.reason as QualityReason))
  ) {
    return false;
  }
  if (value.text !== undefined && typeof value.text !== 'string') {
    return false;
  }
  if (
    value.confidence !== undefined &&
    (typeof value.confidence !== 'number' ||
      !Number.isFinite(value.confidence) ||
      value.confidence < 0 ||
      value.confidence > 1)
  ) {
    return false;
  }
  if (
    value.modelVersion !== undefined &&
    (typeof value.modelVersion !== 'string' ||
      value.modelVersion.trim().length === 0)
  ) {
    return false;
  }

  return true;
}
