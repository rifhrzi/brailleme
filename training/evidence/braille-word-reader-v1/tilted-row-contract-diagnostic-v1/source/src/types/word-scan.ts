import { WORD_LATIN_MASKS, WORD_MAX_LETTERS } from '@/ml/word-model-contract';
import type { NormalizedBox } from '@/types/live-scan';

export type WordRejectionReason =
  | 'dark' | 'glare' | 'blur' | 'too_far' | 'no_grid' | 'low_confidence'
  | 'clipped' | 'multiple_rows' | 'gap' | 'overlapping_cells' | 'too_many_cells'
  | 'unsupported_pattern';
export type WordScanReason = WordRejectionReason | 'processing_error' | 'capture_error';
export type WordScanStatus = 'ready' | 'capturing' | 'processing' | 'success' | 'rejected' | 'error';

export interface WordCell {
  readonly letter: string;
  readonly mask: number;
  /** Minimum of objectness and six selected dot probabilities, not a word probability. */
  readonly confidence: number;
  readonly box: NormalizedBox;
}

/** Contains no photo, pixels, URI, bitmap, or tensor. */
export interface WordScanEvent {
  readonly schemaVersion: 3;
  readonly sessionId: string;
  readonly captureId: number;
  readonly timestampMs: number;
  readonly status: WordScanStatus;
  readonly reason?: WordScanReason;
  readonly text?: string;
  readonly confidence?: number;
  readonly cells: readonly WordCell[];
  /** Normalized upright photo coordinates; retained for rejected diagnostic overlays. */
  readonly boxes: readonly NormalizedBox[];
  readonly decoderVersion: string;
  readonly modelVersion?: string;
}

export type WordImageResult =
  | { readonly status: 'success'; readonly text: string; readonly confidence: number; readonly cells: readonly WordCell[]; readonly boxes: readonly NormalizedBox[] }
  | { readonly status: 'rejected'; readonly reason: WordRejectionReason; readonly cells: readonly []; readonly boxes: readonly NormalizedBox[] }
  | { readonly status: 'error'; readonly reason: 'processing_error'; readonly cells: readonly []; readonly boxes: readonly [] };

export function isNormalizedWordBox(value: unknown): value is NormalizedBox {
  'worklet';
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const box = value as NormalizedBox;
  return [box.x, box.y, box.width, box.height].every(Number.isFinite) &&
    box.x >= 0 && box.y >= 0 && box.width > 0 && box.height > 0 &&
    box.x + box.width <= 1.000001 && box.y + box.height <= 1.000001;
}

const reasons: readonly WordScanReason[] = [
  'dark', 'glare', 'blur', 'too_far', 'no_grid', 'low_confidence', 'clipped',
  'multiple_rows', 'gap', 'overlapping_cells', 'too_many_cells', 'unsupported_pattern',
  'processing_error', 'capture_error',
];

export function isValidWordScanEvent(value: unknown): value is WordScanEvent {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const event = value as WordScanEvent;
  if (event.schemaVersion !== 3 || typeof event.sessionId !== 'string' || !event.sessionId.trim() ||
      !Number.isSafeInteger(event.captureId) || event.captureId < 0 ||
      !Number.isFinite(event.timestampMs) || event.timestampMs < 0 ||
      !['ready', 'capturing', 'processing', 'success', 'rejected', 'error'].includes(event.status) ||
      typeof event.decoderVersion !== 'string' || !event.decoderVersion.trim() ||
      (event.modelVersion !== undefined && (typeof event.modelVersion !== 'string' || !event.modelVersion.trim())) ||
      !Array.isArray(event.cells) || !Array.isArray(event.boxes) || !event.boxes.every(isNormalizedWordBox) ||
      (event.reason !== undefined && !reasons.includes(event.reason))) return false;

  if (event.status !== 'success') {
    return event.text === undefined && event.confidence === undefined && event.cells.length === 0 &&
      (event.status === 'rejected' ? event.reason !== undefined && event.reason !== 'processing_error' && event.reason !== 'capture_error' :
        event.status === 'error' ? event.reason === 'processing_error' || event.reason === 'capture_error' : event.reason === undefined && event.boxes.length === 0);
  }

  if (event.reason !== undefined || typeof event.text !== 'string' || !/^[A-Z]{1,12}$/.test(event.text) ||
      event.cells.length !== event.text.length || event.cells.length > WORD_MAX_LETTERS ||
      event.boxes.length !== event.cells.length || !Number.isFinite(event.confidence) ||
      event.confidence === undefined || event.confidence < 0 || event.confidence > 1) return false;
  let previousCenter = -1;
  let minimumConfidence = 1;
  for (let index = 0; index < event.cells.length; index += 1) {
    const cell = event.cells[index];
    const box = event.boxes[index];
    if (!cell || typeof cell !== 'object' || !box || !isNormalizedWordBox(cell.box) ||
        cell.letter !== event.text[index] || cell.mask !== WORD_LATIN_MASKS[cell.letter.charCodeAt(0) - 65] ||
        !Number.isFinite(cell.confidence) || cell.confidence < 0 || cell.confidence > 1 ||
        box.x !== cell.box.x || box.y !== cell.box.y || box.width !== cell.box.width || box.height !== cell.box.height) return false;
    const center = box.x + box.width / 2;
    if (center <= previousCenter) return false;
    previousCenter = center;
    minimumConfidence = Math.min(minimumConfidence, cell.confidence);
  }
  return Math.abs(minimumConfidence - event.confidence) <= 0.000001;
}
