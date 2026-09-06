/** Tensor contract shared by training, the loader, and the photo worklet. */
export const WORD_MODEL_INPUT_WIDTH = 480;
export const WORD_MODEL_INPUT_HEIGHT = 192;
export const WORD_MODEL_INPUT_CHANNELS = 3;
export const WORD_MODEL_INPUT_LENGTH = 480 * 192 * 3;
export const WORD_MODEL_GRID_WIDTH = 120;
export const WORD_MODEL_GRID_HEIGHT = 48;
export const WORD_MODEL_OUTPUT_CHANNELS = 11;
export const WORD_MODEL_OUTPUT_LENGTH = 120 * 48 * 11;
export const WORD_MODEL_INPUT_SHAPE = [1, 192, 480, 3] as const;
export const WORD_MODEL_OUTPUT_SHAPE = [1, 48, 120, 11] as const;
export const WORD_MODEL_INPUT_SCALE = 'zero_to_one' as const;
export const WORD_DECODER_VERSION = 'six-dot-word-v2';
export const WORD_MAX_LETTERS = 12;
export const WORD_GUIDE_WIDTH_FRACTION = 0.9;
/** Adjustable visible word frame; the model receives context around it. */
export const WORD_GUIDE_ASPECT_RATIO = 4;
export const WORD_GUIDE_MIN_ASPECT_RATIO = 1;
export const WORD_GUIDE_MAX_ASPECT_RATIO = 8;
export const WORD_MODEL_CONTEXT_ASPECT_RATIO = 2.5;

/** Dot 1 is the least significant bit; left column is dots 1, 2, 3. */
export const WORD_LATIN_MASKS = [
  1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13,
  29, 21, 15, 31, 23, 14, 30, 37, 39, 58, 45, 61, 53,
] as const;

export interface WordModelThresholds {
  /** Retain local maxima above this level, even if they cannot be accepted. */
  readonly proposalScore: number;
  readonly acceptScore: number;
  /** Required probability of the selected present/absent state of EVERY dot. */
  readonly dotConfidence: number;
  readonly nmsIou: number;
  readonly minCellWidth: number;
  readonly minCellHeight: number;
  /** Required margin from each ROI edge, normalized separately on each axis. */
  readonly edgeMargin: number;
  /** Allowed center-Y span divided by median cell height. */
  readonly maxRowDeviation: number;
  /** Allowed neighboring center-X distance / median cell height, in pixels. */
  readonly maxGapRatio: number;
  /** Allowed horizontal intersection / narrower cell width. */
  readonly maxOverlapRatio: number;
}

/** Diagnostic starting values only. Active values must come from the manifest. */
export const DEFAULT_WORD_MODEL_THRESHOLDS: WordModelThresholds = {
  proposalScore: 0.25,
  acceptScore: 0.65,
  dotConfidence: 0.9,
  nmsIou: 0.25,
  minCellWidth: 0.008,
  minCellHeight: 0.06,
  edgeMargin: 0.008,
  maxRowDeviation: 0.55,
  maxGapRatio: 1.2,
  maxOverlapRatio: 0.5,
};

export function isValidWordModelThresholds(value: unknown): value is WordModelThresholds {
  'worklet';
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const candidate = value as Record<string, unknown>;
  const keys = [
    'proposalScore', 'acceptScore', 'dotConfidence', 'nmsIou',
    'minCellWidth', 'minCellHeight', 'edgeMargin',
    'maxRowDeviation', 'maxGapRatio', 'maxOverlapRatio',
  ];
  if (!keys.every((key) => typeof candidate[key] === 'number' && Number.isFinite(candidate[key]))) return false;
  const thresholds = value as WordModelThresholds;
  return thresholds.proposalScore > 0 && thresholds.proposalScore <= thresholds.acceptScore &&
    thresholds.acceptScore <= 1 && thresholds.dotConfidence > 0.5 && thresholds.dotConfidence <= 1 &&
    thresholds.nmsIou > 0 && thresholds.nmsIou < 1 &&
    thresholds.minCellWidth > 0 && thresholds.minCellWidth < 1 &&
    thresholds.minCellHeight > 0 && thresholds.minCellHeight < 1 &&
    thresholds.edgeMargin >= 0 && thresholds.edgeMargin < 0.5 &&
    thresholds.maxRowDeviation > 0 && thresholds.maxRowDeviation <= 1 &&
    thresholds.maxGapRatio > 0 && thresholds.maxOverlapRatio >= 0 && thresholds.maxOverlapRatio < 1;
}
