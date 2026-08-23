import { BRAILLE_MODEL_OUTPUT_LENGTH } from '@/ml/braille-model-contract';

export const PROVISIONAL_MIN_CONFIDENCE = 0.9;
export const PROVISIONAL_MIN_MARGIN = 0.25;
export const REQUIRED_CONSECUTIVE_PREDICTIONS = 3;
export const REQUIRED_DWELL_MS = 500;
export const MAX_INTER_PREDICTION_GAP_MS = 750;

export type ModelPrediction = {
  readonly index: number;
  readonly confidence: number;
  readonly runnerUpConfidence: number;
  readonly margin: number;
};

export type PredictionStabilizerState = {
  readonly candidateIndex: number | null;
  readonly consecutiveMatches: number;
  readonly firstSeenAtMs: number;
  readonly lastSeenAtMs: number;
  readonly stableIndex: number | null;
};

export type StabilizerOutcome =
  | { readonly status: 'low_confidence'; readonly state: PredictionStabilizerState }
  | {
      readonly status: 'candidate';
      readonly state: PredictionStabilizerState;
      readonly prediction: ModelPrediction;
    }
  | {
      readonly status: 'stable';
      readonly state: PredictionStabilizerState;
      readonly prediction: ModelPrediction;
    };

export const INITIAL_PREDICTION_STATE: PredictionStabilizerState = Object.freeze({
  candidateIndex: null,
  consecutiveMatches: 0,
  firstSeenAtMs: 0,
  lastSeenAtMs: 0,
  stableIndex: null,
});

export function decodeProbabilities(
  probabilities: ArrayLike<number>,
): ModelPrediction | null {
  'worklet';

  if (probabilities.length !== BRAILLE_MODEL_OUTPUT_LENGTH) {
    return null;
  }

  let bestIndex = -1;
  let best = Number.NEGATIVE_INFINITY;
  let runnerUp = Number.NEGATIVE_INFINITY;
  let sum = 0;

  for (let index = 0; index < probabilities.length; index += 1) {
    const value = probabilities[index];
    if (
      value === undefined ||
      !Number.isFinite(value) ||
      value < 0 ||
      value > 1
    ) {
      return null;
    }
    sum += value;
    if (value > best) {
      runnerUp = best;
      best = value;
      bestIndex = index;
    } else if (value > runnerUp) {
      runnerUp = value;
    }
  }

  if (
    bestIndex < 0 ||
    best < 0 ||
    best > 1 ||
    runnerUp < 0 ||
    runnerUp > 1 ||
    sum < 0.99 ||
    sum > 1.01
  ) {
    return null;
  }

  return {
    index: bestIndex,
    confidence: best,
    runnerUpConfidence: runnerUp,
    margin: best - runnerUp,
  };
}

export function advancePredictionStabilizer(
  previous: PredictionStabilizerState,
  prediction: ModelPrediction,
  timestampMs: number,
): StabilizerOutcome {
  const validPrediction =
    Number.isInteger(prediction.index) &&
    prediction.index >= 0 &&
    prediction.index < BRAILLE_MODEL_OUTPUT_LENGTH &&
    Number.isFinite(prediction.confidence) &&
    prediction.confidence >= 0 &&
    prediction.confidence <= 1 &&
    Number.isFinite(prediction.runnerUpConfidence) &&
    prediction.runnerUpConfidence >= 0 &&
    prediction.runnerUpConfidence <= prediction.confidence &&
    Number.isFinite(prediction.margin) &&
    prediction.margin >= 0 &&
    Number.isFinite(timestampMs) &&
    timestampMs >= 0;
  const eligible =
    validPrediction &&
    prediction.confidence >= PROVISIONAL_MIN_CONFIDENCE &&
    prediction.margin >= PROVISIONAL_MIN_MARGIN;

  if (!eligible) {
    return { status: 'low_confidence', state: INITIAL_PREDICTION_STATE };
  }

  const interPredictionGapMs = timestampMs - previous.lastSeenAtMs;
  const sameCandidate =
    previous.candidateIndex === prediction.index &&
    previous.consecutiveMatches > 0 &&
    interPredictionGapMs >= 0 &&
    interPredictionGapMs <= MAX_INTER_PREDICTION_GAP_MS;
  const state: PredictionStabilizerState = {
    candidateIndex: prediction.index,
    consecutiveMatches: sameCandidate ? previous.consecutiveMatches + 1 : 1,
    firstSeenAtMs: sameCandidate ? previous.firstSeenAtMs : timestampMs,
    lastSeenAtMs: timestampMs,
    stableIndex:
      sameCandidate && previous.stableIndex === prediction.index
        ? prediction.index
        : null,
  };
  const dwellMs = Math.max(0, timestampMs - state.firstSeenAtMs);
  const stable =
    state.stableIndex === prediction.index ||
    (state.consecutiveMatches >= REQUIRED_CONSECUTIVE_PREDICTIONS &&
      dwellMs >= REQUIRED_DWELL_MS);

  if (stable) {
    return {
      status: 'stable',
      prediction,
      state: { ...state, stableIndex: prediction.index },
    };
  }

  return { status: 'candidate', prediction, state };
}
