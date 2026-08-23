import { BRAILLE_MODEL_OUTPUT_LENGTH } from '@/ml/braille-model-contract';
import {
  advancePredictionStabilizer,
  decodeProbabilities,
  INITIAL_PREDICTION_STATE,
  MAX_INTER_PREDICTION_GAP_MS,
  PROVISIONAL_MIN_CONFIDENCE,
  PROVISIONAL_MIN_MARGIN,
  REQUIRED_CONSECUTIVE_PREDICTIONS,
  REQUIRED_DWELL_MS,
  type ModelPrediction,
  type PredictionStabilizerState,
} from '@/ml/prediction-stabilizer';

function createProbabilities(
  entries: readonly (readonly [index: number, value: number])[],
): Float32Array {
  const probabilities = new Float32Array(BRAILLE_MODEL_OUTPUT_LENGTH);
  for (const [index, value] of entries) {
    probabilities[index] = value;
  }
  return probabilities;
}

function createPrediction(
  index: number,
  confidence = 0.95,
  margin = 0.4,
): ModelPrediction {
  return {
    index,
    confidence,
    runnerUpConfidence: confidence - margin,
    margin,
  };
}

function expectCandidate(
  outcome: ReturnType<typeof advancePredictionStabilizer>,
): PredictionStabilizerState {
  expect(outcome.status).toBe('candidate');
  return outcome.state;
}

describe('decodeProbabilities', () => {
  it('mengambil kelas terbaik, runner-up, dan margin', () => {
    const prediction = decodeProbabilities(
      createProbabilities([
        [2, 0.06],
        [7, 0.91],
        [19, 0.03],
      ]),
    );

    expect(prediction).not.toBeNull();
    expect(prediction?.index).toBe(7);
    expect(prediction?.confidence).toBeCloseTo(0.91, 6);
    expect(prediction?.runnerUpConfidence).toBeCloseTo(0.06, 6);
    expect(prediction?.margin).toBeCloseTo(0.85, 6);
  });

  it('memilih indeks pertama secara deterministik ketika nilai terbaik seri', () => {
    expect(
      decodeProbabilities(
        createProbabilities([
          [3, 0.45],
          [8, 0.45],
          [10, 0.1],
        ]),
      ),
    ).toMatchObject({
      index: 3,
      confidence: expect.closeTo(0.45, 6),
      runnerUpConfidence: expect.closeTo(0.45, 6),
      margin: expect.closeTo(0, 6),
    });
  });

  it.each([
    ['panjang salah', new Float32Array(BRAILLE_MODEL_OUTPUT_LENGTH - 1)],
    ['NaN', createProbabilities([[4, Number.NaN]])],
    ['infinity', createProbabilities([[4, Number.POSITIVE_INFINITY]])],
    ['di atas satu', createProbabilities([[4, 1.01]])],
    [
      'negatif meski bukan dua nilai teratas',
      createProbabilities([
        [1, 0.7],
        [2, 0.2],
        [3, -0.1],
      ]),
    ],
    ['jumlah bukan Softmax', createProbabilities([[4, 0.8]])],
  ])('menolak probabilitas tidak valid: %s', (_caseName, probabilities) => {
    expect(decodeProbabilities(probabilities)).toBeNull();
  });
});

describe('advancePredictionStabilizer', () => {
  it.each([
    [PROVISIONAL_MIN_CONFIDENCE - 0.001, PROVISIONAL_MIN_MARGIN],
    [PROVISIONAL_MIN_CONFIDENCE, PROVISIONAL_MIN_MARGIN - 0.001],
  ])(
    'mereset kandidat bila confidence=%s dan margin=%s tidak lolos',
    (confidence, margin) => {
      const previous: PredictionStabilizerState = {
        candidateIndex: 4,
        consecutiveMatches: 2,
        firstSeenAtMs: 1_000,
        lastSeenAtMs: 1_200,
        stableIndex: null,
      };

      const outcome = advancePredictionStabilizer(
        previous,
        createPrediction(4, confidence, margin),
        1_400,
      );

      expect(outcome).toEqual({
        status: 'low_confidence',
        state: INITIAL_PREDICTION_STATE,
      });
    },
  );

  it('menerima confidence dan margin tepat pada ambang', () => {
    const outcome = advancePredictionStabilizer(
      INITIAL_PREDICTION_STATE,
      createPrediction(
        5,
        PROVISIONAL_MIN_CONFIDENCE,
        PROVISIONAL_MIN_MARGIN,
      ),
      1_000,
    );

    expect(outcome.status).toBe('candidate');
    expect(outcome.state).toEqual({
      candidateIndex: 5,
      consecutiveMatches: 1,
      firstSeenAtMs: 1_000,
      lastSeenAtMs: 1_000,
      stableIndex: null,
    });
  });

  it('mensyaratkan jumlah prediksi dan dwell time sebelum stabil', () => {
    const prediction = createPrediction(6);
    let state = expectCandidate(
      advancePredictionStabilizer(
        INITIAL_PREDICTION_STATE,
        prediction,
        1_000,
      ),
    );
    state = expectCandidate(
      advancePredictionStabilizer(state, prediction, 1_200),
    );

    const stable = advancePredictionStabilizer(
      state,
      prediction,
      1_500,
    );
    expect(stable.status).toBe('stable');
    expect(stable.state.consecutiveMatches).toBe(
      REQUIRED_CONSECUTIVE_PREDICTIONS,
    );
    expect(stable.state.firstSeenAtMs).toBe(1_000);
    expect(stable.state.stableIndex).toBe(6);
  });

  it('tetap candidate ketika hanya jumlah prediksi yang cukup', () => {
    const prediction = createPrediction(7);
    let state = expectCandidate(
      advancePredictionStabilizer(
        INITIAL_PREDICTION_STATE,
        prediction,
        2_000,
      ),
    );
    state = expectCandidate(
      advancePredictionStabilizer(state, prediction, 2_200),
    );
    const outcome = advancePredictionStabilizer(state, prediction, 2_499);

    expect(outcome.status).toBe('candidate');
    expect(outcome.state.consecutiveMatches).toBe(
      REQUIRED_CONSECUTIVE_PREDICTIONS,
    );
    expect(2_499 - outcome.state.firstSeenAtMs).toBe(
      REQUIRED_DWELL_MS - 1,
    );
  });

  it('tetap candidate ketika hanya dwell time yang cukup', () => {
    const prediction = createPrediction(8);
    const firstState = expectCandidate(
      advancePredictionStabilizer(
        INITIAL_PREDICTION_STATE,
        prediction,
        3_000,
      ),
    );
    const outcome = advancePredictionStabilizer(
      firstState,
      prediction,
      3_000 + REQUIRED_DWELL_MS,
    );

    expect(outcome.status).toBe('candidate');
    expect(outcome.state.consecutiveMatches).toBe(
      REQUIRED_CONSECUTIVE_PREDICTIONS - 1,
    );
  });

  it('mereset hitungan dan waktu ketika kelas kandidat berubah', () => {
    const previous: PredictionStabilizerState = {
      candidateIndex: 9,
      consecutiveMatches: 2,
      firstSeenAtMs: 4_000,
      lastSeenAtMs: 4_200,
      stableIndex: 9,
    };

    const outcome = advancePredictionStabilizer(
      previous,
      createPrediction(10),
      4_800,
    );

    expect(outcome.status).toBe('candidate');
    expect(outcome.state).toEqual({
      candidateIndex: 10,
      consecutiveMatches: 1,
      firstSeenAtMs: 4_800,
      lastSeenAtMs: 4_800,
      stableIndex: null,
    });
  });

  it('mempertahankan status stable untuk kelas yang sama', () => {
    const prediction = createPrediction(11);
    const previous: PredictionStabilizerState = {
      candidateIndex: 11,
      consecutiveMatches: REQUIRED_CONSECUTIVE_PREDICTIONS,
      firstSeenAtMs: 5_000,
      lastSeenAtMs: 5_000,
      stableIndex: 11,
    };

    const outcome = advancePredictionStabilizer(
      previous,
      prediction,
      5_100,
    );

    expect(outcome.status).toBe('stable');
    expect(outcome.state).toEqual({
      candidateIndex: 11,
      consecutiveMatches: REQUIRED_CONSECUTIVE_PREDICTIONS + 1,
      firstSeenAtMs: 5_000,
      lastSeenAtMs: 5_100,
      stableIndex: 11,
    });
  });

  it('mereset kandidat stabil setelah jeda antar-frame terlalu lama', () => {
    const previous: PredictionStabilizerState = {
      candidateIndex: 12,
      consecutiveMatches: REQUIRED_CONSECUTIVE_PREDICTIONS,
      firstSeenAtMs: 6_000,
      lastSeenAtMs: 6_500,
      stableIndex: 12,
    };
    const timestampMs = 6_500 + MAX_INTER_PREDICTION_GAP_MS + 1;

    const outcome = advancePredictionStabilizer(
      previous,
      createPrediction(12),
      timestampMs,
    );

    expect(outcome.status).toBe('candidate');
    expect(outcome.state).toEqual({
      candidateIndex: 12,
      consecutiveMatches: 1,
      firstSeenAtMs: timestampMs,
      lastSeenAtMs: timestampMs,
      stableIndex: null,
    });
  });

  it('menolak prediksi atau timestamp nonfinite', () => {
    expect(
      advancePredictionStabilizer(
        INITIAL_PREDICTION_STATE,
        { ...createPrediction(1), confidence: Number.NaN },
        1_000,
      ),
    ).toEqual({
      status: 'low_confidence',
      state: INITIAL_PREDICTION_STATE,
    });
    expect(
      advancePredictionStabilizer(
        INITIAL_PREDICTION_STATE,
        createPrediction(1),
        Number.POSITIVE_INFINITY,
      ),
    ).toEqual({
      status: 'low_confidence',
      state: INITIAL_PREDICTION_STATE,
    });
  });
});
