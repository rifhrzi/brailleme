import { Asset } from 'expo-asset';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  loadTensorflowModel,
  type TensorflowModel,
} from 'react-native-fast-tflite';

import {
  ALLOW_UNVERIFIED_MODEL_IN_PRODUCTION,
  IS_PRODUCTION_BUILD,
} from '@/config';
import {
  BRAILLE_MODEL_INPUT_CHANNELS,
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_LENGTH,
  BRAILLE_MODEL_INPUT_WIDTH,
  BRAILLE_MODEL_OUTPUT_LENGTH,
  BRAILLE_MODEL_RELEASE_VERIFIED,
} from '@/ml/braille-model-contract';

// Metro membutuhkan literal require agar model ikut dipaketkan sebagai aset.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const BRAILLE_MODEL_ASSET = require('../../assets/models/braille-cell-classifier-v1/braille_model.tflite');

export type BrailleModelState =
  | { readonly status: 'loading'; readonly model?: never; readonly error?: never }
  | {
      readonly status: 'ready';
      readonly model: TensorflowModel;
      readonly error?: never;
    }
  | {
      readonly status: 'error';
      readonly model?: never;
      readonly error: Error;
      readonly retry: () => void;
    };

type BrailleModelLoadState = Exclude<BrailleModelState, { status: 'error' }> | {
  readonly status: 'error';
  readonly model?: never;
  readonly error: Error;
};

function sameShape(actual: readonly number[], expected: readonly number[]) {
  return (
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

function disposeBrailleModel(model: TensorflowModel) {
  try {
    model.dispose();
  } catch {
    // Native cleanup must not replace a useful validation error or crash unmount.
  }
}

export function assertBrailleModelRuntimeAllowed(
  isProductionBuild: boolean,
  releaseVerified: boolean,
  allowUnverifiedModel = false,
) {
  if (isProductionBuild && !releaseVerified && !allowUnverifiedModel) {
    throw new Error(
      'Build produksi menolak classifier yang belum terverifikasi.',
    );
  }
}

async function loadAndValidateBrailleModel(): Promise<TensorflowModel> {
  assertBrailleModelRuntimeAllowed(
    IS_PRODUCTION_BUILD,
    BRAILLE_MODEL_RELEASE_VERIFIED,
    ALLOW_UNVERIFIED_MODEL_IN_PRODUCTION,
  );

  const [asset] = await Asset.loadAsync(BRAILLE_MODEL_ASSET);
  const localUri = asset?.localUri;

  if (!localUri?.startsWith('file://')) {
    throw new Error('Aset model tidak tersedia sebagai file lokal.');
  }

  const model = await loadTensorflowModel({ url: localUri }, []);

  try {
    const [input] = model.inputs;
    const [output] = model.outputs;
    if (
      model.inputs.length !== 1 ||
      !input ||
      input.dataType !== 'float32' ||
      !sameShape(input.shape, [
        1,
        BRAILLE_MODEL_INPUT_HEIGHT,
        BRAILLE_MODEL_INPUT_WIDTH,
        BRAILLE_MODEL_INPUT_CHANNELS,
      ])
    ) {
      throw new Error('Kontrak input model tidak cocok.');
    }
    if (
      model.outputs.length !== 1 ||
      !output ||
      output.dataType !== 'float32' ||
      !sameShape(output.shape, [1, BRAILLE_MODEL_OUTPUT_LENGTH])
    ) {
      throw new Error('Kontrak output model tidak cocok.');
    }

    const zeroInput = new Float32Array(
      new ArrayBuffer(BRAILLE_MODEL_INPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT),
    );
    const [outputBuffer] = await model.run([zeroInput.buffer]);
    if (!outputBuffer) {
      throw new Error('Model tidak menghasilkan tensor output.');
    }

    if (
      outputBuffer.byteLength !==
      BRAILLE_MODEL_OUTPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error('Ukuran tensor output model tidak cocok.');
    }

    const probabilities = new Float32Array(outputBuffer);
    const probabilitySum = probabilities.reduce((sum, value) => {
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error('Model menghasilkan probabilitas yang tidak valid.');
      }
      return sum + value;
    }, 0);

    if (
      probabilities.length !== BRAILLE_MODEL_OUTPUT_LENGTH ||
      probabilitySum < 0.99 ||
      probabilitySum > 1.01
    ) {
      throw new Error('Self-test Softmax model gagal.');
    }

    return model;
  } catch (error) {
    disposeBrailleModel(model);
    throw error;
  }
}

export function useBrailleModel(): BrailleModelState {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<BrailleModelLoadState>({
    status: 'loading',
  });
  const retryAvailableRef = useRef(false);

  const retry = useCallback(() => {
    if (!retryAvailableRef.current) {
      return;
    }

    retryAvailableRef.current = false;
    setState({ status: 'loading' });
    setAttempt((current) => current + 1);
  }, []);

  useEffect(() => {
    let active = true;
    let loadedModel: TensorflowModel | undefined;
    retryAvailableRef.current = false;

    void loadAndValidateBrailleModel()
      .then((model) => {
        loadedModel = model;
        if (active) {
          retryAvailableRef.current = false;
          setState({ status: 'ready', model });
        } else {
          disposeBrailleModel(model);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          retryAvailableRef.current = true;
          setState({
            status: 'error',
            error: error instanceof Error ? error : new Error('Model gagal dimuat.'),
          });
        }
      });

    return () => {
      active = false;
      retryAvailableRef.current = false;
      if (loadedModel) {
        disposeBrailleModel(loadedModel);
      }
    };
  }, [attempt]);

  return state.status === 'error' ? { ...state, retry } : state;
}
