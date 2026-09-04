import { Asset } from 'expo-asset';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  loadTensorflowModel,
  type TensorflowModel,
} from 'react-native-fast-tflite';
import { loadImage, type Image } from 'react-native-nitro-image';

import { IS_PRODUCTION_BUILD } from '@/config';
import {
  BRAILLE_MODEL_INPUT_CHANNELS,
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_LENGTH,
  BRAILLE_MODEL_INPUT_WIDTH,
  BRAILLE_MODEL_INPUT_SCALE,
  BRAILLE_MODEL_OUTPUT_LENGTH,
  BRAILLE_MODEL_RELEASE_VERIFIED,
  BRAILLE_MODEL_VERSION,
  BRAILLE_NATIVE_SELF_TEST_EXPECTED_INDEX,
  BRAILLE_NATIVE_SELF_TEST_EXPECTED_LABEL,
  BRAILLE_NATIVE_SELF_TEST_MIN_CONFIDENCE,
} from '@/ml/braille-model-contract';
import {
  assertBrailleModelInputLength,
  centerCropAndResizeBrailleImage,
  convertRawPixelsToRgbTensor,
} from '@/ml/frame-preprocessing';
import { decodeProbabilities } from '@/ml/prediction-stabilizer';
import { assertBrailleModelRuntimeAllowed } from '@/ml/braille-model-runtime-policy';

export { assertBrailleModelRuntimeAllowed } from '@/ml/braille-model-runtime-policy';

// Metro membutuhkan literal require agar model ikut dipaketkan sebagai aset.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const BRAILLE_MODEL_ASSET = require('../../assets/models/braille-cell-classifier-v2/braille_model.tflite');
// Fixture 51x51 menempatkan crop 28x28 tepat di tengah untuk menguji helper
// crop, decoder NitroImage, channel-order, normalisasi, dan runSync native.
const BRAILLE_NATIVE_SELF_TEST_ASSET = require('../../assets/models/braille-cell-classifier-v2/native-self-test-v.png');

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

function disposeNativeImage(image: Image | undefined) {
  try {
    image?.dispose();
  } catch {
    // Semua resource lain tetap harus dibersihkan bila satu dispose gagal.
  }
}

type NativePostCropSelfTestResult = {
  readonly confidence: number;
  readonly label: string;
  readonly pixelFormat: string;
};

async function runNativePostCropSelfTest(
  model: TensorflowModel,
  fixtureUri: string,
): Promise<NativePostCropSelfTestResult> {
  let sourceImage: Image | undefined;
  let croppedImage: Image | undefined;
  let resizedImage: Image | undefined;

  try {
    sourceImage = await loadImage({ filePath: fixtureUri });
    ({ croppedImage, resizedImage } =
      centerCropAndResizeBrailleImage(sourceImage));
    const rawPixels = resizedImage.toRawPixelData(false);
    const preprocessed = convertRawPixelsToRgbTensor(
      rawPixels,
      BRAILLE_MODEL_INPUT_SCALE,
    );
    assertBrailleModelInputLength(preprocessed.tensor);

    const [outputBuffer] = model.runSync([preprocessed.tensor.buffer]);
    if (!outputBuffer) {
      throw new Error('Self-test native tidak menghasilkan tensor output.');
    }
    if (
      outputBuffer.byteLength !==
      BRAILLE_MODEL_OUTPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error('Ukuran tensor output self-test native tidak cocok.');
    }

    const prediction = decodeProbabilities(new Float32Array(outputBuffer));
    if (!prediction) {
      throw new Error('Probabilitas self-test native tidak valid.');
    }
    if (prediction.index !== BRAILLE_NATIVE_SELF_TEST_EXPECTED_INDEX) {
      throw new Error(
        `Self-test native memprediksi indeks ${prediction.index}, bukan ${BRAILLE_NATIVE_SELF_TEST_EXPECTED_INDEX}.`,
      );
    }
    if (prediction.confidence < BRAILLE_NATIVE_SELF_TEST_MIN_CONFIDENCE) {
      throw new Error(
        `Confidence self-test native ${prediction.confidence.toFixed(6)} di bawah ${BRAILLE_NATIVE_SELF_TEST_MIN_CONFIDENCE.toFixed(2)}.`,
      );
    }

    return {
      confidence: prediction.confidence,
      label: BRAILLE_NATIVE_SELF_TEST_EXPECTED_LABEL,
      pixelFormat: rawPixels.pixelFormat,
    };
  } finally {
    disposeNativeImage(resizedImage);
    if (croppedImage !== resizedImage) disposeNativeImage(croppedImage);
    if (sourceImage !== croppedImage && sourceImage !== resizedImage) {
      disposeNativeImage(sourceImage);
    }
  }
}

async function loadAndValidateBrailleModel(): Promise<TensorflowModel> {
  assertBrailleModelRuntimeAllowed(
    IS_PRODUCTION_BUILD,
    BRAILLE_MODEL_RELEASE_VERIFIED,
  );

  const [asset, nativeSelfTestAsset] = await Asset.loadAsync([
    BRAILLE_MODEL_ASSET,
    BRAILLE_NATIVE_SELF_TEST_ASSET,
  ]);
  const localUri = asset?.localUri;
  const nativeSelfTestUri = nativeSelfTestAsset?.localUri;

  if (
    !localUri?.startsWith('file://') ||
    !nativeSelfTestUri?.startsWith('file://')
  ) {
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

    const nativeSelfTest = await runNativePostCropSelfTest(
      model,
      nativeSelfTestUri,
    );
    if (__DEV__ && process.env.NODE_ENV !== 'test') {
      console.info(
        'BRAILLE_NATIVE_POST_CROP_SELF_TEST',
        JSON.stringify({
          ...nativeSelfTest,
          modelVersion: BRAILLE_MODEL_VERSION,
          passed: true,
        }),
      );
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
