import { Asset } from 'expo-asset';
import { useCallback, useEffect, useRef, useState } from 'react';
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite';
import { loadImage, type Image } from 'react-native-nitro-image';
import { createWorkletRuntime, runOnRuntimeAsync } from 'react-native-worklets';

import { IS_PRODUCTION_BUILD } from '@/config';
import {
  WORD_MODEL_ASSET, WORD_MODEL_BUNDLED, WORD_MODEL_MANIFEST,
  WORD_MODEL_RELEASE_VERIFIED, WORD_MODEL_VERSION, WORD_NATIVE_FIXTURE_ASSET,
} from '@/ml/word-model-assets';
import {
  isValidWordModelThresholds, WORD_DECODER_VERSION, WORD_MODEL_INPUT_LENGTH, WORD_MODEL_INPUT_SHAPE,
  WORD_MODEL_OUTPUT_LENGTH, WORD_MODEL_OUTPUT_SHAPE, type WordModelThresholds,
} from '@/ml/word-model-contract';
import { recognizeWordImage } from '@/ml/word-image-processing';
import type { NormalizedBox } from '@/types/live-scan';

export type AcquireWordModel = () => () => void;

/** Keep the native model alive until all background photo operations settle. */
export function createWordModelLease(model: Pick<TensorflowModel, 'dispose'>) {
  let borrowers = 0;
  let closing = false;
  let disposed = false;
  const disposeIfIdle = () => {
    if (closing && borrowers === 0 && !disposed) {
      disposed = true;
      try { model.dispose(); } catch { /* Native cleanup must remain idempotent. */ }
    }
  };
  const acquire: AcquireWordModel = () => {
    if (closing || disposed) throw new Error('Pengenal kata sudah ditutup.');
    borrowers += 1;
    let released = false;
    return () => {
      if (released) return;
      released = true;
      borrowers -= 1;
      disposeIfIdle();
    };
  };
  return { acquire, close: () => { closing = true; disposeIfIdle(); } };
}

type ReadyWordModel = {
  readonly status: 'ready';
  readonly model: TensorflowModel;
  readonly thresholds: WordModelThresholds;
  readonly modelVersion: string;
  readonly acquireModel: AcquireWordModel;
};
type LoadState =
  | { readonly status: 'loading' }
  | { readonly status: 'error'; readonly error: Error }
  | ReadyWordModel;
export type WordModelState = Exclude<LoadState, { status: 'error' }> |
  { readonly status: 'error'; readonly error: Error; readonly retry: () => void };

type NativeWordFixture = {
  readonly expectedText: string;
  readonly roi: NormalizedBox;
};

export async function loadAndValidateWordModel(): Promise<TensorflowModel> {
  if ((IS_PRODUCTION_BUILD && !WORD_MODEL_RELEASE_VERIFIED) || !WORD_MODEL_BUNDLED ||
      WORD_MODEL_ASSET === null || WORD_NATIVE_FIXTURE_ASSET === null) {
    throw new Error('Pengenal kata belum tersedia pada build ini.');
  }
  const thresholds = WORD_MODEL_MANIFEST.thresholds;
  if (WORD_MODEL_MANIFEST.decoderVersion !== WORD_DECODER_VERSION) {
    throw new Error('Konfigurasi pembacaan pengenal kata tidak sesuai.');
  }
  if (!isValidWordModelThresholds(thresholds)) throw new Error('Konfigurasi pengenal kata tidak sesuai.');
  const fixture = WORD_MODEL_MANIFEST.nativeSelfTest as NativeWordFixture | null;
  if (!fixture || !/^[A-Z]{1,12}$/.test(fixture.expectedText)) {
    throw new Error('Pemeriksaan awal pengenal kata belum tersedia.');
  }
  const [asset] = await Asset.loadAsync(WORD_MODEL_ASSET);
  if (!asset?.localUri?.startsWith('file://')) throw new Error('Pengenal kata tidak tersedia secara lokal.');
  const model = await loadTensorflowModel({ url: asset.localUri }, []);
  try {
    const input = model.inputs[0];
    const output = model.outputs[0];
    if (model.inputs.length !== 1 || model.outputs.length !== 1 ||
        input?.dataType !== 'float32' || output?.dataType !== 'float32' ||
        input.shape.join(',') !== WORD_MODEL_INPUT_SHAPE.join(',') ||
        output.shape.join(',') !== WORD_MODEL_OUTPUT_SHAPE.join(',')) {
      throw new Error('Kontrak tensor pengenal kata tidak sesuai.');
    }
    const [buffer] = await model.run([new Float32Array(WORD_MODEL_INPUT_LENGTH).buffer]);
    if (!buffer || buffer.byteLength !== WORD_MODEL_OUTPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT ||
        !new Float32Array(buffer).every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
      throw new Error('Pengenal kata menghasilkan tensor yang tidak valid.');
    }
    const image = await loadImage(WORD_NATIVE_FIXTURE_ASSET);
    try {
      const runtime = createWorkletRuntime({ name: 'Braille word model self-test' });
      const roi = fixture.roi;
      const result = await runOnRuntimeAsync(runtime, (source: Image) => {
        'worklet';
        return recognizeWordImage(source, model, thresholds, roi);
      }, image);
      if (result.status !== 'success' || result.text !== fixture.expectedText) {
        throw new Error('Pemeriksaan gambar awal pengenal kata gagal.');
      }
    } finally {
      image.dispose();
    }
    return model;
  } catch (error) {
    try { model.dispose(); } catch { /* Preserve the original validation error. */ }
    throw error;
  }
}

export function useWordModel(): WordModelState {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<LoadState>({ status: 'loading' });
  const canRetry = useRef(false);
  const retry = useCallback(() => {
    if (!canRetry.current) return;
    canRetry.current = false;
    setState({ status: 'loading' });
    setAttempt((value) => value + 1);
  }, []);

  useEffect(() => {
    let active = true;
    let lease: ReturnType<typeof createWordModelLease> | undefined;
    canRetry.current = false;
    void loadAndValidateWordModel().then((model) => {
      lease = createWordModelLease(model);
      if (!active) { lease.close(); return; }
      setState({ status: 'ready', model, modelVersion: WORD_MODEL_VERSION,
        thresholds: WORD_MODEL_MANIFEST.thresholds, acquireModel: lease.acquire });
    }).catch((error: unknown) => {
      if (!active) return;
      canRetry.current = true;
      setState({ status: 'error', error: error instanceof Error ? error : new Error('Pengenal kata gagal dimuat.') });
    });
    return () => { active = false; canRetry.current = false; lease?.close(); };
  }, [attempt]);
  return state.status === 'error' ? { ...state, retry } : state;
}
