import { act, renderHook, waitFor } from '@testing-library/react-native';
import { Asset } from 'expo-asset';
import { loadTensorflowModel } from 'react-native-fast-tflite';
import { loadImage } from 'react-native-nitro-image';
import { runOnRuntimeAsync } from 'react-native-worklets';

import { createWordModelLease, loadAndValidateWordModel, useWordModel } from '@/ml/word-model-loader';
import { recognizeWordImage } from '@/ml/word-image-processing';
import { WORD_MODEL_OUTPUT_LENGTH } from '@/ml/word-model-contract';

jest.mock('@/config', () => ({ IS_PRODUCTION_BUILD: false }));
jest.mock('expo-asset', () => ({ Asset: { loadAsync: jest.fn() } }));
jest.mock('react-native-fast-tflite', () => ({ loadTensorflowModel: jest.fn() }));
jest.mock('react-native-nitro-image', () => ({ loadImage: jest.fn() }));
jest.mock('react-native-worklets', () => ({
  createWorkletRuntime: jest.fn(() => ({ name: 'test-runtime' })),
  runOnRuntimeAsync: jest.fn(async (_runtime: unknown, task: (...args: unknown[]) => unknown, ...args: unknown[]) => task(...args)),
}));
jest.mock('@/ml/word-image-processing', () => ({ recognizeWordImage: jest.fn() }));
jest.mock('@/ml/word-model-assets', () => ({
  WORD_MODEL_ASSET: 501, WORD_NATIVE_FIXTURE_ASSET: 502, WORD_MODEL_BUNDLED: true,
  WORD_MODEL_RELEASE_VERIFIED: false, WORD_MODEL_VERSION: 'test-word-v1',
  WORD_MODEL_MANIFEST: {
    decoderVersion: jest.requireActual('@/ml/word-model-contract').WORD_DECODER_VERSION,
    thresholds: jest.requireActual('@/ml/word-model-contract').DEFAULT_WORD_MODEL_THRESHOLDS,
    nativeSelfTest: { expectedText: 'MAMA', roi: { x: 0, y: 0, width: 1, height: 1 } },
  },
}));

const assetMock = Asset.loadAsync as jest.Mock;
const modelMock = loadTensorflowModel as jest.Mock;
const imageMock = loadImage as jest.Mock;
const inferenceMock = recognizeWordImage as jest.Mock;

function makeModel() {
  return {
    inputs: [{ dataType: 'float32', shape: [1, 192, 480, 3] }],
    outputs: [{ dataType: 'float32', shape: [1, 48, 120, 11] }],
    run: jest.fn(async () => [new Float32Array(WORD_MODEL_OUTPUT_LENGTH).buffer]),
    runSync: jest.fn(), dispose: jest.fn(),
  };
}

describe('word model load and photo lifetime', () => {
  beforeEach(() => {
    assetMock.mockResolvedValue([{ localUri: 'file:///cache/word-reader.tflite' }]);
    modelMock.mockResolvedValue(makeModel());
    imageMock.mockResolvedValue({ dispose: jest.fn() });
    inferenceMock.mockReturnValue({ status: 'success', text: 'MAMA', boxes: [], cells: [], confidence: .99 });
  });

  it('verifies the native photo fixture after tensor warmup', async () => {
    const model = await loadAndValidateWordModel();
    expect(assetMock).toHaveBeenCalledWith(501);
    expect(modelMock).toHaveBeenCalledWith({ url: 'file:///cache/word-reader.tflite' }, []);
    expect(imageMock).toHaveBeenCalledWith(502);
    expect(inferenceMock).toHaveBeenCalledTimes(1);
    expect(model.dispose).not.toHaveBeenCalled();
  });

  it('blocks a bundled experimental model in production before loading assets', async () => {
    const config = jest.requireMock('@/config') as { IS_PRODUCTION_BUILD: boolean };
    config.IS_PRODUCTION_BUILD = true;
    try {
      await expect(loadAndValidateWordModel()).rejects.toThrow();
      expect(assetMock).not.toHaveBeenCalled();
      expect(modelMock).not.toHaveBeenCalled();
    } finally { config.IS_PRODUCTION_BUILD = false; }
  });

  it('rejects remote model locations before loading native inference', async () => {
    assetMock.mockResolvedValue([{ localUri: 'https://example.test/model' }]);
    await expect(loadAndValidateWordModel()).rejects.toThrow('lokal');
    expect(modelMock).not.toHaveBeenCalled();
  });

  it.each(['six-dot-word-v1', undefined])('rejects decoder version %s before native loading', async (version) => {
    const { WORD_MODEL_MANIFEST: manifest } = jest.requireMock('@/ml/word-model-assets') as {
      WORD_MODEL_MANIFEST: { decoderVersion: string | undefined };
    };
    const original = manifest.decoderVersion;
    manifest.decoderVersion = version;
    try {
      await expect(loadAndValidateWordModel()).rejects.toThrow('Konfigurasi pembacaan');
      expect(assetMock).not.toHaveBeenCalled();
      expect(modelMock).not.toHaveBeenCalled();
    } finally { manifest.decoderVersion = original; }
  });

  it('rejects the old single-cell contract and disposes its model', async () => {
    const model = makeModel();
    model.inputs[0]!.shape = [1, 28, 28, 3];
    modelMock.mockResolvedValue(model);
    await expect(loadAndValidateWordModel()).rejects.toThrow('tensor');
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('rejects nonfinite warmup output', async () => {
    const model = makeModel();
    const values = new Float32Array(WORD_MODEL_OUTPUT_LENGTH); values[8] = NaN;
    model.run.mockResolvedValue([values.buffer]); modelMock.mockResolvedValue(model);
    await expect(loadAndValidateWordModel()).rejects.toThrow('tensor');
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('does not become ready when native pixels yield the wrong word', async () => {
    const model = makeModel(); modelMock.mockResolvedValue(model);
    const fixture = { dispose: jest.fn() }; imageMock.mockResolvedValue(fixture);
    inferenceMock.mockReturnValue({ status: 'success', text: 'MA' });
    await expect(loadAndValidateWordModel()).rejects.toThrow('gambar awal');
    expect(fixture.dispose).toHaveBeenCalledTimes(1);
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('waits for the background self-test before releasing its source image', async () => {
    const fixture = { dispose: jest.fn() };
    imageMock.mockResolvedValue(fixture);
    let finish: ((value: unknown) => void) | undefined;
    (runOnRuntimeAsync as jest.Mock).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const pending = loadAndValidateWordModel();
    await waitFor(() => expect(finish).toBeDefined());
    expect(fixture.dispose).not.toHaveBeenCalled();
    finish!({ status: 'success', text: 'MAMA' });
    await pending;
    expect(fixture.dispose).toHaveBeenCalledTimes(1);
  });

  it('refuses an unavailable candidate before accessing native resources', async () => {
    const writable = jest.requireMock('@/ml/word-model-assets') as { WORD_MODEL_BUNDLED: boolean };
    writable.WORD_MODEL_BUNDLED = false;
    try {
      await expect(loadAndValidateWordModel()).rejects.toThrow('belum tersedia');
      expect(assetMock).not.toHaveBeenCalled();
    } finally { writable.WORD_MODEL_BUNDLED = true; }
  });

  it('waits for outstanding operations and releases exactly once', () => {
    const model = makeModel();
    const lease = createWordModelLease(model);
    const first = lease.acquire(), second = lease.acquire();
    lease.close(); first(); first();
    expect(model.dispose).not.toHaveBeenCalled();
    expect(() => lease.acquire()).toThrow('ditutup');
    second(); lease.close();
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('defers unmount disposal until background inference has settled', async () => {
    const model = makeModel(); modelMock.mockResolvedValue(model);
    const hook = await renderHook(() => useWordModel());
    await waitFor(() => expect(hook.result.current.status).toBe('ready'));
    const state = hook.result.current;
    if (state.status !== 'ready') throw new Error('Expected ready model');
    const release = state.acquireModel();
    await hook.unmount();
    expect(model.dispose).not.toHaveBeenCalled();
    release(); expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('recovers from load failure with an explicit retry', async () => {
    assetMock.mockRejectedValueOnce(new Error('offline asset failure'));
    const hook = await renderHook(() => useWordModel());
    await waitFor(() => expect(hook.result.current.status).toBe('error'));
    await act(() => {
      if (hook.result.current.status === 'error') hook.result.current.retry();
    });
    await waitFor(() => expect(hook.result.current.status).toBe('ready'));
    await hook.unmount();
  });
});
