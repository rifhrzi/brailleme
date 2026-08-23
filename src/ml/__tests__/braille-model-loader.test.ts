import { act, renderHook, waitFor } from '@testing-library/react-native';
import { Asset } from 'expo-asset';
import {
  loadTensorflowModel,
  type TensorflowModel,
} from 'react-native-fast-tflite';

import {
  BRAILLE_MODEL_INPUT_CHANNELS,
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_LENGTH,
  BRAILLE_MODEL_INPUT_WIDTH,
  BRAILLE_MODEL_OUTPUT_LENGTH,
} from '@/ml/braille-model-contract';
import {
  assertBrailleModelRuntimeAllowed,
  useBrailleModel,
} from '@/ml/braille-model-loader';

jest.mock('expo-asset', () => ({
  Asset: {
    loadAsync: jest.fn(),
  },
}));

jest.mock('react-native-fast-tflite', () => ({
  loadTensorflowModel: jest.fn(),
}));

jest.mock(
  '../../../assets/models/braille-cell-classifier-v1/braille_model.tflite',
  () => 101,
);

const loadAssetMock = Asset.loadAsync as jest.MockedFunction<
  typeof Asset.loadAsync
>;
const loadTensorflowModelMock =
  loadTensorflowModel as jest.MockedFunction<typeof loadTensorflowModel>;

type FakeModel = TensorflowModel & {
  readonly dispose: jest.Mock<void, []>;
  readonly run: jest.Mock<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>;
};

type TensorMetadata = TensorflowModel['inputs'][number];

const validInput: TensorMetadata = {
  name: 'conv2d_input',
  dataType: 'float32',
  shape: [
    1,
    BRAILLE_MODEL_INPUT_HEIGHT,
    BRAILLE_MODEL_INPUT_WIDTH,
    BRAILLE_MODEL_INPUT_CHANNELS,
  ],
};

const validOutput: TensorMetadata = {
  name: 'Identity',
  dataType: 'float32',
  shape: [1, BRAILLE_MODEL_OUTPUT_LENGTH],
};

function validProbabilityBuffer(): ArrayBuffer {
  const probabilities = new Float32Array(BRAILLE_MODEL_OUTPUT_LENGTH);
  probabilities.fill(1 / BRAILLE_MODEL_OUTPUT_LENGTH);
  return probabilities.buffer;
}

function probabilityBuffer(values: readonly number[]): ArrayBuffer {
  const probabilities = new Float32Array(BRAILLE_MODEL_OUTPUT_LENGTH);
  probabilities.set(values);
  return probabilities.buffer;
}

function createModel({
  inputs = [validInput],
  outputs = [validOutput],
  run = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(async () => [
    validProbabilityBuffer(),
  ]),
  dispose = jest.fn<void, []>(),
}: {
  readonly inputs?: TensorMetadata[];
  readonly outputs?: TensorMetadata[];
  readonly run?: jest.Mock<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>;
  readonly dispose?: jest.Mock<void, []>;
} = {}): FakeModel {
  return {
    inputs,
    outputs,
    run,
    runSync: jest.fn(),
    dispose,
  } as unknown as FakeModel;
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, reject, resolve };
}

async function expectErrorState(expectedMessage: string) {
  const hook = await renderHook(() => useBrailleModel());

  await waitFor(() => expect(hook.result.current.status).toBe('error'));
  const state = hook.result.current;
  if (state.status !== 'error') {
    throw new Error('Hook tidak masuk ke state error.');
  }
  expect(state.error.message).toBe(expectedMessage);
  await hook.unmount();
}

describe('useBrailleModel', () => {
  beforeEach(() => {
    loadAssetMock.mockResolvedValue([
      { localUri: 'file:///data/user/0/app/cache/braille_model.tflite' },
    ] as Awaited<ReturnType<typeof Asset.loadAsync>>);
  });

  it('memuat aset lokal, memvalidasi model, dan masuk ke state ready', async () => {
    const selfTest = deferred<ArrayBuffer[]>();
    const run = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(
      () => selfTest.promise,
    );
    const model = createModel({ run });
    loadTensorflowModelMock.mockResolvedValue(model);

    const hook = await renderHook(() => useBrailleModel());
    expect(hook.result.current).toEqual({ status: 'loading' });

    await waitFor(() => expect(run).toHaveBeenCalledTimes(1));
    const [inputBuffer] = run.mock.calls[0]?.[0] ?? [];
    expect(inputBuffer).toBeInstanceOf(ArrayBuffer);
    if (!inputBuffer) {
      throw new Error('Self-test tidak menerima tensor input.');
    }
    expect(inputBuffer.byteLength).toBe(
      BRAILLE_MODEL_INPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT,
    );
    expect(new Float32Array(inputBuffer).every((value) => value === 0)).toBe(
      true,
    );
    expect(loadTensorflowModelMock).toHaveBeenCalledWith(
      { url: 'file:///data/user/0/app/cache/braille_model.tflite' },
      [],
    );

    await act(async () => {
      selfTest.resolve([validProbabilityBuffer()]);
      await selfTest.promise;
    });
    await waitFor(() => expect(hook.result.current.status).toBe('ready'));
    expect(hook.result.current).toEqual({ status: 'ready', model });

    await hook.unmount();
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['tidak mengembalikan aset', []],
    ['tidak memiliki localUri', [{}]],
    ['mengembalikan URI non-lokal', [{ localUri: 'https://example.test/model.tflite' }]],
  ])('menolak aset yang %s', async (_caseName, assets) => {
    loadAssetMock.mockResolvedValue(
      assets as Awaited<ReturnType<typeof Asset.loadAsync>>,
    );

    await expectErrorState('Aset model tidak tersedia sebagai file lokal.');
    expect(loadTensorflowModelMock).not.toHaveBeenCalled();
  });

  it('meneruskan Error dari pemuatan native tanpa mencoba cleanup model yang tidak ada', async () => {
    loadTensorflowModelMock.mockRejectedValue(new Error('Runtime TFLite gagal.'));

    await expectErrorState('Runtime TFLite gagal.');
  });

  it('menormalkan rejection non-Error agar aman ditampilkan oleh UI', async () => {
    loadTensorflowModelMock.mockRejectedValue('native failure');

    await expectErrorState('Model gagal dimuat.');
  });

  it('retry dari error hanya memulai satu attempt baru dan tidak double-dispose', async () => {
    const invalidModel = createModel({ inputs: [] });
    const retrySelfTest = deferred<ArrayBuffer[]>();
    const retryRun = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(
      () => retrySelfTest.promise,
    );
    const recoveredModel = createModel({ run: retryRun });
    loadTensorflowModelMock
      .mockResolvedValueOnce(invalidModel)
      .mockResolvedValueOnce(recoveredModel);

    const hook = await renderHook(() => useBrailleModel());
    await waitFor(() => expect(hook.result.current.status).toBe('error'));
    const failedState = hook.result.current;
    if (failedState.status !== 'error') {
      throw new Error('Hook tidak masuk ke state error.');
    }

    await act(async () => {
      failedState.retry();
      failedState.retry();
    });

    await waitFor(() => expect(retryRun).toHaveBeenCalledTimes(1));
    expect(loadTensorflowModelMock).toHaveBeenCalledTimes(2);
    expect(invalidModel.dispose).toHaveBeenCalledTimes(1);

    await act(async () => {
      retrySelfTest.resolve([validProbabilityBuffer()]);
      await retrySelfTest.promise;
    });
    await waitFor(() => expect(hook.result.current.status).toBe('ready'));

    await hook.unmount();
    expect(invalidModel.dispose).toHaveBeenCalledTimes(1);
    expect(recoveredModel.dispose).toHaveBeenCalledTimes(1);
  });

  it('membuang model retry tepat sekali jika layar ditutup saat self-test', async () => {
    const retrySelfTest = deferred<ArrayBuffer[]>();
    const retryRun = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(
      () => retrySelfTest.promise,
    );
    const recoveredModel = createModel({ run: retryRun });
    loadTensorflowModelMock
      .mockRejectedValueOnce(new Error('Runtime sementara gagal.'))
      .mockResolvedValueOnce(recoveredModel);

    const hook = await renderHook(() => useBrailleModel());
    await waitFor(() => expect(hook.result.current.status).toBe('error'));
    const failedState = hook.result.current;
    if (failedState.status !== 'error') {
      throw new Error('Hook tidak masuk ke state error.');
    }

    await act(async () => failedState.retry());
    await waitFor(() => expect(retryRun).toHaveBeenCalledTimes(1));
    await hook.unmount();

    await act(async () => {
      retrySelfTest.resolve([validProbabilityBuffer()]);
      await retrySelfTest.promise;
    });
    await waitFor(() => expect(recoveredModel.dispose).toHaveBeenCalledTimes(1));
  });

  it.each([
    ['tanpa input', () => []],
    ['lebih dari satu input', () => [validInput, validInput]],
    [
      'dtype input salah',
      () => [{ ...validInput, dataType: 'uint8' as const }],
    ],
    ['shape input salah', () => [{ ...validInput, shape: [1, 28, 28, 1] }]],
  ])('menolak kontrak %s dan membuang model', async (_caseName, createInputs) => {
    const model = createModel({ inputs: createInputs() });
    loadTensorflowModelMock.mockResolvedValue(model);

    await expectErrorState('Kontrak input model tidak cocok.');
    expect(model.run).not.toHaveBeenCalled();
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['tanpa output', () => []],
    ['lebih dari satu output', () => [validOutput, validOutput]],
    [
      'dtype output salah',
      () => [{ ...validOutput, dataType: 'float16' as const }],
    ],
    ['shape output salah', () => [{ ...validOutput, shape: [26] }]],
  ])('menolak kontrak %s dan membuang model', async (_caseName, createOutputs) => {
    const model = createModel({ outputs: createOutputs() });
    loadTensorflowModelMock.mockResolvedValue(model);

    await expectErrorState('Kontrak output model tidak cocok.');
    expect(model.run).not.toHaveBeenCalled();
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('membuang model jika self-test native gagal', async () => {
    const run = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(() =>
      Promise.reject(new Error('Inference gagal.')),
    );
    const model = createModel({ run });
    loadTensorflowModelMock.mockResolvedValue(model);

    await expectErrorState('Inference gagal.');
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['tanpa tensor', [], 'Model tidak menghasilkan tensor output.'],
    [
      'byte length salah',
      [new ArrayBuffer(BRAILLE_MODEL_OUTPUT_LENGTH * 4 - 1)],
      'Ukuran tensor output model tidak cocok.',
    ],
    [
      'NaN',
      [probabilityBuffer([Number.NaN])],
      'Model menghasilkan probabilitas yang tidak valid.',
    ],
    [
      'Infinity',
      [probabilityBuffer([Number.POSITIVE_INFINITY])],
      'Model menghasilkan probabilitas yang tidak valid.',
    ],
    [
      'nilai negatif',
      [probabilityBuffer([-0.25, 1])],
      'Model menghasilkan probabilitas yang tidak valid.',
    ],
    [
      'nilai di atas satu',
      [probabilityBuffer([1.01])],
      'Model menghasilkan probabilitas yang tidak valid.',
    ],
    [
      'jumlah Softmax terlalu kecil',
      [probabilityBuffer([0.98])],
      'Self-test Softmax model gagal.',
    ],
    [
      'jumlah Softmax terlalu besar',
      [probabilityBuffer([0.6, 0.42])],
      'Self-test Softmax model gagal.',
    ],
  ])('menolak output self-test: %s', async (_caseName, output, message) => {
    const run = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(
      async () => output,
    );
    const model = createModel({ run });
    loadTensorflowModelMock.mockResolvedValue(model);

    await expectErrorState(message);
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('membuang model tepat sekali ketika komponen dilepas selama self-test', async () => {
    const selfTest = deferred<ArrayBuffer[]>();
    const run = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(
      () => selfTest.promise,
    );
    const model = createModel({ run });
    loadTensorflowModelMock.mockResolvedValue(model);

    const hook = await renderHook(() => useBrailleModel());
    await waitFor(() => expect(run).toHaveBeenCalledTimes(1));
    await hook.unmount();

    await act(async () => {
      selfTest.resolve([validProbabilityBuffer()]);
      await selfTest.promise;
    });
    await waitFor(() => expect(model.dispose).toHaveBeenCalledTimes(1));
  });

  it('mengabaikan error self-test yang tiba setelah komponen dilepas', async () => {
    const selfTest = deferred<ArrayBuffer[]>();
    const run = jest.fn<Promise<ArrayBuffer[]>, [ArrayBuffer[]]>(
      () => selfTest.promise,
    );
    const model = createModel({ run });
    loadTensorflowModelMock.mockResolvedValue(model);

    const hook = await renderHook(() => useBrailleModel());
    await waitFor(() => expect(run).toHaveBeenCalledTimes(1));
    await hook.unmount();

    await act(async () => {
      selfTest.reject(new Error('Inference selesai terlambat.'));
      await expect(selfTest.promise).rejects.toThrow(
        'Inference selesai terlambat.',
      );
    });
    await waitFor(() => expect(model.dispose).toHaveBeenCalledTimes(1));
  });

  it('mempertahankan error validasi bila native dispose ikut melempar', async () => {
    const dispose = jest.fn<void, []>(() => {
      throw new Error('Dispose native gagal.');
    });
    const model = createModel({ inputs: [], dispose });
    loadTensorflowModelMock.mockResolvedValue(model);

    await expectErrorState('Kontrak input model tidak cocok.');
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it('tidak melempar saat cleanup state ready gagal di native', async () => {
    const dispose = jest.fn<void, []>(() => {
      throw new Error('Dispose native gagal.');
    });
    const model = createModel({ dispose });
    loadTensorflowModelMock.mockResolvedValue(model);

    const hook = await renderHook(() => useBrailleModel());
    await waitFor(() => expect(hook.result.current.status).toBe('ready'));

    await expect(hook.unmount()).resolves.toBeUndefined();
    expect(dispose).toHaveBeenCalledTimes(1);
  });
});

describe('assertBrailleModelRuntimeAllowed', () => {
  it('menolak model kandidat pada build produksi', () => {
    expect(() => assertBrailleModelRuntimeAllowed(true, false)).toThrow(
      'Build produksi menolak classifier yang belum terverifikasi.',
    );
  });

  it.each([
    ['preview dengan model kandidat', false, false],
    ['produksi dengan model terverifikasi', true, true],
  ])('mengizinkan %s', (_caseName, isProduction, releaseVerified) => {
    expect(() =>
      assertBrailleModelRuntimeAllowed(isProduction, releaseVerified),
    ).not.toThrow();
  });

  it('mengizinkan model kandidat hanya dengan override produksi eksplisit', () => {
    expect(() =>
      assertBrailleModelRuntimeAllowed(true, false, true),
    ).not.toThrow();
  });
});
