import {
  BRAILLE_DECODER_VERSION,
  BRAILLE_LABELS,
  BRAILLE_MODEL_INPUT_CHANNELS,
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_LENGTH,
  BRAILLE_MODEL_INPUT_SCALE,
  BRAILLE_MODEL_INPUT_WIDTH,
  BRAILLE_MODEL_OUTPUT_LENGTH,
  BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS,
  BRAILLE_MODEL_RELEASE_VERIFIED,
  BRAILLE_MODEL_VERSION,
} from '@/ml/braille-model-contract';

describe('kontrak model Braille', () => {
  it('menjaga kontrak input NHWC 28x28 RGB tetap konsisten', () => {
    expect(BRAILLE_MODEL_INPUT_LENGTH).toBe(
      BRAILLE_MODEL_INPUT_WIDTH *
        BRAILLE_MODEL_INPUT_HEIGHT *
        BRAILLE_MODEL_INPUT_CHANNELS,
    );
    expect([
      BRAILLE_MODEL_INPUT_WIDTH,
      BRAILLE_MODEL_INPUT_HEIGHT,
      BRAILLE_MODEL_INPUT_CHANNELS,
    ]).toEqual([28, 28, 3]);
    expect(BRAILLE_MODEL_INPUT_SCALE).toBe('zero_to_one');
  });

  it('mengikat urutan 26 output ke alfabet Latin tanpa duplikat', () => {
    const expectedLabels = 'abcdefghijklmnopqrstuvwxyz'.split('');

    expect(BRAILLE_LABELS).toEqual(expectedLabels);
    expect(BRAILLE_LABELS).toHaveLength(BRAILLE_MODEL_OUTPUT_LENGTH);
    expect(new Set(BRAILLE_LABELS).size).toBe(BRAILLE_MODEL_OUTPUT_LENGTH);
    expect(Object.isFrozen(BRAILLE_LABELS)).toBe(true);
  });

  it('menyediakan versi model dan decoder yang eksplisit', () => {
    expect(BRAILLE_MODEL_VERSION).toMatch(/^\d+\.\d+\.\d+/u);
    expect(BRAILLE_DECODER_VERSION).toMatch(/labels/u);
  });

  it('menjaga runtime fail-closed selama manifest masih kandidat', () => {
    expect(BRAILLE_MODEL_RELEASE_VERIFIED).toBe(false);
    expect(BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS).toHaveLength(9);
    expect(Object.isFrozen(BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS)).toBe(true);
  });

  it('menolak manifest verified yang mengganti bukti wajib dengan placeholder', () => {
    const manifestModule =
      '../../../assets/models/braille-cell-classifier-v1/manifest.json';
    jest.resetModules();
    jest.doMock(manifestModule, () => ({
      artifact: { status: 'verified', version: '1.0.0' },
      verification: { placeholder: true },
    }));

    jest.isolateModules(() => {
      const contract = jest.requireActual('@/ml/braille-model-contract') as {
        BRAILLE_MODEL_RELEASE_VERIFIED: boolean;
      };
      expect(contract.BRAILLE_MODEL_RELEASE_VERIFIED).toBe(false);
    });

    jest.dontMock(manifestModule);
    jest.resetModules();
  });

  it('gagal cepat bila jumlah label aset tidak sama dengan output model', () => {
    const labelsModule =
      '../../../assets/models/braille-cell-classifier-v1/label.json';
    jest.resetModules();
    jest.doMock(labelsModule, () => ['a']);

    expect(() => {
      jest.isolateModules(() => {
        jest.requireActual('@/ml/braille-model-contract');
      });
    }).toThrow('Jumlah label Braille tidak sesuai dengan output model.');

    jest.dontMock(labelsModule);
    jest.resetModules();
  });

  it('gagal cepat bila urutan label aset berbeda dari kontrak a-z', () => {
    const labelsModule =
      '../../../assets/models/braille-cell-classifier-v1/label.json';
    jest.resetModules();
    jest.doMock(labelsModule, () =>
      'abcdefghijklmnopqrstuvwxyz'.split('').reverse(),
    );

    expect(() => {
      jest.isolateModules(() => {
        jest.requireActual('@/ml/braille-model-contract');
      });
    }).toThrow('Urutan label Braille tidak sesuai dengan kontrak decoder.');

    jest.dontMock(labelsModule);
    jest.resetModules();
  });
});
