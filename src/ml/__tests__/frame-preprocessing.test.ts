import {
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_LENGTH,
  BRAILLE_MODEL_INPUT_WIDTH,
} from '@/ml/braille-model-contract';
import {
  assertBrailleModelInputLength,
  convertRawPixelsToRgbTensor,
} from '@/ml/frame-preprocessing';

type RawPixels = Parameters<typeof convertRawPixelsToRgbTensor>[0];

function createRawPixels(
  bytes: ArrayLike<number>,
  width: number,
  height: number,
  pixelFormat: RawPixels['pixelFormat'],
): RawPixels {
  const buffer = new ArrayBuffer(bytes.length);
  new Uint8Array(buffer).set(bytes);

  return { buffer, width, height, pixelFormat };
}

const MODEL_PIXEL_COUNT =
  BRAILLE_MODEL_INPUT_WIDTH * BRAILLE_MODEL_INPUT_HEIGHT;

function repeatPixel(pixelBytes: readonly number[]): Uint8Array {
  const bytes = new Uint8Array(MODEL_PIXEL_COUNT * pixelBytes.length);
  for (let index = 0; index < MODEL_PIXEL_COUNT; index += 1) {
    bytes.set(pixelBytes, index * pixelBytes.length);
  }
  return bytes;
}

function createAlternatingRgbPixels(): Uint8Array {
  const bytes = new Uint8Array(BRAILLE_MODEL_INPUT_LENGTH);
  for (let index = 0; index < MODEL_PIXEL_COUNT; index += 1) {
    if (index % 2 === 1) {
      bytes.set([255, 255, 255], index * 3);
    }
  }
  return bytes;
}

describe('convertRawPixelsToRgbTensor', () => {
  it.each<readonly [RawPixels['pixelFormat'], readonly number[]]>([
    ['ARGB', [99, 11, 22, 33]],
    ['XRGB', [99, 11, 22, 33]],
    ['BGRA', [33, 22, 11, 99]],
    ['BGRX', [33, 22, 11, 99]],
    ['ABGR', [99, 33, 22, 11]],
    ['XBGR', [99, 33, 22, 11]],
    ['RGBA', [11, 22, 33, 99]],
    ['RGBX', [11, 22, 33, 99]],
    ['RGB', [11, 22, 33]],
    ['BGR', [33, 22, 11]],
  ])('mengubah format %s menjadi RGB interleaved', (pixelFormat, bytes) => {
    const result = convertRawPixelsToRgbTensor(
      createRawPixels(
        repeatPixel(bytes),
        BRAILLE_MODEL_INPUT_WIDTH,
        BRAILLE_MODEL_INPUT_HEIGHT,
        pixelFormat,
      ),
      'zero_to_255',
    );

    expect(result.tensor).toHaveLength(BRAILLE_MODEL_INPUT_LENGTH);
    expect(Array.from(result.tensor.subarray(0, 3))).toEqual([11, 22, 33]);
    expect(result.tensor.buffer).toBeInstanceOf(ArrayBuffer);
    expect(result.meanLuminance).toBeCloseTo(
      (0.2126 * 11 + 0.7152 * 22 + 0.0722 * 33) / 255,
      10,
    );
    expect(result.contrast).toBeCloseTo(0, 10);
  });

  it('menormalisasi RGB ke 0..1 dan menghitung statistik luminance', () => {
    const result = convertRawPixelsToRgbTensor(
      createRawPixels(
        createAlternatingRgbPixels(),
        BRAILLE_MODEL_INPUT_WIDTH,
        BRAILLE_MODEL_INPUT_HEIGHT,
        'RGB',
      ),
      'zero_to_one',
    );

    expect(result.tensor).toHaveLength(BRAILLE_MODEL_INPUT_LENGTH);
    expect(Array.from(result.tensor.subarray(0, 6))).toEqual([
      0, 0, 0, 1, 1, 1,
    ]);
    expect(result.meanLuminance).toBeCloseTo(0.5, 10);
    expect(result.contrast).toBeCloseTo(0.5, 10);
  });

  it.each([
    [0, 0],
    [BRAILLE_MODEL_INPUT_WIDTH - 1, BRAILLE_MODEL_INPUT_HEIGHT],
    [BRAILLE_MODEL_INPUT_WIDTH, BRAILLE_MODEL_INPUT_HEIGHT - 1],
    [28.5, BRAILLE_MODEL_INPUT_HEIGHT],
    [Number.NaN, BRAILLE_MODEL_INPUT_HEIGHT],
    [Number.MAX_SAFE_INTEGER + 1, BRAILLE_MODEL_INPUT_HEIGHT],
  ])('menolak dimensi native yang tidak aman atau bukan 28 x 28 (%p x %p)', (width, height) => {
    expect(() =>
      convertRawPixelsToRgbTensor(
        createRawPixels([], width, height, 'RGB'),
        'zero_to_one',
      ),
    ).toThrow('Dimensi pixel kamera harus tepat 28 x 28.');
  });

  it('menolak buffer yang lebih kecil daripada dimensi gambar', () => {
    expect(() =>
      convertRawPixelsToRgbTensor(
        createRawPixels(
          [1, 2],
          BRAILLE_MODEL_INPUT_WIDTH,
          BRAILLE_MODEL_INPUT_HEIGHT,
          'RGB',
        ),
        'zero_to_one',
      ),
    ).toThrow('Buffer pixel kamera terlalu kecil.');
  });

  it('menolak format pixel yang tidak didukung', () => {
    expect(() =>
      convertRawPixelsToRgbTensor(
        createRawPixels(
          [],
          BRAILLE_MODEL_INPUT_WIDTH,
          BRAILLE_MODEL_INPUT_HEIGHT,
          'unknown',
        ),
        'zero_to_one',
      ),
    ).toThrow('Format pixel kamera tidak didukung.');
  });

  it('menolak panjang tensor yang tidak sama dengan kontrak model', () => {
    expect(() =>
      assertBrailleModelInputLength(
        new Float32Array(BRAILLE_MODEL_INPUT_LENGTH - 1),
      ),
    ).toThrow('Panjang tensor input model tidak sesuai kontrak.');
    expect(() =>
      assertBrailleModelInputLength(
        new Float32Array(BRAILLE_MODEL_INPUT_LENGTH),
      ),
    ).not.toThrow();
  });
});
