import type { Image } from 'react-native-nitro-image';

import {
  BRAILLE_MODEL_INPUT_HEIGHT,
  BRAILLE_MODEL_INPUT_LENGTH,
  BRAILLE_MODEL_INPUT_WIDTH,
  type BrailleInputScale,
} from '@/ml/braille-model-contract';

export type PreprocessedFrame = {
  readonly tensor: Float32Array<ArrayBuffer>;
  readonly meanLuminance: number;
  readonly contrast: number;
};

type RawPixelData = ReturnType<Image['toRawPixelData']>;

export const BRAILLE_CENTER_CROP_FRACTION = 0.55;

export type PreparedBrailleImage = {
  readonly croppedImage: Image;
  readonly resizedImage: Image;
};

/** Memakai crop tengah yang sama untuk frame kamera dan self-test native. */
export function centerCropAndResizeBrailleImage(
  sourceImage: Image,
): PreparedBrailleImage {
  'worklet';

  const cropSize = Math.max(
    1,
    Math.floor(
      Math.min(sourceImage.width, sourceImage.height) *
        BRAILLE_CENTER_CROP_FRACTION,
    ),
  );
  const startX = Math.floor((sourceImage.width - cropSize) / 2);
  const startY = Math.floor((sourceImage.height - cropSize) / 2);
  const croppedImage = sourceImage.crop(
    startX,
    startY,
    startX + cropSize,
    startY + cropSize,
  );

  try {
    return {
      croppedImage,
      resizedImage: croppedImage.resize(
        BRAILLE_MODEL_INPUT_WIDTH,
        BRAILLE_MODEL_INPUT_HEIGHT,
      ),
    };
  } catch (error) {
    try {
      croppedImage.dispose();
    } catch {}
    throw error;
  }
}

/** Mengubah pixel native menjadi tensor RGB interleaved (NHWC). */
export function convertRawPixelsToRgbTensor(
  raw: RawPixelData,
  inputScale: BrailleInputScale,
): PreprocessedFrame {
  'worklet';

  if (
    !Number.isSafeInteger(raw.width) ||
    !Number.isSafeInteger(raw.height) ||
    raw.width !== BRAILLE_MODEL_INPUT_WIDTH ||
    raw.height !== BRAILLE_MODEL_INPUT_HEIGHT
  ) {
    throw new Error('Dimensi pixel kamera harus tepat 28 x 28.');
  }

  let stride: number;
  let redOffset: number;
  let greenOffset: number;
  let blueOffset: number;

  switch (raw.pixelFormat) {
    case 'ARGB':
    case 'XRGB':
      stride = 4;
      redOffset = 1;
      greenOffset = 2;
      blueOffset = 3;
      break;
    case 'BGRA':
    case 'BGRX':
      stride = 4;
      redOffset = 2;
      greenOffset = 1;
      blueOffset = 0;
      break;
    case 'ABGR':
    case 'XBGR':
      stride = 4;
      redOffset = 3;
      greenOffset = 2;
      blueOffset = 1;
      break;
    case 'RGBA':
    case 'RGBX':
      stride = 4;
      redOffset = 0;
      greenOffset = 1;
      blueOffset = 2;
      break;
    case 'RGB':
      stride = 3;
      redOffset = 0;
      greenOffset = 1;
      blueOffset = 2;
      break;
    case 'BGR':
      stride = 3;
      redOffset = 2;
      greenOffset = 1;
      blueOffset = 0;
      break;
    default:
      throw new Error('Format pixel kamera tidak didukung.');
  }

  const pixels = BRAILLE_MODEL_INPUT_WIDTH * BRAILLE_MODEL_INPUT_HEIGHT;
  const source = new Uint8Array(raw.buffer);
  if (source.byteLength < pixels * stride) {
    throw new Error('Buffer pixel kamera terlalu kecil.');
  }

  const tensor = new Float32Array(
    new ArrayBuffer(
      BRAILLE_MODEL_INPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT,
    ),
  );
  const multiplier = inputScale === 'zero_to_one' ? 1 / 255 : 1;
  let luminanceSum = 0;
  let luminanceSquaredSum = 0;

  for (let pixelIndex = 0; pixelIndex < pixels; pixelIndex += 1) {
    const sourceIndex = pixelIndex * stride;
    const red = source[sourceIndex + redOffset] ?? 0;
    const green = source[sourceIndex + greenOffset] ?? 0;
    const blue = source[sourceIndex + blueOffset] ?? 0;
    const targetIndex = pixelIndex * 3;

    tensor[targetIndex] = red * multiplier;
    tensor[targetIndex + 1] = green * multiplier;
    tensor[targetIndex + 2] = blue * multiplier;

    const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;
    luminanceSum += luminance;
    luminanceSquaredSum += luminance * luminance;
  }

  const meanLuminance = pixels > 0 ? luminanceSum / pixels : 0;
  const variance =
    pixels > 0
      ? Math.max(0, luminanceSquaredSum / pixels - meanLuminance * meanLuminance)
      : 0;

  return {
    tensor,
    meanLuminance,
    contrast: Math.sqrt(variance),
  };
}

/** Menjaga panjang tensor tetap sesuai kontrak sebelum melewati batas native. */
export function assertBrailleModelInputLength(tensor: Float32Array): void {
  'worklet';

  if (tensor.length !== BRAILLE_MODEL_INPUT_LENGTH) {
    throw new Error('Panjang tensor input model tidak sesuai kontrak.');
  }
}
