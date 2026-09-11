import type { TfliteModel } from 'react-native-fast-tflite';
import type { Image } from 'react-native-nitro-image';
import { Platform } from 'react-native';

import { decodeWordOutput, detectWordProposals, wordBoxIou } from '@/ml/word-decoder';
import { getWordTrackingContext, groupWordProposals, type WordTrackingGroup } from '@/ml/word-tracking';
import {
  WORD_GUIDE_ASPECT_RATIO,
  WORD_GUIDE_MIN_ASPECT_RATIO,
  WORD_GUIDE_MAX_ASPECT_RATIO,
  WORD_GUIDE_WIDTH_FRACTION,
  WORD_MODEL_INPUT_HEIGHT,
  WORD_MODEL_INPUT_LENGTH,
  WORD_MODEL_INPUT_WIDTH,
  WORD_MODEL_CONTEXT_ASPECT_RATIO,
  type WordModelThresholds,
} from '@/ml/word-model-contract';
import type { NormalizedBox } from '@/types/live-scan';
import { isNormalizedWordBox, type WordImageResult, type WordRejectionReason } from '@/types/word-scan';

export const WORD_IMAGE_MIN_LUMINANCE = 0.06;
export const WORD_IMAGE_MAX_LUMINANCE = 0.985;
export const WORD_IMAGE_MIN_CONTRAST = 0.01;
// Nitro Image 0.15.1 labels Android ARGB_8888 bytes BGRA, although Bitmap emits RGBA.
// Verified using a colored native fixture; recheck this workaround when upgrading Nitro.
const ANDROID_NITRO_BITMAP_RGBA = Platform.OS === 'android';

export interface WordImageTensor {
  readonly tensor: Float32Array<ArrayBuffer>;
  readonly meanLuminance: number;
  readonly contrast: number;
}

export interface WordPixelProcessor {
  resizePixels(buffer: ArrayBuffer, width: number, height: number, pixelFormat: string, androidBitmap: boolean): {
    tensor: ArrayBuffer;
    meanLuminance: number;
    contrast: number;
  };
}

function assertDimensions(width: number, height: number): void {
  'worklet';
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error('Ukuran gambar atau pratinjau tidak valid.');
  }
}

/** Adjustable 1:1..8:1 word frame, with fixed visible height as its width changes. */
export function getWordGuideBox(previewWidth: number, previewHeight: number, aspect = WORD_GUIDE_ASPECT_RATIO): NormalizedBox {
  'worklet';
  assertDimensions(previewWidth, previewHeight);
  if (!Number.isFinite(aspect) || aspect < WORD_GUIDE_MIN_ASPECT_RATIO || aspect > WORD_GUIDE_MAX_ASPECT_RATIO) throw new Error('Lebar bingkai kata tidak valid.');
  const height = Math.min(previewWidth * WORD_GUIDE_WIDTH_FRACTION / WORD_GUIDE_MAX_ASPECT_RATIO, previewHeight * 0.3);
  const width = height * aspect;
  return { x: (previewWidth - width) / 2 / previewWidth, y: (previewHeight - height) / 2 / previewHeight, width: width / previewWidth, height: height / previewHeight };
}

/** Add context on either axis at 2.5:1 without stretching or trimming the selection. */
export function getWordContextBox(selection: NormalizedBox, imageWidth: number, imageHeight: number): NormalizedBox {
  'worklet';
  assertDimensions(imageWidth, imageHeight);
  if (!isNormalizedWordBox(selection)) throw new Error('Area pilihan kata tidak valid.');
  const width = Math.max(selection.width * imageWidth, selection.height * imageHeight * WORD_MODEL_CONTEXT_ASPECT_RATIO);
  const height = width / WORD_MODEL_CONTEXT_ASPECT_RATIO;
  if (width > imageWidth || height > imageHeight) {
    throw new Error('Konteks foto tidak cukup untuk area pilihan kata.');
  }
  const centerX = (selection.x + selection.width / 2) * imageWidth;
  const centerY = (selection.y + selection.height / 2) * imageHeight;
  const x = Math.max(0, Math.min(imageWidth - width, centerX - width / 2));
  const y = Math.max(0, Math.min(imageHeight - height, centerY - height / 2));
  return { x: x / imageWidth, y: y / imageHeight, width: width / imageWidth, height: height / imageHeight };
}

/** Map a centered cover preview to an UPRIGHT source photo (Photo.toImageAsync on Android). */
export function mapPreviewBoxToImage(
  box: NormalizedBox,
  previewWidth: number,
  previewHeight: number,
  imageWidth: number,
  imageHeight: number,
): NormalizedBox {
  'worklet';
  assertDimensions(previewWidth, previewHeight);
  assertDimensions(imageWidth, imageHeight);
  if (!isNormalizedWordBox(box)) throw new Error('Area panduan tidak valid.');
  const scale = Math.max(previewWidth / imageWidth, previewHeight / imageHeight);
  const offsetX = (imageWidth * scale - previewWidth) / 2;
  const offsetY = (imageHeight * scale - previewHeight) / 2;
  return {
    x: (box.x * previewWidth + offsetX) / scale / imageWidth,
    y: (box.y * previewHeight + offsetY) / scale / imageHeight,
    width: box.width * previewWidth / scale / imageWidth,
    height: box.height * previewHeight / scale / imageHeight,
  };
}

/** Inverse of mapPreviewBoxToImage; a box outside the visible cover can be outside 0..1. */
export function mapImageBoxToPreview(
  box: NormalizedBox,
  previewWidth: number,
  previewHeight: number,
  imageWidth: number,
  imageHeight: number,
): NormalizedBox {
  'worklet';
  assertDimensions(previewWidth, previewHeight);
  assertDimensions(imageWidth, imageHeight);
  if (!isNormalizedWordBox(box)) throw new Error('Kotak gambar tidak valid.');
  const scale = Math.max(previewWidth / imageWidth, previewHeight / imageHeight);
  const offsetX = (imageWidth * scale - previewWidth) / 2;
  const offsetY = (imageHeight * scale - previewHeight) / 2;
  return {
    x: (box.x * imageWidth * scale - offsetX) / previewWidth,
    y: (box.y * imageHeight * scale - offsetY) / previewHeight,
    width: box.width * imageWidth * scale / previewWidth,
    height: box.height * imageHeight * scale / previewHeight,
  };
}

/** For parity fixtures only; captured Android Images are already physically upright. */
export function rotateNormalizedWordBox(box: NormalizedBox, degrees: 0 | 90 | 180 | 270): NormalizedBox {
  'worklet';
  if (!isNormalizedWordBox(box)) throw new Error('Kotak rotasi tidak valid.');
  switch (degrees) {
    case 0: return { ...box };
    case 90: return { x: 1 - box.y - box.height, y: box.x, width: box.height, height: box.width };
    case 180: return { x: 1 - box.x - box.width, y: 1 - box.y - box.height, width: box.width, height: box.height };
    case 270: return { x: box.y, y: 1 - box.x - box.width, width: box.height, height: box.width };
    default: throw new Error('Rotasi gambar tidak valid.');
  }
}

/** Exact native RGB byte order -> float32 [0,1]; ImageNet rescaling belongs inside the model. */
export function convertWordRawPixels(raw: ReturnType<Image['toRawPixelData']>): WordImageTensor {
  'worklet';
  if (raw.width !== WORD_MODEL_INPUT_WIDTH || raw.height !== WORD_MODEL_INPUT_HEIGHT) throw new Error('Pixel model kata harus 480 x 192.');
  let stride: number;
  let red: number;
  let green: number;
  let blue: number;
  switch (raw.pixelFormat) {
    case 'ARGB': case 'XRGB': stride = 4; red = 1; green = 2; blue = 3; break;
    case 'BGRA': case 'BGRX': stride = 4; red = 2; green = 1; blue = 0; break;
    case 'ABGR': case 'XBGR': stride = 4; red = 3; green = 2; blue = 1; break;
    case 'RGBA': case 'RGBX': stride = 4; red = 0; green = 1; blue = 2; break;
    case 'RGB': stride = 3; red = 0; green = 1; blue = 2; break;
    case 'BGR': stride = 3; red = 2; green = 1; blue = 0; break;
    default: throw new Error('Format pixel foto tidak didukung.');
  }
  const source = new Uint8Array(raw.buffer);
  const pixels = raw.width * raw.height;
  if (source.length !== pixels * stride) throw new Error('Ukuran buffer pixel foto tidak sesuai.');
  const tensor = new Float32Array(new ArrayBuffer(WORD_MODEL_INPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT));
  let luminanceSum = 0;
  let luminanceSquaredSum = 0;
  for (let pixel = 0; pixel < pixels; pixel += 1) {
    const sourceIndex = pixel * stride;
    const targetIndex = pixel * 3;
    const r = source[sourceIndex + red]! / 255;
    const g = source[sourceIndex + green]! / 255;
    const b = source[sourceIndex + blue]! / 255;
    tensor[targetIndex] = r;
    tensor[targetIndex + 1] = g;
    tensor[targetIndex + 2] = b;
    const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    luminanceSum += luminance;
    luminanceSquaredSum += luminance * luminance;
  }
  const meanLuminance = luminanceSum / pixels;
  return { tensor, meanLuminance, contrast: Math.sqrt(Math.max(0, luminanceSquaredSum / pixels - meanLuminance * meanLuminance)) };
}

function bilinearWeights(inputSize: number, outputSize: number) {
  'worklet';
  const scale = inputSize / outputSize;
  const support = Math.max(1, scale);
  const inverseSupport = 1 / support;
  const kernelSize = Math.ceil(support) * 2 + 1;
  const starts = new Int32Array(outputSize);
  const counts = new Int32Array(outputSize);
  const weights = new Int32Array(outputSize * kernelSize);
  for (let target = 0; target < outputSize; target += 1) {
    const center = (target + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5));
    const end = Math.min(inputSize, Math.trunc(center + support + 0.5));
    let total = 0;
    for (let index = start; index < end; index += 1) total += Math.max(0, 1 - Math.abs((index - center + 0.5) * inverseSupport));
    starts[target] = start;
    counts[target] = end - start;
    for (let index = start; index < end; index += 1) {
      const weight = Math.max(0, 1 - Math.abs((index - center + 0.5) * inverseSupport)) / total;
      weights[target * kernelSize + index - start] = Math.floor(weight * 4194304 + 0.5);
    }
  }
  return { starts, counts, weights, kernelSize };
}

/** Deterministic RGB bilinear antialias resize matching Pillow's 8-bit two-pass contract.
 * Coefficients use 22 fractional bits and each pass rounds to 8 bits.
 * Reference: https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
 * androidNitroBitmap corrects the pinned Android bitmap metadata bug; generic BGRA remains BGRA.
 */
export function resizeWordRawPixels(
  raw: ReturnType<Image['toRawPixelData']>,
  androidNitroBitmap = false,
): WordImageTensor {
  'worklet';
  if (!Number.isSafeInteger(raw.width) || !Number.isSafeInteger(raw.height) || raw.width <= 0 || raw.height <= 0) throw new Error('Dimensi pixel foto tidak valid.');
  const format = androidNitroBitmap && raw.pixelFormat === 'BGRA' ? 'RGBA' : raw.pixelFormat;
  let stride: number, red: number, green: number, blue: number;
  switch (format) {
    case 'ARGB': case 'XRGB': stride = 4; red = 1; green = 2; blue = 3; break;
    case 'BGRA': case 'BGRX': stride = 4; red = 2; green = 1; blue = 0; break;
    case 'ABGR': case 'XBGR': stride = 4; red = 3; green = 2; blue = 1; break;
    case 'RGBA': case 'RGBX': stride = 4; red = 0; green = 1; blue = 2; break;
    case 'RGB': stride = 3; red = 0; green = 1; blue = 2; break;
    case 'BGR': stride = 3; red = 2; green = 1; blue = 0; break;
    default: throw new Error('Format pixel foto tidak didukung.');
  }
  const source = new Uint8Array(raw.buffer);
  if (source.length !== raw.width * raw.height * stride) throw new Error('Ukuran buffer pixel foto tidak sesuai.');
  const horizontal = bilinearWeights(raw.width, WORD_MODEL_INPUT_WIDTH);
  const vertical = bilinearWeights(raw.height, WORD_MODEL_INPUT_HEIGHT);
  const intermediate = new Uint8Array(WORD_MODEL_INPUT_WIDTH * raw.height * 3);
  for (let row = 0; row < raw.height; row += 1) {
    for (let column = 0; column < WORD_MODEL_INPUT_WIDTH; column += 1) {
      let r = 2097152, g = 2097152, b = 2097152;
      const start = (row * raw.width + horizontal.starts[column]!) * stride;
      const count = horizontal.counts[column]!;
      const kernel = column * horizontal.kernelSize;
      for (let tap = 0; tap < count; tap += 1) {
        const index = start + tap * stride;
        const weight = horizontal.weights[kernel + tap]!;
        r += source[index + red]! * weight;
        g += source[index + green]! * weight;
        b += source[index + blue]! * weight;
      }
      const target = (row * WORD_MODEL_INPUT_WIDTH + column) * 3;
      intermediate[target] = Math.min(255, Math.floor(r / 4194304));
      intermediate[target + 1] = Math.min(255, Math.floor(g / 4194304));
      intermediate[target + 2] = Math.min(255, Math.floor(b / 4194304));
    }
  }
  const resized = new Uint8Array(new ArrayBuffer(WORD_MODEL_INPUT_LENGTH));
  const rowStride = WORD_MODEL_INPUT_WIDTH * 3;
  for (let row = 0; row < WORD_MODEL_INPUT_HEIGHT; row += 1) {
    const start = vertical.starts[row]! * rowStride;
    const count = vertical.counts[row]!;
    const kernel = row * vertical.kernelSize;
    for (let column = 0; column < rowStride; column += 1) {
      let value = 2097152;
      for (let tap = 0; tap < count; tap += 1) value += intermediate[start + tap * rowStride + column]! * vertical.weights[kernel + tap]!;
      resized[row * rowStride + column] = Math.min(255, Math.floor(value / 4194304));
    }
  }
  return convertWordRawPixels({ width: WORD_MODEL_INPUT_WIDTH, height: WORD_MODEL_INPUT_HEIGHT, pixelFormat: 'RGB', buffer: resized.buffer });
}

/** These global exposure/contrast checks do not measure focus or certify Braille presence. */
export function assessWordImageQuality(image: Pick<WordImageTensor, 'meanLuminance' | 'contrast'>): WordRejectionReason | undefined {
  'worklet';
  if (!Number.isFinite(image.meanLuminance) || !Number.isFinite(image.contrast)) throw new Error('Statistik foto tidak valid.');
  if (image.meanLuminance < WORD_IMAGE_MIN_LUMINANCE) return 'dark';
  if (image.meanLuminance > WORD_IMAGE_MAX_LUMINANCE) return 'glare';
  if (image.contrast < WORD_IMAGE_MIN_CONTRAST) return 'low_confidence';
  return undefined;
}

function mapRoiBox(box: NormalizedBox, roi: NormalizedBox): NormalizedBox {
  'worklet';
  return { x: roi.x + box.x * roi.width, y: roi.y + box.y * roi.height, width: box.width * roi.width, height: box.height * roi.height };
}

function prepareWordImagePixels(raw: ReturnType<Image['toRawPixelData']>, pixelProcessor?: WordPixelProcessor): WordImageTensor {
  'worklet';
  if (!pixelProcessor) return resizeWordRawPixels(raw, ANDROID_NITRO_BITMAP_RGBA);
  const prepared = pixelProcessor.resizePixels(raw.buffer, raw.width, raw.height, raw.pixelFormat, ANDROID_NITRO_BITMAP_RGBA);
  if (prepared.tensor.byteLength !== WORD_MODEL_INPUT_LENGTH * Float32Array.BYTES_PER_ELEMENT ||
      !Number.isFinite(prepared.meanLuminance) || !Number.isFinite(prepared.contrast)) throw new Error('Preprocessing native menghasilkan tensor tidak valid.');
  return { tensor: new Float32Array(prepared.tensor), meanLuminance: prepared.meanLuminance, contrast: prepared.contrast };
}

/** roi is the visible guide. The model sees context; only the visible guide is decoded.
 * Call on a background worklet runtime. The caller owns and disposes sourceImage. */
export function recognizeWordImage(
  sourceImage: Image,
  model: Pick<TfliteModel, 'runSync'>,
  thresholds: WordModelThresholds,
  roi: NormalizedBox,
  pixelProcessor?: WordPixelProcessor,
): WordImageResult {
  'worklet';
  let cropped: Image | undefined;
  try {
    if (!Number.isSafeInteger(sourceImage.width) || !Number.isSafeInteger(sourceImage.height)) throw new Error('Dimensi foto tidak valid.');
    assertDimensions(sourceImage.width, sourceImage.height);
    if (!isNormalizedWordBox(roi)) throw new Error('Area foto tidak valid.');
    const context = getWordContextBox(roi, sourceImage.width, sourceImage.height);
    const startX = Math.max(0, Math.floor(context.x * sourceImage.width));
    const startY = Math.max(0, Math.floor(context.y * sourceImage.height));
    const endX = Math.min(sourceImage.width, Math.ceil((context.x + context.width) * sourceImage.width));
    const endY = Math.min(sourceImage.height, Math.ceil((context.y + context.height) * sourceImage.height));
    if (endX <= startX || endY <= startY) throw new Error('Area foto kosong.');
    const actualRoi = { x: startX / sourceImage.width, y: startY / sourceImage.height, width: (endX - startX) / sourceImage.width, height: (endY - startY) / sourceImage.height };
    const selection = {
      x: (roi.x - actualRoi.x) / actualRoi.width,
      y: (roi.y - actualRoi.y) / actualRoi.height,
      width: roi.width / actualRoi.width,
      height: roi.height / actualRoi.height,
    };
    // Bitmap.createBitmap may alias its source when the crop covers the whole image.
    // Preserve source ownership instead of creating a second wrapper that recycles it.
    cropped = startX === 0 && startY === 0 && endX === sourceImage.width && endY === sourceImage.height
      ? sourceImage
      : sourceImage.crop(startX, startY, endX, endY);
    const prepared = prepareWordImagePixels(cropped.toRawPixelData(), pixelProcessor);
    const quality = assessWordImageQuality(prepared);
    if (quality) return { status: 'rejected', reason: quality, cells: [], boxes: [] };
    const outputs = model.runSync([prepared.tensor.buffer]);
    if (outputs.length !== 1 || !outputs[0]) throw new Error('Output model kata tidak tersedia.');
    const result = decodeWordOutput(new Float32Array(outputs[0]), thresholds, selection);
    if (result.status === 'success') {
      const cells = result.cells.map((cell) => ({ ...cell, box: mapRoiBox(cell.box, actualRoi) }));
      return { ...result, cells, boxes: cells.map(({ box }) => box) };
    }
    if (result.status === 'error') return result;
    return { ...result, boxes: result.boxes.map((box) => mapRoiBox(box, actualRoi)) };
  } catch {
    return { status: 'error', reason: 'processing_error', cells: [], boxes: [] };
  } finally {
    if (cropped && cropped !== sourceImage) { try { cropped.dispose(); } catch {} }
  }
}

export interface WordImageDetection {
  readonly status: 'success' | 'rejected' | 'error';
  readonly reason?: WordRejectionReason | 'processing_error';
  readonly groups: readonly WordTrackingGroup[];
  readonly context: NormalizedBox;
}

/** Crop only an undistorted context; return its actual pixel-rounded source coordinates. */
function inferWordContext(
  sourceImage: Image,
  model: Pick<TfliteModel, 'runSync'>,
  context: NormalizedBox,
  pixelProcessor?: WordPixelProcessor,
): { context: NormalizedBox; output?: Float32Array; reason?: WordRejectionReason } {
  'worklet';
  if (!isNormalizedWordBox(context)) throw new Error('Konteks pelacakan tidak valid.');
  assertDimensions(sourceImage.width, sourceImage.height);
  const startX = Math.max(0, Math.floor(context.x * sourceImage.width));
  const startY = Math.max(0, Math.floor(context.y * sourceImage.height));
  const endX = Math.min(sourceImage.width, Math.ceil((context.x + context.width) * sourceImage.width));
  const endY = Math.min(sourceImage.height, Math.ceil((context.y + context.height) * sourceImage.height));
  if (endX <= startX || endY <= startY) throw new Error('Konteks pelacakan kosong.');
  const actual = { x: startX / sourceImage.width, y: startY / sourceImage.height, width: (endX - startX) / sourceImage.width, height: (endY - startY) / sourceImage.height };
  let cropped: Image | undefined;
  try {
    cropped = startX === 0 && startY === 0 && endX === sourceImage.width && endY === sourceImage.height
      ? sourceImage : sourceImage.crop(startX, startY, endX, endY);
    const prepared = prepareWordImagePixels(cropped.toRawPixelData(), pixelProcessor);
    const reason = assessWordImageQuality(prepared);
    if (reason) return { context: actual, reason };
    const outputs = model.runSync([prepared.tensor.buffer]);
    if (outputs.length !== 1 || !outputs[0]) throw new Error('Output model kata tidak tersedia.');
    return { context: actual, output: new Float32Array(outputs[0]) };
  } finally {
    if (cropped && cropped !== sourceImage) { try { cropped.dispose(); } catch {} }
  }
}

/** Localization only: never require supported text or drop control/low-confidence cells. */
export function detectWordImageCells(
  sourceImage: Image,
  model: Pick<TfliteModel, 'runSync'>,
  thresholds: WordModelThresholds,
  context: NormalizedBox,
  pixelProcessor?: WordPixelProcessor,
): WordImageDetection {
  'worklet';
  try {
    const inference = inferWordContext(sourceImage, model, context, pixelProcessor);
    if (!inference.output) return { status: 'rejected', reason: inference.reason ?? 'low_confidence', groups: [], context: inference.context };
    const proposals = detectWordProposals(inference.output, thresholds).map((proposal) => ({ ...proposal, box: mapRoiBox(proposal.box, inference.context) }));
    const groups = groupWordProposals(proposals, sourceImage.width, sourceImage.height, thresholds, inference.context);
    return { status: groups.length ? 'success' : 'rejected', ...(groups.length ? {} : { reason: 'no_grid' as const }), groups, context: inference.context };
  } catch {
    return { status: 'error', reason: 'processing_error', groups: [], context };
  }
}

/** Re-localize the photographed target in context before deciding which complete cells to decode.
 * At most three inferences (initial plus two expansions); the tracker union never trims indicators. */
export function recognizeTrackedWordImage(
  sourceImage: Image,
  model: Pick<TfliteModel, 'runSync'>,
  thresholds: WordModelThresholds,
  targetRoi: NormalizedBox,
  pixelProcessor?: WordPixelProcessor,
): WordImageResult {
  'worklet';
  let latestBoxes: readonly NormalizedBox[] = [];
  try {
    if (!isNormalizedWordBox(targetRoi)) throw new Error('Target foto tidak valid.');
    let group: WordTrackingGroup = { box: targetRoi, boxes: [targetRoi], score: 0, complete: false };
    let previousContext: NormalizedBox | undefined;
    for (let pass = 0; pass < 3; pass += 1) {
      const context = getWordTrackingContext(group, sourceImage.width, sourceImage.height, pass === 0 ? 1 : 1.6);
      if (previousContext && wordBoxIou(previousContext, context) > 0.9999) break;
      previousContext = context;
      const inference = inferWordContext(sourceImage, model, context, pixelProcessor);
      if (!inference.output) return { status: 'rejected', reason: inference.reason ?? 'low_confidence', cells: [], boxes: [] };
      const proposals = detectWordProposals(inference.output, thresholds);
      const mapped = proposals.map((proposal) => ({ ...proposal, box: mapRoiBox(proposal.box, inference.context) }));
      const groups = groupWordProposals(mapped, sourceImage.width, sourceImage.height, thresholds, inference.context);
      const candidates = groups.filter((candidate) => {
        // A complete word can be much wider than the initially tracked fragment.
        // Anchor coverage preserves identity without preferring a short high-IoU fragment.
        const width = Math.max(0, Math.min(candidate.box.x + candidate.box.width, targetRoi.x + targetRoi.width) - Math.max(candidate.box.x, targetRoi.x));
        const height = Math.max(0, Math.min(candidate.box.y + candidate.box.height, targetRoi.y + targetRoi.height) - Math.max(candidate.box.y, targetRoi.y));
        return width * height / (targetRoi.width * targetRoi.height) >= 0.5;
      })
        .sort((a, b) => wordBoxIou(b.box, targetRoi) - wordBoxIou(a.box, targetRoi));
      if (!candidates.length) return { status: 'rejected', reason: 'no_grid', cells: [], boxes: [] };
      group = candidates[0]!;
      latestBoxes = group.boxes;
      // Several overlapping groups imply an ambiguous row, not permission to select a fragment.
      if (candidates.length > 1) return { status: 'rejected', reason: 'multiple_rows', cells: [], boxes: group.boxes };
      const heights = group.boxes.map((box) => box.height).sort((a, b) => a - b);
      const medianHeight = heights[Math.floor(heights.length / 2)]!;
      const endMargin = medianHeight * sourceImage.height / sourceImage.width;
      const clearEnds = group.box.x - inference.context.x >= endMargin &&
        inference.context.x + inference.context.width - group.box.x - group.box.width >= endMargin;
      if (!group.complete || !clearEnds) continue;
      // Include every retained physical cell in this group with a small numerical margin.
      // Grouping, not rendered text length or the live rectangle, defines this selection.
      const epsilonX = 1 / sourceImage.width;
      const epsilonY = 1 / sourceImage.height;
      const left = Math.max(inference.context.x, group.box.x - epsilonX);
      const top = Math.max(inference.context.y, group.box.y - epsilonY);
      const right = Math.min(inference.context.x + inference.context.width, group.box.x + group.box.width + epsilonX);
      const bottom = Math.min(inference.context.y + inference.context.height, group.box.y + group.box.height + epsilonY);
      const selection = { x: (left - inference.context.x) / inference.context.width, y: (top - inference.context.y) / inference.context.height, width: (right - left) / inference.context.width, height: (bottom - top) / inference.context.height };
      const result = decodeWordOutput(inference.output, thresholds, selection);
      if (result.status === 'success') {
        const cells = result.cells.map((cell) => ({ ...cell, box: mapRoiBox(cell.box, inference.context) }));
        return { ...result, cells, boxes: cells.map(({ box }) => box) };
      }
      if (result.status === 'error') return result;
      return { ...result, boxes: result.boxes.map((box) => mapRoiBox(box, inference.context)) };
    }
    return { status: 'rejected', reason: 'clipped', cells: [], boxes: latestBoxes };
  } catch {
    return { status: 'error', reason: 'processing_error', cells: [], boxes: [] };
  }
}
