import type { Image } from 'react-native-nitro-image';

import {
  assessWordImageQuality, convertWordRawPixels, getWordGuideBox, getWordContextBox,
  mapImageBoxToPreview, mapPreviewBoxToImage, recognizeWordImage, rotateNormalizedWordBox,
} from '@/ml/word-image-processing';
import { DEFAULT_WORD_MODEL_THRESHOLDS, WORD_MODEL_INPUT_LENGTH, WORD_MODEL_OUTPUT_LENGTH } from '@/ml/word-model-contract';

const pixelCount = 480 * 192;
function rawPixels(pixelFormat: Parameters<typeof convertWordRawPixels>[0]['pixelFormat'] = 'RGB', pixel?: readonly number[]) {
  const stride = pixel?.length ?? 3;
  const bytes = new Uint8Array(pixelCount * stride);
  for (let index = 0; index < pixelCount; index += 1) bytes.set(pixel ?? (index % 2 ? [192, 192, 192] : [64, 64, 64]), index * stride);
  return { width: 480, height: 192, pixelFormat, buffer: bytes.buffer };
}

function fixture() {
  const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
  for (let index = 0; index < 4; index += 1) {
    const mask = index % 2 ? 1 : 13;
    const column = 24 + index * 12;
    const offset = (24 * 120 + column) * 11;
    output.set([0.98, 0, 0, 0.07, 0.3], offset);
    for (let dot = 0; dot < 6; dot += 1) output[offset + 5 + dot] = mask & (1 << dot) ? 0.99 : 0.01;
  }
  const cropped = { dispose: jest.fn(), toRawPixelData: jest.fn(() => rawPixels()), resize: jest.fn() };
  const source = { width: 1440, height: 1920, dispose: jest.fn(), crop: jest.fn(() => cropped) };
  const model = { runSync: jest.fn((_input: ArrayBuffer[]) => [output.buffer]) };
  return { source, cropped, model };
}

describe('word photo geometry', () => {
  it.each([[360, 360, 1440, 1920], [360, 640, 1440, 1920], [640, 360, 1920, 1440], [360, 360, 1920, 1440]])(
    'maps and round-trips a guide through cover preview %sx%s and upright photo %sx%s', (pw, ph, iw, ih) => {
      const guide = getWordGuideBox(pw, ph);
      expect(guide.width * pw / (guide.height * ph)).toBeCloseTo(4);
      const roi = mapPreviewBoxToImage(guide, pw, ph, iw, ih);
      expect(roi.width * iw / (roi.height * ih)).toBeCloseTo(4);
      const back = mapImageBoxToPreview(roi, pw, ph, iw, ih);
      for (const key of ['x', 'y', 'width', 'height'] as const) expect(back[key]).toBeCloseTo(guide[key]);
    },
  );

  it('accounts for vertical source pixels cropped off in a square portrait preview', () => {
    const mapped = mapPreviewBoxToImage(getWordGuideBox(360, 360, 8), 360, 360, 1440, 1920);
    expect(mapped.x).toBeCloseTo(0.05);
    expect(mapped.y).toBeCloseTo(0.4578125);
    expect(mapped.width).toBeCloseTo(0.9);
    expect(mapped.height).toBeCloseTo(0.084375);
  });

  it('adds 2.5:1 vertical context while preserving an 8:1 visible word row', () => {
    const selection = mapPreviewBoxToImage(getWordGuideBox(360, 360, 8), 360, 360, 1440, 1920);
    const context = getWordContextBox(selection, 1440, 1920);
    expect(context.x).toBeCloseTo(selection.x);
    expect(context.width).toBeCloseTo(selection.width);
    expect(context.y).toBeCloseTo(0.365);
    expect(context.height).toBeCloseTo(0.27);
    expect((selection.y - context.y) / context.height).toBeCloseTo(0.34375);
    expect(selection.height / context.height).toBeCloseTo(0.3125);
  });

  it('keeps guide height fixed while selecting short and long words', () => {
    const short = getWordGuideBox(360, 360, 1);
    const middle = getWordGuideBox(360, 360);
    const long = getWordGuideBox(360, 360, 8);
    expect(short.height).toBeCloseTo(0.1125);
    expect(middle.height).toBe(short.height);
    expect(long.height).toBe(short.height);
    expect(short.width).toBeCloseTo(0.1125);
    expect(middle.width).toBeCloseTo(0.45);
    expect(long.width).toBeCloseTo(0.9);
    expect(getWordGuideBox(1000, 100, 8).height).toBeCloseTo(0.3);
    for (const aspect of [0.5, 9, Number.NaN]) expect(() => getWordGuideBox(360, 360, aspect)).toThrow('bingkai');
  });

  it.each([1, 2.5, 4, 8])('preserves the selected %s:1 geometry inside a 2.5:1 context without stretching', (aspect) => {
    const selection = mapPreviewBoxToImage(getWordGuideBox(360, 360, aspect), 360, 360, 1440, 1920);
    const context = getWordContextBox(selection, 1440, 1920);
    expect(context.width * 1440 / (context.height * 1920)).toBeCloseTo(2.5);
    expect(selection.x + 0.000001).toBeGreaterThanOrEqual(context.x);
    expect(selection.y + 0.000001).toBeGreaterThanOrEqual(context.y);
    expect(selection.x + selection.width).toBeLessThanOrEqual(context.x + context.width + 0.000001);
    expect(selection.y + selection.height).toBeLessThanOrEqual(context.y + context.height + 0.000001);
    if (aspect === 1) {
      expect(context.width).toBeCloseTo(0.28125);
      expect(context.height).toBeCloseTo(selection.height);
      expect(selection.width / context.width).toBeCloseTo(0.4);
    }
  });

  it('shifts context at image edges without cropping the visible selection', () => {
    const top = getWordContextBox({ x: 0.1, y: 0.01, width: 0.8, height: 0.075 }, 1440, 1920);
    expect(top.y).toBe(0);
    expect(top.height).toBeCloseTo(0.24);
    const bottom = getWordContextBox({ x: 0.1, y: 0.9, width: 0.8, height: 0.075 }, 1440, 1920);
    expect(bottom.y).toBeCloseTo(0.76);
    expect(bottom.y + bottom.height).toBeCloseTo(1);
    const left = getWordContextBox({ x: 0.01, y: 0.4, width: 0.05, height: 0.15 }, 1440, 1920);
    expect(left.x).toBe(0);
    expect(left.width).toBeCloseTo(0.5);
    const right = getWordContextBox({ x: 0.94, y: 0.4, width: 0.05, height: 0.15 }, 1440, 1920);
    expect(right.x).toBeCloseTo(0.5);
    expect(() => getWordContextBox({ x: 0, y: 0, width: 1, height: 1 }, 1440, 1920)).toThrow('Konteks');
    expect(() => getWordContextBox({ x: -1, y: 0, width: 1, height: 1 }, 1440, 1920)).toThrow('pilihan');
  });

  it.each([0, 90, 180, 270] as const)('round-trips normalized orientation %s without swapping dot order', (degrees) => {
    const box = { x: 0.1, y: 0.2, width: 0.3, height: 0.4 };
    const rotated = rotateNormalizedWordBox(box, degrees);
    const back = rotateNormalizedWordBox(rotated, ((360 - degrees) % 360) as 0 | 90 | 180 | 270);
    for (const key of ['x', 'y', 'width', 'height'] as const) expect(back[key]).toBeCloseTo(box[key]);
    if (degrees === 90) expect(rotated).toMatchObject({ y: 0.1, width: 0.4, height: 0.3 });
  });

  it('rejects invalid dimensions and boxes before accessing native images', () => {
    expect(() => getWordGuideBox(0, 360)).toThrow('Ukuran');
    expect(() => mapPreviewBoxToImage({ x: -1, y: 0, width: 1, height: 1 }, 360, 360, 1920, 1440)).toThrow('panduan');
    expect(() => mapImageBoxToPreview({ x: 0, y: 0, width: 0, height: 1 }, 360, 360, 1920, 1440)).toThrow('gambar');
    expect(() => rotateNormalizedWordBox({ x: 0, y: 0, width: 0, height: 1 }, 90)).toThrow('rotasi');
  });
});

describe('word photo pixel contract', () => {
  it.each([
    ['ARGB', [99, 11, 22, 33]], ['XRGB', [99, 11, 22, 33]],
    ['BGRA', [33, 22, 11, 99]], ['BGRX', [33, 22, 11, 99]],
    ['ABGR', [99, 33, 22, 11]], ['XBGR', [99, 33, 22, 11]],
    ['RGBA', [11, 22, 33, 99]], ['RGBX', [11, 22, 33, 99]],
    ['RGB', [11, 22, 33]], ['BGR', [33, 22, 11]],
  ] as const)('converts %s byte ordering to float32 RGB [0,1]', (format, pixel) => {
    const prepared = convertWordRawPixels(rawPixels(format, pixel));
    expect(prepared.tensor).toHaveLength(WORD_MODEL_INPUT_LENGTH);
    [11, 22, 33].forEach((value, channel) => expect(prepared.tensor[channel]).toBeCloseTo(value / 255));
    expect(prepared.meanLuminance).toBeCloseTo((0.2126 * 11 + 0.7152 * 22 + 0.0722 * 33) / 255);
    expect(prepared.contrast).toBeCloseTo(0);
  });

  it('rejects invalid shape, unsupported format and padded or truncated pixel buffers', () => {
    expect(() => convertWordRawPixels({ ...rawPixels(), width: 28 })).toThrow('480 x 192');
    expect(() => convertWordRawPixels(rawPixels('unknown'))).toThrow('Format');
    expect(() => convertWordRawPixels({ ...rawPixels(), buffer: new ArrayBuffer(1) })).toThrow('buffer');
    expect(() => convertWordRawPixels({ ...rawPixels(), buffer: new ArrayBuffer(WORD_MODEL_INPUT_LENGTH + 1) })).toThrow('buffer');
  });

  it('checks exposure and low contrast without claiming those prove focus', () => {
    expect(assessWordImageQuality({ meanLuminance: 0.01, contrast: 0.1 })).toBe('dark');
    expect(assessWordImageQuality({ meanLuminance: 0.99, contrast: 0.1 })).toBe('glare');
    expect(assessWordImageQuality({ meanLuminance: 0.5, contrast: 0 })).toBe('low_confidence');
    expect(assessWordImageQuality({ meanLuminance: 0.5, contrast: 0.03 })).toBeUndefined();
    expect(() => assessWordImageQuality({ meanLuminance: Number.NaN, contrast: 0.1 })).toThrow('Statistik');
  });
});

describe('word photo inference and resource ownership', () => {
  const roi = { x: 0.1, y: 0.3, width: 0.8, height: 0.24 };

  it('recognizes a complete word in one 480x192 ROI and maps boxes to the actual rounded photo crop', () => {
    const { source, cropped, model } = fixture();
    const result = recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, roi);
    expect(result).toMatchObject({ status: 'success', text: 'mama' });
    expect(source.crop).toHaveBeenCalledWith(144, 576, 1296, 1037);
    expect(cropped.resize).not.toHaveBeenCalled();
    expect(model.runSync).toHaveBeenCalledTimes(1);
    expect(new Float32Array(model.runSync.mock.calls[0]![0]![0]!)).toHaveLength(WORD_MODEL_INPUT_LENGTH);
    expect(result.boxes[0]!.x).toBeCloseTo(0.1 + (0.2 - 0.07 / 2) * 0.8);
    expect(result.boxes[0]!.height).toBeCloseTo(0.3 * 461 / 1920);
    expect(cropped.dispose).toHaveBeenCalledTimes(1);
    expect(source.dispose).not.toHaveBeenCalled();
  });

  it('does not infer or expose text for an unusable image', () => {
    const { source, cropped, model } = fixture();
    cropped.toRawPixelData.mockReturnValue(rawPixels('RGB', [0, 0, 0]));
    expect(recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, roi)).toEqual({ status: 'rejected', reason: 'dark', cells: [], boxes: [] });
    expect(model.runSync).not.toHaveBeenCalled();
    expect(cropped.dispose).toHaveBeenCalled();
  });

  it('does not wrap or recycle a full-source crop that Android would alias', () => {
    const { model } = fixture();
    const source = { width: 480, height: 192, crop: jest.fn(), dispose: jest.fn(), toRawPixelData: jest.fn(() => rawPixels()) };
    expect(recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, { x: 0, y: 0, width: 1, height: 1 })).toMatchObject({ status: 'success', text: 'mama' });
    expect(source.crop).not.toHaveBeenCalled();
    expect(source.dispose).not.toHaveBeenCalled();
  });

  it.each(['crop', 'pixels', 'inference', 'output'] as const)('cleans every acquired child image after %s failure, preserving source ownership', (stage) => {
    const { source, cropped, model } = fixture();
    const fail = () => { throw new Error('native failed'); };
    if (stage === 'crop') source.crop.mockImplementation(fail);
    if (stage === 'pixels') cropped.toRawPixelData.mockImplementation(fail);
    if (stage === 'inference') model.runSync.mockImplementation(fail);
    if (stage === 'output') model.runSync.mockReturnValue([new Float32Array(26).buffer]);
    expect(recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, roi)).toEqual({ status: 'error', reason: 'processing_error', cells: [], boxes: [] });
    expect(source.dispose).not.toHaveBeenCalled();
    expect(cropped.dispose).toHaveBeenCalledTimes(stage === 'crop' ? 0 : 1);
  });

  it('preserves a successful result if crop disposal throws', () => {
    const { source, cropped, model } = fixture();
    cropped.dispose.mockImplementation(() => { throw new Error('dispose'); });
    expect(recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, roi).status).toBe('success');
    expect(cropped.dispose).toHaveBeenCalled();
  });

  it('maps rejected boxes but never supplies a shortened word', () => {
    const { source, model } = fixture();
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    output.set([0.4, 0, 0, 0.07, 0.3, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01], (24 * 120 + 24) * 11);
    model.runSync.mockReturnValue([output.buffer]);
    const result = recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, roi);
    expect(result).toMatchObject({ status: 'rejected', reason: 'low_confidence', cells: [] });
    expect(result).not.toHaveProperty('text');
    expect(result.boxes[0]!.x).toBeCloseTo(0.1 + 0.165 * 0.8);
  });

  it('infers the padded context once but returns only the chosen visible word', () => {
    const { source, cropped, model } = fixture();
    const selection = mapPreviewBoxToImage(getWordGuideBox(360, 360, 8), 360, 360, source.width, source.height);
    const result = recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, selection);
    expect(result).toMatchObject({ status: 'success', text: 'mama' });
    expect(cropped.resize).not.toHaveBeenCalled();
    expect(source.crop).toHaveBeenCalledWith(72, 700, 1368, 1220);
    for (const box of result.boxes) {
      expect(box.y).toBeGreaterThanOrEqual(selection.y);
      expect(box.y + box.height).toBeLessThanOrEqual(selection.y + selection.height);
    }
  });

  it('adds horizontal context for a short guide and excludes neighboring words', () => {
    const { source, model } = fixture();
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    for (const column of [12, 60, 108]) {
      output.set([0.98, 0, 0, 0.2, 0.8, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01], (24 * 120 + column) * 11);
    }
    model.runSync.mockReturnValue([output.buffer]);
    const selection = mapPreviewBoxToImage(getWordGuideBox(360, 360, 1), 360, 360, source.width, source.height);
    const result = recognizeWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, selection);
    expect(result).toMatchObject({ status: 'success', text: 'a' });
    expect(result.boxes).toHaveLength(1);
    expect(source.crop).toHaveBeenCalledWith(517, 879, 923, 1041);
    expect(result.boxes[0]!.x).toBeGreaterThan(selection.x);
    expect(result.boxes[0]!.x + result.boxes[0]!.width).toBeLessThan(selection.x + selection.width);
  });
});
