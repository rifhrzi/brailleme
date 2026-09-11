import type { Image } from 'react-native-nitro-image';
import type { NormalizedBox } from '@/types/live-scan';
import { detectWordImageCells, recognizeTrackedWordImage } from '@/ml/word-image-processing';
import { DEFAULT_WORD_MODEL_THRESHOLDS, WORD_MODEL_INPUT_LENGTH, WORD_MODEL_OUTPUT_LENGTH } from '@/ml/word-model-contract';

function scene(cells: readonly { mask: number; box: NormalizedBox; score?: number }[]) {
  let context = { x: 0, y: 0, width: 1, height: 1 };
  const crops: { dispose: jest.Mock }[] = [];
  const bytes = new Uint8Array(480 * 192 * 3);
  for (let index = 0; index < bytes.length; index += 1) bytes[index] = index % 2 ? 192 : 64;
  const toRawPixelData = jest.fn(() => ({ width: 480, height: 192, pixelFormat: 'RGB', buffer: bytes.buffer }));
  const source = {
    width: 1200, height: 1200, dispose: jest.fn(), toRawPixelData,
    crop: jest.fn((x: number, y: number, right: number, bottom: number) => {
      context = { x: x / 1200, y: y / 1200, width: (right - x) / 1200, height: (bottom - y) / 1200 };
      const crop = { dispose: jest.fn(), toRawPixelData };
      crops.push(crop);
      return crop;
    }),
  };
  const model = { runSync: jest.fn(() => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    for (const cell of cells) {
      const x = (cell.box.x + cell.box.width / 2 - context.x) / context.width;
      const y = (cell.box.y + cell.box.height / 2 - context.y) / context.height;
      if (x < 0 || x >= 1 || y < 0 || y >= 1) continue;
      const column = Math.floor(x * 120);
      const row = Math.floor(y * 48);
      const offset = (row * 120 + column) * 11;
      output.set([cell.score ?? 0.98, x * 120 - column, y * 48 - row, cell.box.width / context.width, cell.box.height / context.height], offset);
      for (let dot = 0; dot < 6; dot += 1) output[offset + 5 + dot] = cell.mask & (1 << dot) ? 0.99 : 0.01;
    }
    return [output.buffer];
  }) };
  return { source, model, crops };
}

describe('tracked photo re-localization', () => {
  it.each([[32, 'Ab'], [60, '12']])('recovers an indicator mask %s outside the tracked text rectangle', (mask, text) => {
    const cells = [mask as number, 1, 3].map((value, index) => ({ mask: value, box: { x: 0.35 + index * 0.075, y: 0.45, width: 0.05, height: 0.1 } }));
    const { source, model, crops } = scene(cells);
    const result = recognizeTrackedWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, { x: 0.425, y: 0.45, width: 0.125, height: 0.1 });
    expect(result).toMatchObject({ status: 'success', text });
    expect(result.boxes).toHaveLength(3);
    expect(result.boxes[0]?.x).toBeCloseTo(0.35);
    expect(source.dispose).not.toHaveBeenCalled();
    expect(crops.every((crop) => crop.dispose.mock.calls.length === 1)).toBe(true);
  });

  it('expands a short tracked fragment to the whole connected word within three inferences', () => {
    const cells = Array.from({ length: 9 }, (_, index) => ({ mask: index === 0 ? 32 : 1, box: { x: 0.3 + index * 0.05, y: 0.475, width: 0.035, height: 0.05 } }));
    const { source, model } = scene(cells);
    const result = recognizeTrackedWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, cells[8]!.box);
    expect(result).toMatchObject({ status: 'success', text: 'Aaaaaaaa' });
    expect(result.boxes).toHaveLength(9);
    expect(model.runSync.mock.calls.length).toBeLessThanOrEqual(3);
    expect(model.runSync.mock.calls.length).toBeGreaterThan(1);
  });

  it('rejects an actual clipped prefix instead of accepting the remaining digits', () => {
    const { source, model } = scene([60, 1, 3].map((mask, index) => ({ mask, box: { x: -0.01 + index * 0.075, y: 0.45, width: 0.05, height: 0.1 } })));
    const result = recognizeTrackedWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, { x: 0.065, y: 0.45, width: 0.125, height: 0.1 });
    expect(result).toMatchObject({ status: 'rejected', reason: 'clipped' });
    expect(result).not.toHaveProperty('text');
    expect(model.runSync.mock.calls.length).toBeLessThanOrEqual(3);
  });

  it('keeps uncertain prefix geometry for tracking and rejects it during final decoding', () => {
    const cells = [32, 1, 3].map((mask, index) => ({ mask, score: index === 0 ? 0.3 : 0.98, box: { x: 0.35 + index * 0.075, y: 0.45, width: 0.05, height: 0.1 } }));
    const { source, model } = scene(cells);
    const detection = detectWordImageCells(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, { x: 0, y: 0.3, width: 1, height: 0.4 });
    expect(detection.status).toBe('success');
    expect(detection.groups[0]?.boxes).toHaveLength(3);
    expect(detection.groups[0]?.score).toBeCloseTo(0.3);
    const result = recognizeTrackedWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, { x: 0.425, y: 0.45, width: 0.125, height: 0.1 });
    expect(result).toMatchObject({ status: 'rejected', reason: 'low_confidence' });
  });

  it('does not switch the shutter target to a neighboring word after it disappears', () => {
    const { source, model } = scene([{ mask: 1, box: { x: 0.65, y: 0.45, width: 0.04, height: 0.1 } }]);
    const result = recognizeTrackedWordImage(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, { x: 0.4, y: 0.45, width: 0.1, height: 0.1 });
    expect(result).toMatchObject({ status: 'rejected', reason: 'no_grid' });
  });

  it('releases context bitmaps when the interpreter throws', () => {
    const { source, model, crops } = scene([]);
    model.runSync.mockImplementation(() => { throw new Error('Interpreter failed'); });
    const result = detectWordImageCells(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, { x: 0, y: 0.3, width: 1, height: 0.4 });
    expect(result).toMatchObject({ status: 'error', reason: 'processing_error', groups: [] });
    expect(crops[0]?.dispose).toHaveBeenCalledTimes(1);
    expect(source.dispose).not.toHaveBeenCalled();
  });

  it('uses the native processor output and rejects malformed native tensors before inference', () => {
    const { source, model, crops } = scene([{ mask: 1, box: { x: 0.4, y: 0.45, width: 0.05, height: 0.1 } }]);
    const tensor = new Float32Array(WORD_MODEL_INPUT_LENGTH).fill(0.5).buffer;
    const resizePixels = jest.fn(() => ({ tensor, meanLuminance: 0.5, contrast: 0.1 }));
    const context = { x: 0, y: 0.3, width: 1, height: 0.4 };
    const result = detectWordImageCells(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, context, { resizePixels });
    expect(result.status).toBe('success');
    expect(resizePixels).toHaveBeenCalledWith(expect.any(ArrayBuffer), 480, 192, 'RGB', expect.any(Boolean));
    expect(model.runSync).toHaveBeenCalledWith([tensor]);
    resizePixels.mockReturnValueOnce({ tensor: new ArrayBuffer(4), meanLuminance: 0.5, contrast: 0.1 });
    expect(detectWordImageCells(source as unknown as Image, model, DEFAULT_WORD_MODEL_THRESHOLDS, context, { resizePixels }).status).toBe('error');
    expect(model.runSync).toHaveBeenCalledTimes(1);
    expect(crops.every((crop) => crop.dispose.mock.calls.length === 1)).toBe(true);
  });
});
