import { resizeWordRawPixels } from '@/ml/word-image-processing';

import references from './fixtures/word-resize-reference.json';

const { createHash } = jest.requireActual<{
  createHash(algorithm: string): { update(bytes: Uint8Array): { digest(format: 'hex'): string } };
}>('node:crypto');

describe('deterministic word photo resampling', () => {
  it.each(references.cases)('matches all Pillow RGB bytes and float32 values for $width x $height', (fixture) => {
    const bytes = new Uint8Array(fixture.width * fixture.height * 3);
    for (let y = 0; y < fixture.height; y += 1) {
      for (let x = 0; x < fixture.width; x += 1) {
        const index = (y * fixture.width + x) * 3;
        bytes[index] = (3 * x + 5 * y) % 256;
        bytes[index + 1] = (11 * x + 2 * y) % 256;
        bytes[index + 2] = ((Math.floor(x / 3) + Math.floor(y / 5)) % 2) * 255;
      }
    }
    const { tensor } = resizeWordRawPixels({ width: fixture.width, height: fixture.height, pixelFormat: 'RGB', buffer: bytes.buffer });
    const resized = new Uint8Array(tensor.length);
    for (let index = 0; index < tensor.length; index += 1) resized[index] = Math.round(tensor[index]! * 255);
    expect(createHash('sha256').update(resized).digest('hex')).toBe(fixture.rgb_sha256);
    expect(createHash('sha256').update(new Uint8Array(tensor.buffer)).digest('hex')).toBe(fixture.float32_sha256);
  });

  it('corrects only the pinned Android Nitro BGRA metadata and preserves generic BGRA', () => {
    const raw = { width: 1, height: 1, pixelFormat: 'BGRA' as const, buffer: new Uint8Array([11, 22, 33, 255]).buffer };
    const android = resizeWordRawPixels(raw, true);
    const generic = resizeWordRawPixels(raw);
    expect(Array.from(android.tensor.slice(0, 3)).map((value) => Math.round(value * 255))).toEqual([11, 22, 33]);
    expect(Array.from(generic.tensor.slice(0, 3)).map((value) => Math.round(value * 255))).toEqual([33, 22, 11]);
  });

  it.each([
    ['ARGB', [255, 11, 22, 33]], ['XRGB', [0, 11, 22, 33]],
    ['ABGR', [255, 33, 22, 11]], ['XBGR', [0, 33, 22, 11]],
    ['RGBA', [11, 22, 33, 255]], ['RGBX', [11, 22, 33, 0]],
    ['BGRX', [33, 22, 11, 0]], ['BGR', [33, 22, 11]],
  ] as const)('resamples %s channel ordering consistently', (pixelFormat, pixel) => {
    const result = resizeWordRawPixels({ width: 1, height: 1, pixelFormat, buffer: new Uint8Array(pixel).buffer });
    expect(Array.from(result.tensor.slice(0, 3)).map((value) => Math.round(value * 255))).toEqual([11, 22, 33]);
  });

  it.each([[0, 1], [1, 0], [0.5, 2], [Number.NaN, 1]])('rejects invalid source dimensions %s x %s', (width, height) => {
    expect(() => resizeWordRawPixels({ width, height, pixelFormat: 'RGB', buffer: new ArrayBuffer(3) })).toThrow('Dimensi');
  });

  it('rejects unsupported pixel formats and malformed buffers before allocating passes', () => {
    expect(() => resizeWordRawPixels({ width: 1, height: 1, pixelFormat: 'unknown', buffer: new ArrayBuffer(3) })).toThrow('Format');
    expect(() => resizeWordRawPixels({ width: 1, height: 1, pixelFormat: 'RGBA', buffer: new ArrayBuffer(3) })).toThrow('buffer');
  });
});
