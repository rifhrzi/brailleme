import {
  MAX_FRAME_LUMINANCE,
  MIN_FRAME_CONTRAST,
  MIN_FRAME_LUMINANCE,
  getFrameQualityIssue,
} from '@/ml/frame-quality';

describe('getFrameQualityIssue', () => {
  it.each([
    ['dark', MIN_FRAME_LUMINANCE - 0.001, MIN_FRAME_CONTRAST, 'dark'],
    ['glare', MAX_FRAME_LUMINANCE + 0.001, MIN_FRAME_CONTRAST, 'glare'],
    ['no_grid', 0.5, MIN_FRAME_CONTRAST - 0.001, 'no_grid'],
    ['lolos', MIN_FRAME_LUMINANCE, MIN_FRAME_CONTRAST, null],
    ['lolos batas terang', MAX_FRAME_LUMINANCE, MIN_FRAME_CONTRAST, null],
  ] as const)(
    'mengklasifikasikan frame %s sebelum inferensi',
    (_caseName, luminance, contrast, expected) => {
      expect(getFrameQualityIssue(luminance, contrast)).toBe(expected);
    },
  );

  it.each([
    [Number.NaN, 0.1],
    [0.5, Number.POSITIVE_INFINITY],
  ])('menolak statistik nonfinite', (luminance, contrast) => {
    expect(() => getFrameQualityIssue(luminance, contrast)).toThrow(
      'Statistik kualitas frame tidak valid.',
    );
  });
});
