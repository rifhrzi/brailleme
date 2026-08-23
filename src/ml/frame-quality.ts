import type { QualityReason } from '@/types';

export const MIN_FRAME_LUMINANCE = 0.08;
export const MAX_FRAME_LUMINANCE = 0.96;
export const MIN_FRAME_CONTRAST = 0.035;

/**
 * Quality gate murah yang dapat dijalankan di worklet sebelum inferensi.
 * Nilai null berarti frame lolos gate provisional ini.
 */
export function getFrameQualityIssue(
  meanLuminance: number,
  contrast: number,
): QualityReason | null {
  'worklet';

  if (!Number.isFinite(meanLuminance) || !Number.isFinite(contrast)) {
    throw new Error('Statistik kualitas frame tidak valid.');
  }
  if (meanLuminance < MIN_FRAME_LUMINANCE) {
    return 'dark';
  }
  if (meanLuminance > MAX_FRAME_LUMINANCE) {
    return 'glare';
  }
  if (contrast < MIN_FRAME_CONTRAST) {
    return 'no_grid';
  }

  return null;
}
