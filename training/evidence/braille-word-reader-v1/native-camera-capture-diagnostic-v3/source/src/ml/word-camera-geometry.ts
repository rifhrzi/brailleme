import type { PhotoGeometry } from '../../modules/braille-word-geometry/src/WordPhotoGeometry.nitro';
import type { NormalizedBox } from '@/types/live-scan';
import { isNormalizedWordBox } from '@/types/word-scan';

export interface SensorPoint { readonly x: number; readonly y: number }
type PreviewGeometry = { convertViewPointToCameraPoint(point: SensorPoint): SensorPoint };

function invalid(): never {
  throw new Error('Area panduan kamera belum sesuai dengan foto. Silakan pindai ulang.');
}

/** VisionCamera receives view points in React Native dp and returns opaque sensor pixels. */
export function snapshotWordGuideSensorPoints(
  camera: PreviewGeometry | null,
  guide: NormalizedBox,
  previewWidth: number,
  previewHeight: number,
): readonly SensorPoint[] {
  if (!camera || !isNormalizedWordBox(guide) || ![previewWidth, previewHeight].every((v) => Number.isFinite(v) && v > 0)) invalid();
  const left = guide.x * previewWidth;
  const top = guide.y * previewHeight;
  const right = (guide.x + guide.width) * previewWidth;
  const bottom = (guide.y + guide.height) * previewHeight;
  return [{ x: left, y: top }, { x: right, y: top }, { x: right, y: bottom }, { x: left, y: bottom }].map((point) => {
    const sensor = camera.convertViewPointToCameraPoint(point);
    if (!sensor || !Number.isFinite(sensor.x) || !Number.isFinite(sensor.y)) invalid();
    // Copy values; never retain a live native point or preview object across the shutter await.
    return { x: sensor.x, y: sensor.y };
  });
}

/** Maps actual preview sensor points into the full upright Bitmap returned by Photo.toImageAsync. */
export function mapSensorGuideToWordPhoto(
  points: readonly SensorPoint[],
  geometry: PhotoGeometry,
  imageWidth: number,
  imageHeight: number,
): NormalizedBox {
  if (!geometry || points.length !== 4 || !points.every((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y))) invalid();
  const { width, height, rotationDegrees: rotation, isMirrored, sensorToBuffer: matrix, cropRect: crop } = geometry;
  if (![width, height, imageWidth, imageHeight].every((n) => Number.isSafeInteger(n) && n > 0) ||
      ![0, 90, 180, 270].includes(rotation) || typeof isMirrored !== 'boolean' ||
      !Array.isArray(matrix) || matrix.length !== 9 || !matrix.every(Number.isFinite) ||
      !Array.isArray(crop) || crop.length !== 4 || !crop.every(Number.isFinite)) invalid();
  const [a, b, c, d, e, f, g, h, i] = matrix as [number, number, number, number, number, number, number, number, number];
  const determinantTerms = [a * (e * i - f * h), -b * (d * i - f * g), c * (d * h - e * g)];
  const determinant = determinantTerms.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= 32 * Number.EPSILON * determinantTerms.reduce((sum, value) => sum + Math.abs(value), 0)) invalid();
  const quarterTurn = rotation === 90 || rotation === 270;
  if (imageWidth !== (quarterTurn ? height : width) || imageHeight !== (quarterTurn ? width : height)) invalid();
  const [cropLeft, cropTop, cropRight, cropBottom] = crop as [number, number, number, number];
  if (cropLeft < 0 || cropTop < 0 || cropRight > width || cropBottom > height || cropRight <= cropLeft || cropBottom <= cropTop) invalid();
  const upright = points.map((point) => {
    const divisor = g * point.x + h * point.y + i;
    if (!Number.isFinite(divisor) || Math.abs(divisor) < Number.EPSILON) invalid();
    let x = (a * point.x + b * point.y + c) / divisor;
    const y = (d * point.x + e * point.y + f) / divisor;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < cropLeft || y < cropTop || x > cropRight || y > cropBottom) invalid();
    // Bitmap.createBitmap applies mirror first, then clockwise rotation, then translates its bounds to zero.
    if (isMirrored) x = width - x;
    switch (rotation) {
      case 90: return { x: height - y, y: x };
      case 180: return { x: width - x, y: height - y };
      case 270: return { x: y, y: width - x };
      default: return { x, y };
    }
  });
  const left = Math.min(...upright.map((p) => p.x));
  const right = Math.max(...upright.map((p) => p.x));
  const top = Math.min(...upright.map((p) => p.y));
  const bottom = Math.max(...upright.map((p) => p.y));
  // CameraX rigid scale/crop/quarter-turn transforms must still form an axis-aligned rectangle.
  // Reject skew or degenerate quadrilaterals instead of including extra pixels in a bounding box.
  if (right - left < 1 || bottom - top < 1) invalid();
  const corners = new Set(upright.map((p) => {
    const x = Math.abs(p.x - left) <= 0.5 ? 0 : Math.abs(p.x - right) <= 0.5 ? 1 : invalid();
    const y = Math.abs(p.y - top) <= 0.5 ? 0 : Math.abs(p.y - bottom) <= 0.5 ? 1 : invalid();
    return `${x},${y}`;
  }));
  if (corners.size !== 4) invalid();
  const result = { x: left / imageWidth, y: top / imageHeight, width: (right - left) / imageWidth, height: (bottom - top) / imageHeight };
  if (!isNormalizedWordBox(result)) invalid();
  return result;
}
