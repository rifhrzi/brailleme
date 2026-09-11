import type { PhotoGeometry } from '../../modules/braille-word-geometry/src/WordPhotoGeometry.nitro';
import type { NormalizedBox } from '@/types/live-scan';
import { isNormalizedWordBox } from '@/types/word-scan';

export interface SensorPoint { readonly x: number; readonly y: number }
type PreviewGeometry = { convertViewPointToCameraPoint(point: SensorPoint): SensorPoint };
export interface WordFrameGeometry {
  readonly width: number;
  readonly height: number;
  readonly orientation: 'up' | 'right' | 'down' | 'left';
  readonly isMirrored: boolean;
  convertFramePointToCameraPoint(point: SensorPoint): SensorPoint;
}

/** Snapshot while Frame is alive: Nitro converter mirrors raw pixels, then rotates clockwise. */
export function snapshotWordFrameBoxSensorPoints(frame: WordFrameGeometry, box: NormalizedBox): readonly SensorPoint[] {
  'worklet';
  if (!isNormalizedWordBox(box) || ![frame.width, frame.height].every((v) => Number.isFinite(v) && v > 0)) throw new Error('Geometri frame tidak valid.');
  const quarterTurn = frame.orientation === 'right' || frame.orientation === 'left';
  const uprightWidth = quarterTurn ? frame.height : frame.width;
  const uprightHeight = quarterTurn ? frame.width : frame.height;
  const left = box.x * uprightWidth;
  const top = box.y * uprightHeight;
  const right = (box.x + box.width) * uprightWidth;
  const bottom = (box.y + box.height) * uprightHeight;
  return [{ x: left, y: top }, { x: right, y: top }, { x: right, y: bottom }, { x: left, y: bottom }].map((point) => {
    let raw: SensorPoint;
    switch (frame.orientation) {
      case 'up': raw = point; break;
      case 'right': raw = { x: point.y, y: frame.height - point.x }; break;
      case 'down': raw = { x: frame.width - point.x, y: frame.height - point.y }; break;
      case 'left': raw = { x: frame.width - point.y, y: point.x }; break;
      default: throw new Error('Orientasi frame tidak valid.');
    }
    if (frame.isMirrored) raw = { x: frame.width - raw.x, y: raw.y };
    const sensor = frame.convertFramePointToCameraPoint(raw);
    if (!sensor || !Number.isFinite(sensor.x) || !Number.isFinite(sensor.y)) throw new Error('Koordinat sensor frame tidak valid.');
    return { x: sensor.x, y: sensor.y };
  });
}

/** Maps copied preview sensor corners into the UPRIGHT Image returned by FrameConverter. */
export function mapWordSensorBoxToFrame(
  points: readonly SensorPoint[],
  frame: WordFrameGeometry & { convertCameraPointToFramePoint(point: SensorPoint): SensorPoint },
): NormalizedBox {
  'worklet';
  if (points.length !== 4 || ![frame.width, frame.height].every((v) => Number.isFinite(v) && v > 0)) throw new Error('Geometri frame tidak valid.');
  const quarterTurn = frame.orientation === 'right' || frame.orientation === 'left';
  const width = quarterTurn ? frame.height : frame.width;
  const height = quarterTurn ? frame.width : frame.height;
  const upright = points.map((point) => {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) throw new Error('Koordinat sensor tidak valid.');
    const raw = frame.convertCameraPointToFramePoint(point);
    if (!raw || !Number.isFinite(raw.x) || !Number.isFinite(raw.y)) throw new Error('Koordinat frame tidak valid.');
    const x = frame.isMirrored ? frame.width - raw.x : raw.x;
    const y = raw.y;
    switch (frame.orientation) {
      case 'up': return { x, y };
      case 'right': return { x: frame.height - y, y: x };
      case 'down': return { x: frame.width - x, y: frame.height - y };
      case 'left': return { x: y, y: frame.width - x };
      default: throw new Error('Orientasi frame tidak valid.');
    }
  });
  const left = Math.min(...upright.map((point) => point.x));
  const top = Math.min(...upright.map((point) => point.y));
  const right = Math.max(...upright.map((point) => point.x));
  const bottom = Math.max(...upright.map((point) => point.y));
  // Android Matrix.convertPoint converts through float32. A window placed
  // exactly at a sensor edge can return a tiny negative/overshooting bound
  // after the preview inverse and frame transform. Snap only subpixel noise;
  // larger clipping still fails instead of silently removing unseen context.
  const overshoot = Math.max(0, -left, -top, right - width, bottom - height);
  if (overshoot > 0.001) {
    if (overshoot <= 1) throw new Error('[word-frame:outside-subpixel]');
    if (overshoot <= Math.max(width, height) * 0.05) throw new Error('[word-frame:outside-small]');
    throw new Error('[word-frame:outside-large]');
  }
  const boundedLeft = Math.max(0, left);
  const boundedTop = Math.max(0, top);
  const boundedRight = Math.min(width, right);
  const boundedBottom = Math.min(height, bottom);
  const box = { x: boundedLeft / width, y: boundedTop / height, width: (boundedRight - boundedLeft) / width, height: (boundedBottom - boundedTop) / height };
  if (!isNormalizedWordBox(box)) throw new Error('Area pratinjau tidak seluruhnya tersedia pada frame.');
  return box;
}

/** Uses actual preview sensor transform; an off-screen rectangle is returned unclamped. */
export function mapWordSensorBoxToPreview(
  points: readonly SensorPoint[],
  camera: { convertCameraPointToViewPoint(point: SensorPoint): SensorPoint },
  previewWidth: number,
  previewHeight: number,
): NormalizedBox {
  if (points.length !== 4 || ![previewWidth, previewHeight].every((v) => Number.isFinite(v) && v > 0)) invalid();
  const view = points.map((point) => {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) invalid();
    const mapped = camera.convertCameraPointToViewPoint(point);
    if (!mapped || !Number.isFinite(mapped.x) || !Number.isFinite(mapped.y)) invalid();
    return { x: mapped.x, y: mapped.y };
  });
  const left = Math.min(...view.map((p) => p.x));
  const top = Math.min(...view.map((p) => p.y));
  const right = Math.max(...view.map((p) => p.x));
  const bottom = Math.max(...view.map((p) => p.y));
  if (right <= left || bottom <= top) invalid();
  return { x: left / previewWidth, y: top / previewHeight, width: (right - left) / previewWidth, height: (bottom - top) / previewHeight };
}

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
