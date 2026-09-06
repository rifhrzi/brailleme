import type { PhotoGeometry } from '../../../modules/braille-word-geometry/src/WordPhotoGeometry.nitro';
import { mapSensorGuideToWordPhoto, snapshotWordGuideSensorPoints } from '@/ml/word-camera-geometry';

const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const rect = (x: number, y: number, w: number, h: number) => [{ x, y }, { x: x + w, y }, { x: x + w, y: y + h }, { x, y: y + h }];
const photo = (overrides: Partial<PhotoGeometry> = {}): PhotoGeometry => ({ width: 1440, height: 1920, sensorToBuffer: identity, rotationDegrees: 0, isMirrored: false, cropRect: [0, 0, 1440, 1920], ...overrides });

describe('preview sensor to captured photo mapping', () => {
  it('passes view coordinates in dp without multiplying the device pixel ratio', () => {
    const convert = jest.fn(({ x, y }: { x: number; y: number }) => ({ x: x * 4.5, y: y * 4.5 + 240 }));
    const points = snapshotWordGuideSensorPoints({ convertViewPointToCameraPoint: convert }, { x: 0.275, y: 0.44375, width: 0.45, height: 0.1125 }, 320, 320);
    expect(convert.mock.calls[0]?.[0]).toEqual({ x: 88, y: 142 });
    expect(mapSensorGuideToWordPhoto(points, photo(), 1440, 1920)).toEqual({ x: 0.275, y: 0.4578125, width: expect.closeTo(0.45), height: 0.084375 });
  });

  it('handles a 16:9 preview cropped from the center of a 4:3 photo sensor', () => {
    const points = snapshotWordGuideSensorPoints({ convertViewPointToCameraPoint: ({ x, y }) => ({ x: 180 + x * 3, y: 420 + y * 3 }) }, { x: 0.275, y: 0.44375, width: 0.45, height: 0.1125 }, 360, 360);
    const box = mapSensorGuideToWordPhoto(points, photo(), 1440, 1920);
    expect(box.x).toBeCloseTo(0.33125);
    expect(box.y).toBeCloseTo(0.468359375);
    expect(box.width).toBeCloseTo(0.3375);
    expect(box.height).toBeCloseTo(0.06328125);
  });

  it('handles the reverse case with a cropped 16:9 photo and a 4:3 preview', () => {
    const box = mapSensorGuideToWordPhoto(rect(396, 879, 648, 162), photo({ width: 1080, sensorToBuffer: [1, 0, -180, 0, 1, 0, 0, 0, 1], cropRect: [0, 0, 1080, 1920] }), 1080, 1920);
    expect(box).toEqual({ x: 0.2, y: 879 / 1920, width: 0.6, height: 162 / 1920 });
  });

  it('uses the actual sensor origin and scale rather than treating sensor coordinates as normalized', () => {
    const geometry = photo({ sensorToBuffer: [0.5, 0, -100, 0, 0.25, -75, 0, 0, 1] });
    expect(mapSensorGuideToWordPhoto(rect(1000, 1900, 400, 800), geometry, 1440, 1920)).toEqual({ x: 400 / 1440, y: 400 / 1920, width: 200 / 1440, height: 200 / 1920 });
  });

  it.each([
    [0, false, 100, 50, 20, 10], [90, false, 50, 100, 25, 20],
    [180, false, 100, 50, 60, 25], [270, false, 50, 100, 10, 60],
    [0, true, 100, 50, 60, 10], [90, true, 50, 100, 25, 60],
    [180, true, 100, 50, 20, 25], [270, true, 50, 100, 10, 20],
  ])('matches upright bitmap rotation %s and mirror %s', (rotation, mirrored, iw, ih, x, y) => {
    const geometry = photo({ width: 100, height: 50, cropRect: [0, 0, 100, 50], rotationDegrees: rotation as number, isMirrored: mirrored as boolean });
    const box = mapSensorGuideToWordPhoto(rect(20, 10, 20, 15), geometry, iw as number, ih as number);
    expect(box.x).toBeCloseTo((x as number) / (iw as number));
    expect(box.y).toBeCloseTo((y as number) / (ih as number));
    expect(box.width).toBeCloseTo((rotation === 90 || rotation === 270 ? 15 : 20) / (iw as number));
    expect(box.height).toBeCloseTo((rotation === 90 || rotation === 270 ? 20 : 15) / (ih as number));
  });

  it.each([
    { sensorToBuffer: [] }, { sensorToBuffer: [0, 0, 0, 0, 0, 0, 0, 0, 0] },
    { sensorToBuffer: [1, 2, 3, 2, 4, 6, 0, 0, 1] }, { sensorToBuffer: [NaN, ...identity.slice(1)] },
    { rotationDegrees: 45 }, { cropRect: [0, 0, 100, 100] },
    { sensorToBuffer: [1, 0, -500, 0, 1, 0, 0, 0, 1] }, { sensorToBuffer: [1, 0.1, 0, 0, 1, 0, 0, 0, 1] },
  ])('fails closed for invalid, singular, cropped-out or skewed metadata %j', (overrides) => {
    expect(() => mapSensorGuideToWordPhoto(rect(200, 200, 100, 50), photo(overrides), 1440, 1920)).toThrow('panduan');
  });

  it('rejects incompatible decoded dimensions and collapsed or missing sensor coordinates', () => {
    expect(() => mapSensorGuideToWordPhoto(rect(20, 10, 20, 15), photo(), 1920, 1440)).toThrow('panduan');
    expect(() => mapSensorGuideToWordPhoto(rect(20, 10, 0, 15), photo(), 1440, 1920)).toThrow('panduan');
    expect(() => mapSensorGuideToWordPhoto([], photo(), 1440, 1920)).toThrow('panduan');
    expect(() => snapshotWordGuideSensorPoints(null, { x: 0, y: 0, width: 1, height: 1 }, 320, 320)).toThrow('panduan');
    expect(() => snapshotWordGuideSensorPoints({ convertViewPointToCameraPoint: () => ({ x: Infinity, y: 0 }) }, { x: 0, y: 0, width: 1, height: 1 }, 320, 320)).toThrow('panduan');
  });
});
