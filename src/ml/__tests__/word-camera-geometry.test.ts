import type { PhotoGeometry } from '../../../modules/braille-word-geometry/src/WordPhotoGeometry.nitro';
import { mapSensorGuideToWordPhoto, snapshotWordGuideSensorPoints, mapWordSensorBoxToFrame, mapWordSensorBoxToPreview, snapshotWordFrameBoxSensorPoints, type WordFrameGeometry } from '@/ml/word-camera-geometry';

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

describe('live frame sensor geometry', () => {
  const frame = (orientation: WordFrameGeometry['orientation'], isMirrored: boolean) => ({
    width: 100, height: 50, orientation, isMirrored,
    convertFramePointToCameraPoint: ({ x, y }: { x: number; y: number }) => ({ x: x * 2 + 30, y: y * 3 - 20 }),
    convertCameraPointToFramePoint: ({ x, y }: { x: number; y: number }) => ({ x: (x - 30) / 2, y: (y + 20) / 3 }),
  });

  it.each([
    ['up', false, 100, 50, 20, 10], ['right', false, 50, 100, 25, 20],
    ['down', false, 100, 50, 60, 25], ['left', false, 50, 100, 10, 60],
    ['up', true, 100, 50, 60, 10], ['right', true, 50, 100, 25, 60],
    ['down', true, 100, 50, 20, 25], ['left', true, 50, 100, 10, 20],
  ] as const)('maps upright %s mirror=%s through actual sensor origin and scale', (orientation, mirrored, iw, ih, x, y) => {
    const live = frame(orientation, mirrored);
    const sensor = rect(70, 10, 40, 45);
    const box = mapWordSensorBoxToFrame(sensor, live);
    expect(box.x).toBeCloseTo(x / iw);
    expect(box.y).toBeCloseTo(y / ih);
    expect(box.width).toBeCloseTo((iw === 50 ? 15 : 20) / iw);
    expect(box.height).toBeCloseTo((iw === 50 ? 20 : 15) / ih);
    const roundtrip = snapshotWordFrameBoxSensorPoints(live, box);
    expect(Math.min(...roundtrip.map((point) => point.x))).toBeCloseTo(70);
    expect(Math.min(...roundtrip.map((point) => point.y))).toBeCloseTo(10);
    expect(Math.max(...roundtrip.map((point) => point.x))).toBeCloseTo(110);
    expect(Math.max(...roundtrip.map((point) => point.y))).toBeCloseTo(55);
  });

  it('maps to preview using its real crop and dp, preserving offscreen bounds', () => {
    const camera = { convertCameraPointToViewPoint: ({ x, y }: { x: number; y: number }) => ({ x: (x - 100) / 2, y: (y - 300) / 2 }) };
    expect(mapWordSensorBoxToPreview(rect(60, 500, 400, 100), camera, 320, 320)).toEqual({ x: -0.0625, y: 0.3125, width: 0.625, height: 0.15625 });
  });

  it('snaps float32 edge noise on a physically upright 480x640 frame', () => {
    const live = {
      width: 480, height: 640, orientation: 'up' as const, isMirrored: false,
      convertCameraPointToFramePoint: ({ x, y }: { x: number; y: number }) => ({ x, y }),
      convertFramePointToCameraPoint: ({ x, y }: { x: number; y: number }) => ({ x, y }),
    };
    const noise = 2 ** -14;
    expect(mapWordSensorBoxToFrame(rect(-noise, 128, 480 + noise * 2, 192), live)).toEqual({ x: 0, y: 0.2, width: 1, height: 0.3 });
    expect(mapWordSensorBoxToFrame(rect(48, -noise, 96, 640 + noise * 2), live)).toEqual({ x: 0.1, y: 0, width: 0.2, height: 1 });
  });

  it.each(['up', 'right', 'down', 'left'] as const)('keeps edge-noise correction in upright pixels for %s rotation and both mirror modes', (orientation) => {
    for (const mirrored of [false, true]) {
      const live = frame(orientation, mirrored);
      const points = snapshotWordFrameBoxSensorPoints(live, { x: 0, y: 0, width: 1, height: 1 });
      const convert = live.convertCameraPointToFramePoint;
      live.convertCameraPointToFramePoint = (point) => {
        const raw = convert(point);
        return { x: raw.x + (raw.x === 0 ? -0.0001 : 0.0001), y: raw.y + (raw.y === 0 ? -0.0001 : 0.0001) };
      };
      expect(mapWordSensorBoxToFrame(points, live)).toEqual({ x: 0, y: 0, width: 1, height: 1 });
    }
  });

  it.each([
    [0.0011, 'outside-subpixel'], [0.5, 'outside-subpixel'],
    [2, 'outside-small'], [10, 'outside-large'],
  ] as const)('still rejects actual clipping of %spx without publishing absolute coordinates', (overshoot, code) => {
    const live = frame('up', false);
    // Native frame x=0 is sensor x=30; a twofold sensor scale preserves the
    // overshoot in frame pixels for the tolerance boundary test.
    expect(() => mapWordSensorBoxToFrame(rect(30 - overshoot * 2, 10, 40, 30), live)).toThrow(`[word-frame:${code}]`);
  });

  it('rejects missing geometry and preview windows cropped outside the live sensor buffer', () => {
    expect(() => mapWordSensorBoxToFrame([], frame('up', false))).toThrow('Geometri');
    expect(() => mapWordSensorBoxToFrame(rect(0, 0, 400, 100), frame('up', false))).toThrow('[word-frame:outside-large]');
    expect(() => snapshotWordFrameBoxSensorPoints(frame('up', false), { x: -0.1, y: 0, width: 0.5, height: 0.5 })).toThrow('Geometri');
    expect(() => mapWordSensorBoxToPreview(rect(0, 0, 1, 1), { convertCameraPointToViewPoint: () => ({ x: NaN, y: 0 }) }, 320, 320)).toThrow('panduan');
  });
});
