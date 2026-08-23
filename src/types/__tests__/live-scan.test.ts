import { isValidLiveScanEvent } from '@/types';

const VALID_EVENT = {
  schemaVersion: 1,
  sessionId: 'session-1',
  timestampMs: 1_786_128_000_000,
  status: 'stable',
  text: 'ABC',
  confidence: 0.95,
  boxes: [{ x: 0.1, y: 0.2, width: 0.3, height: 0.4 }],
  decoderVersion: 'decoder-v1',
  modelVersion: 'model-v1',
} as const;

describe('validasi event Live', () => {
  it('menerima event dengan schema dan koordinat ternormalisasi', () => {
    expect(isValidLiveScanEvent(VALID_EVENT)).toBe(true);
  });

  it.each([
    null,
    { ...VALID_EVENT, schemaVersion: 2 },
    { ...VALID_EVENT, sessionId: '  ' },
    { ...VALID_EVENT, timestampMs: Number.NaN },
    { ...VALID_EVENT, status: 'ready' },
    { ...VALID_EVENT, confidence: Number.POSITIVE_INFINITY },
    { ...VALID_EVENT, confidence: 1.1 },
    { ...VALID_EVENT, boxes: [{ x: 0.8, y: 0, width: 0.3, height: 1 }] },
    { ...VALID_EVENT, decoderVersion: '' },
  ])('menolak payload native yang tidak valid', (event) => {
    expect(isValidLiveScanEvent(event)).toBe(false);
  });
});
