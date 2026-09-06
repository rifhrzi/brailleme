import { isNormalizedWordBox, isValidWordScanEvent, type WordScanEvent } from '@/types/word-scan';

function event(): WordScanEvent {
  const boxes = [{ x: 0.2, y: 0.4, width: 0.1, height: 0.2 }, { x: 0.4, y: 0.4, width: 0.1, height: 0.2 }];
  return {
    schemaVersion: 3, sessionId: 'session', captureId: 2, timestampMs: 100,
    status: 'success', text: 'AA', confidence: 0.95, boxes,
    cells: boxes.map((box) => ({ letter: 'A', mask: 1, confidence: 0.95, box })),
    decoderVersion: 'six-dot-word-v2', modelVersion: 'candidate-v1',
  };
}

describe('word scan event boundary', () => {
  it('accepts a complete ordered word including repeated letters', () => {
    expect(isValidWordScanEvent(event())).toBe(true);
  });

  it.each([
    { schemaVersion: 2 }, { sessionId: '' }, { captureId: -1 }, { captureId: 1.5 },
    { timestampMs: Number.NaN }, { status: 'stable' }, { decoderVersion: '' }, { modelVersion: '' },
    { confidence: 1.1 }, { confidence: 0.99 }, { confidence: undefined },
    { text: 'A' }, { text: 'aA' }, { text: 'A A' }, { text: '' }, { text: 'AAAAAAAAAAAAA' },
    { reason: 'dark' }, { cells: [] }, { boxes: [] },
  ])('rejects inconsistent success DTO %p', (change) => {
    expect(isValidWordScanEvent({ ...event(), ...change })).toBe(false);
  });

  it('rejects mismatched masks, unordered cells, invalid boxes and differing overlay coordinates', () => {
    const original = event();
    expect(isValidWordScanEvent({ ...original, cells: [...original.cells].reverse(), boxes: [...original.boxes].reverse() })).toBe(false);
    expect(isValidWordScanEvent({ ...original, cells: original.cells.map((cell) => ({ ...cell, mask: 3 })) })).toBe(false);
    expect(isValidWordScanEvent({ ...original, boxes: original.boxes.map((box) => ({ ...box, x: box.x + 0.01 })) })).toBe(false);
    expect(isValidWordScanEvent({ ...original, cells: [null, original.cells[1]] })).toBe(false);
    expect(isNormalizedWordBox({ x: 0, y: 0, width: 0, height: 1 })).toBe(false);
    expect(isNormalizedWordBox({ x: 0.9, y: 0, width: 0.2, height: 1 })).toBe(false);
  });

  it.each(['ready', 'capturing', 'processing', 'rejected', 'error'] as const)('accepts %s without partial text, confidence or decoded cells', (status) => {
    const { text: _text, confidence: _confidence, ...base } = event();
    const candidate = { ...base, status, cells: [], boxes: [], ...(status === 'rejected' ? { reason: 'low_confidence' } : status === 'error' ? { reason: 'processing_error' } : {}) };
    expect(isValidWordScanEvent(candidate)).toBe(true);
    expect(isValidWordScanEvent({ ...candidate, text: 'A' })).toBe(false);
    expect(isValidWordScanEvent({ ...candidate, confidence: 0.95 })).toBe(false);
  });

  it('requires reasons for failures and rejects values that are not DTOs', () => {
    const { text: _text, confidence: _confidence, ...base } = event();
    expect(isValidWordScanEvent({ ...base, status: 'rejected', boxes: [], cells: [] })).toBe(false);
    expect(isValidWordScanEvent({ ...base, status: 'error', reason: 'dark', boxes: [], cells: [] })).toBe(false);
    expect(isValidWordScanEvent({ ...base, status: 'error', reason: 'capture_error', boxes: [], cells: [] })).toBe(true);
    for (const value of [null, undefined, 1, '', []]) expect(isValidWordScanEvent(value)).toBe(false);
  });
});
