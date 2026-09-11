import { isNormalizedWordBox, isValidWordScanEvent, type WordScanEvent } from '@/types/word-scan';
import { WORD_DECODER_VERSION } from '@/ml/word-model-contract';
import { parseWordMasks } from '@/ml/word-symbols';

function event(masks = [1, 1]): WordScanEvent {
  const parsed = parseWordMasks(masks);
  if (parsed.status !== 'success') throw new Error(parsed.reason);
  const boxes = masks.map((_, index) => ({ x: 0.02 + index * 0.04, y: 0.4, width: 0.03, height: 0.2 }));
  return {
    schemaVersion: 4, sessionId: 'session', captureId: 2, timestampMs: 100,
    status: 'success', text: parsed.text, confidence: 0.95, boxes,
    cells: boxes.map((box, index) => ({ ...parsed.symbols[index]!, confidence: 0.95, box })),
    decoderVersion: WORD_DECODER_VERSION, modelVersion: 'candidate-v1',
  };
}

describe('word scan event boundary', () => {
  it('accepts a complete ordered word including repeated letters', () => {
    expect(isValidWordScanEvent(event())).toBe(true);
  });

  it.each([
    { schemaVersion: 2 }, { schemaVersion: 3 }, { sessionId: '' }, { captureId: -1 }, { captureId: 1.5 },
    { timestampMs: Number.NaN }, { status: 'stable' }, { decoderVersion: '' }, { modelVersion: '' },
    { decoderVersion: 'six-dot-word-v2' },
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
    expect(isValidWordScanEvent({ ...base, status: 'error', reason: 'tracking_error', boxes: [], cells: [] })).toBe(true);
    expect(isValidWordScanEvent({ ...base, status: 'rejected', reason: 'tracking_error', boxes: [], cells: [] })).toBe(false);
    for (const value of [null, undefined, 1, '', []]) expect(isValidWordScanEvent(value)).toBe(false);
  });

  it.each([[32, 1, 3], [32, 32, 1, 3], [60, 26, 26, 1], Array.from({ length: 12 }, () => [32, 1]).flat()])('validates indicator-to-text mapping for %p', (...masks) => {
    expect(isValidWordScanEvent(event(masks))).toBe(true);
  });

  it('rejects invented case, omitted indicators, and forged roles/output', () => {
    const original = event([32, 1, 3]);
    expect(isValidWordScanEvent({ ...original, text: 'ab' })).toBe(false);
    expect(isValidWordScanEvent({ ...original, cells: original.cells.slice(1), boxes: original.boxes.slice(1) })).toBe(false);
    expect(isValidWordScanEvent({ ...original, cells: original.cells.map((cell) => ({ ...cell, role: 'letter' })) })).toBe(false);
    expect(isValidWordScanEvent({ ...original, cells: original.cells.map((cell) => ({ ...cell, output: 'a' })) })).toBe(false);
    const numeric = event([60, 1, 3]);
    expect(isValidWordScanEvent({ ...numeric, text: 'ab' })).toBe(false);
  });

  it('requires indicator confidence in the complete token confidence', () => {
    const original = event([60, 1, 3]);
    const cells = original.cells.map((cell, index) => ({ ...cell, confidence: index === 0 ? 0.8 : 0.95 }));
    expect(isValidWordScanEvent({ ...original, cells })).toBe(false);
    expect(isValidWordScanEvent({ ...original, cells, confidence: 0.8 })).toBe(true);
  });
});
