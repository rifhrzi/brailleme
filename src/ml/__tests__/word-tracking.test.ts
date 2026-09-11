import { DEFAULT_WORD_MODEL_THRESHOLDS } from '@/ml/word-model-contract';
import {
  advanceWordTracking, getWordSearchWindows, getWordTrackingContext, getWordTrackingTarget,
  groupWordProposals, INITIAL_WORD_TRACKING_STATE, type WordTrackingGroup,
} from '@/ml/word-tracking';

const group = (x = 0.4, y = 0.45, complete = true): WordTrackingGroup => ({
  box: { x, y, width: 0.15, height: 0.1 },
  boxes: [{ x, y, width: 0.05, height: 0.1 }, { x: x + 0.1, y, width: 0.05, height: 0.1 }],
  score: 0.9, complete,
});

describe('word localization windows and groups', () => {
  it.each([[360, 360], [360, 640], [640, 360], [1000, 100]])('covers %sx%s without stretching and searches center first', (width, height) => {
    const windows = getWordSearchWindows(width, height);
    for (const window of windows) {
      expect(window.width * width / (window.height * height)).toBeCloseTo(2.5);
      expect(window.x).toBeGreaterThanOrEqual(0);
      expect(window.y).toBeGreaterThanOrEqual(0);
      expect(window.x + window.width).toBeLessThanOrEqual(1.000001);
      expect(window.y + window.height).toBeLessThanOrEqual(1.000001);
    }
    for (let x = 0; x <= 1; x += 0.1) for (let y = 0; y <= 1; y += 0.1) {
      expect(windows.some((window) => x >= window.x - 1e-8 && x <= window.x + window.width + 1e-8 && y >= window.y - 1e-8 && y <= window.y + window.height + 1e-8)).toBe(true);
    }
    const distance = (window: typeof windows[number]) => Math.hypot(window.x + window.width / 2 - 0.5, window.y + window.height / 2 - 0.5);
    expect(distance(windows[0]!)).toBe(Math.min(...windows.map(distance)));
  });

  it('uses three overlapping center/top/bottom strips for the square scanner', () => {
    expect(getWordSearchWindows(360, 360)).toEqual([
      { x: 0, y: 0.3, width: 1, height: 0.4 },
      { x: 0, y: 0, width: 1, height: 0.4 },
      { x: 0, y: 0.6, width: 1, height: 0.4 },
    ]);
  });

  it('keeps weak physical prefix cells, splits words at a real gap, and separates rows', () => {
    const proposals = [
      { score: 0.3, box: { x: 0.2, y: 0.3, width: 0.03, height: 0.08 } },
      { score: 0.98, box: { x: 0.26, y: 0.3, width: 0.03, height: 0.08 } },
      { score: 0.98, box: { x: 0.55, y: 0.3, width: 0.03, height: 0.08 } },
      { score: 0.98, box: { x: 0.2, y: 0.6, width: 0.03, height: 0.08 } },
    ];
    const groups = groupWordProposals(proposals, 480, 480, DEFAULT_WORD_MODEL_THRESHOLDS);
    expect(groups.map((item) => item.boxes.length)).toEqual([2, 1, 1]);
    expect(groups[0]?.score).toBe(0.3);
    expect(groups[0]?.box.x).toBe(0.2);
  });

  it('marks boundary cells incomplete and pads context using physical cell height', () => {
    const [incomplete] = groupWordProposals([{ score: 0.9, box: { x: -0.01, y: 0.2, width: 0.05, height: 0.08 } }], 480, 480, DEFAULT_WORD_MODEL_THRESHOLDS);
    expect(incomplete?.complete).toBe(false);
    expect(incomplete?.box.x).toBe(0);
    const context = getWordTrackingContext(group(), 480, 640);
    expect(context.width * 480 / (context.height * 640)).toBeCloseTo(2.5);
    expect(context.x).toBeLessThan(group().box.x);
    expect(context.x + context.width).toBeGreaterThan(group().box.x + group().box.width);
  });
});

describe('automatic word tracking', () => {
  it('requires two observations and prevents an old pending observation from acquiring', () => {
    const one = advanceWordTracking(INITIAL_WORD_TRACKING_STATE, [group()], 100);
    expect(one.target).toBeUndefined();
    expect(advanceWordTracking(one, [group()], 1000).target).toBeUndefined();
    const two = advanceWordTracking(one, [group()], 300);
    expect(getWordTrackingTarget(two, 300)?.ready).toBe(true);
  });

  it('keeps raw coordinates current while smoothing only the overlay', () => {
    let state = advanceWordTracking(INITIAL_WORD_TRACKING_STATE, [group()], 100);
    state = advanceWordTracking(state, [group()], 300);
    state = advanceWordTracking(state, [group(0.42)], 500);
    expect(state.target?.raw.box.x).toBe(0.42);
    expect(state.target?.display.box.x).toBeCloseTo(0.407);
    expect(state.target?.lastSeenMs).toBe(500);
  });

  it('disables a stale/incomplete target and eventually clears it even without another frame', () => {
    let state = advanceWordTracking(INITIAL_WORD_TRACKING_STATE, [group()], 100);
    state = advanceWordTracking(state, [group()], 300);
    expect(getWordTrackingTarget(state, 800)?.ready).toBe(true);
    expect(getWordTrackingTarget(state, 801)?.ready).toBe(false);
    expect(getWordTrackingTarget(state, 1100)).toBeUndefined();
    state = advanceWordTracking(state, [group(0.4, 0.45, false)], 500);
    expect(getWordTrackingTarget(state, 500)?.ready).toBe(false);
  });

  it('resets the outer rectangle together with cells when a newly visible indicator changes the count', () => {
    let state = advanceWordTracking(INITIAL_WORD_TRACKING_STATE, [group()], 100);
    state = advanceWordTracking(state, [group()], 300);
    const expanded: WordTrackingGroup = { ...group(), box: { x: 0.32, y: 0.45, width: 0.23, height: 0.1 }, boxes: [{ x: 0.32, y: 0.45, width: 0.05, height: 0.1 }, ...group().boxes] };
    state = advanceWordTracking(state, [expanded], 500);
    expect(state.target?.display.box).toEqual(expanded.box);
    expect(state.target?.display.boxes).toEqual(expanded.boxes);
  });

  it('switches to a substantially nearer group only after two observations', () => {
    const far = group(0.05);
    const near = group(0.4);
    let state = advanceWordTracking(INITIAL_WORD_TRACKING_STATE, [far], 100);
    state = advanceWordTracking(state, [far], 300);
    const oldId = state.target?.id;
    state = advanceWordTracking(state, [far, near], 500);
    expect(state.target?.id).toBe(oldId);
    state = advanceWordTracking(state, [far, near], 700);
    expect(state.target?.raw.box.x).toBe(near.box.x);
    expect(state.target?.id).not.toBe(oldId);
  });

  it('keeps the last target through a discovery miss and ignores out-of-order observations', () => {
    let state = advanceWordTracking(INITIAL_WORD_TRACKING_STATE, [group()], 100);
    state = advanceWordTracking(state, [group()], 300);
    const discovery = advanceWordTracking(state, [], 500);
    expect(discovery.target?.lastSeenMs).toBe(300);
    expect(getWordTrackingTarget(discovery, 500)?.ready).toBe(true);
    expect(advanceWordTracking(discovery, [group(0.42)], 200)).toBe(discovery);
  });

  it('does not associate an overlapping tall rectangle from a different row', () => {
    const upper: WordTrackingGroup = { ...group(0.4, 0.2), box: { x: 0.4, y: 0.2, width: 0.15, height: 0.3 } };
    const lower: WordTrackingGroup = { ...group(0.4, 0.35), box: { x: 0.4, y: 0.35, width: 0.15, height: 0.3 } };
    let state = advanceWordTracking(INITIAL_WORD_TRACKING_STATE, [upper], 100);
    state = advanceWordTracking(state, [upper], 300);
    const next = advanceWordTracking(state, [lower], 500);
    expect(next.target?.lastSeenMs).toBe(300);
    expect(next.target?.raw).toBe(upper);
  });
});
