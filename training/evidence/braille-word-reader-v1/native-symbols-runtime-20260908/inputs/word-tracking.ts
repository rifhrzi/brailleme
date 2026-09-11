import { wordBoxIou } from '@/ml/word-decoder';
import { WORD_MODEL_CONTEXT_ASPECT_RATIO, type WordModelThresholds } from '@/ml/word-model-contract';
import type { NormalizedBox } from '@/types/live-scan';

export interface WordTrackingGroup {
  readonly box: NormalizedBox;
  readonly boxes: readonly NormalizedBox[];
  readonly score: number;
  readonly complete: boolean;
}
export interface WordTrackedTarget {
  readonly id: number;
  readonly raw: WordTrackingGroup;
  readonly display: WordTrackingGroup;
  readonly lastSeenMs: number;
}
export interface WordTrackingState {
  readonly target?: WordTrackedTarget | undefined;
  readonly pending?: WordTrackingGroup;
  readonly pendingSeenMs?: number;
  readonly observations: number;
  readonly nextId: number;
}
export const INITIAL_WORD_TRACKING_STATE: WordTrackingState = { observations: 0, nextId: 1 };
export const WORD_TRACKING_INTERVAL_MS = 200;
export const WORD_TRACKING_CAPTURE_AGE_MS = 500;
export const WORD_TRACKING_CLEAR_AGE_MS = 800;
const FULL_IMAGE = { x: 0, y: 0, width: 1, height: 1 };

function median(values: readonly number[]): number {
  'worklet';
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

export function unionWordBoxes(boxes: readonly NormalizedBox[]): NormalizedBox {
  'worklet';
  if (!boxes.length) throw new Error('Tidak ada sel untuk dilacak.');
  const x = Math.min(...boxes.map((box) => box.x));
  const y = Math.min(...boxes.map((box) => box.y));
  return {
    x, y,
    width: Math.max(...boxes.map((box) => box.x + box.width)) - x,
    height: Math.max(...boxes.map((box) => box.y + box.height)) - y,
  };
}

function clampBox(box: NormalizedBox): NormalizedBox {
  'worklet';
  const x = Math.max(0, box.x);
  const y = Math.max(0, box.y);
  return { x, y, width: Math.min(1, box.x + box.width) - x, height: Math.min(1, box.y + box.height) - y };
}

/** Same-row grouping precedes semantic decoding, so indicators and uncertain cells survive. */
export function groupWordProposals(
  proposals: readonly { readonly score: number; readonly box: NormalizedBox }[],
  sourceWidth: number,
  sourceHeight: number,
  thresholds: WordModelThresholds,
  context: NormalizedBox = FULL_IMAGE,
): WordTrackingGroup[] {
  'worklet';
  const rows: { score: number; box: NormalizedBox }[][] = [];
  for (const proposal of [...proposals].sort((a, b) => a.box.y + a.box.height / 2 - b.box.y - b.box.height / 2)) {
    if (proposal.box.width <= 0 || proposal.box.height <= 0 || proposal.box.x >= 1 || proposal.box.y >= 1 ||
        proposal.box.x + proposal.box.width <= 0 || proposal.box.y + proposal.box.height <= 0) continue;
    const center = proposal.box.y + proposal.box.height / 2;
    const row = rows.find((candidate) => {
      const centers = [...candidate.map(({ box }) => box.y + box.height / 2), center];
      return Math.max(...centers) - Math.min(...centers) <= thresholds.maxRowDeviation * median([...candidate.map(({ box }) => box.height), proposal.box.height]);
    });
    if (row) row.push(proposal);
    else rows.push([proposal]);
  }
  const result: WordTrackingGroup[] = [];
  for (const row of rows) {
    row.sort((a, b) => a.box.x + a.box.width / 2 - b.box.x - b.box.width / 2);
    const height = median(row.map(({ box }) => box.height)) * sourceHeight;
    const groups: typeof rows = [];
    for (const proposal of row) {
      const previousGroup = groups[groups.length - 1];
      const previous = previousGroup?.[previousGroup.length - 1]?.box;
      const gap = previous ? (proposal.box.x + proposal.box.width / 2 - previous.x - previous.width / 2) * sourceWidth : Infinity;
      if (previousGroup && gap <= thresholds.maxGapRatio * height) previousGroup.push(proposal);
      else groups.push([proposal]);
    }
    for (const group of groups) {
      const union = unionWordBoxes(group.map(({ box }) => box));
      const marginX = Math.max(context.width * thresholds.edgeMargin, height / sourceWidth * 0.2);
      const marginY = context.height * thresholds.edgeMargin;
      const boxes = group.map(({ box }) => clampBox(box));
      result.push({
        box: unionWordBoxes(boxes), boxes,
        score: Math.min(...group.map(({ score }) => score)),
        complete: union.x > context.x + marginX && union.y > context.y + marginY &&
          union.x + union.width < context.x + context.width - marginX &&
          union.y + union.height < context.y + context.height - marginY,
      });
    }
  }
  return result;
}

function centerDistance(box: NormalizedBox): number {
  'worklet';
  return Math.hypot(box.x + box.width / 2 - 0.5, box.y + box.height / 2 - 0.5);
}

/** Cover the actual viewport with overlapping undistorted 2.5:1 model windows. */
export function getWordSearchWindows(width: number, height: number): NormalizedBox[] {
  'worklet';
  if (![width, height].every((value) => Number.isFinite(value) && value > 0)) throw new Error('Ukuran pratinjau tidak valid.');
  const tileWidth = Math.min(width, height * WORD_MODEL_CONTEXT_ASPECT_RATIO);
  const tileHeight = tileWidth / WORD_MODEL_CONTEXT_ASPECT_RATIO;
  const horizontal = width > tileWidth;
  const extent = horizontal ? width : height;
  const tileExtent = horizontal ? tileWidth : tileHeight;
  const count = Math.max(1, Math.ceil((extent - tileExtent) / (tileExtent * 0.75)) + 1);
  const windows = Array.from({ length: count }, (_, index) => {
    const offset = count === 1 ? 0 : index * (extent - tileExtent) / (count - 1);
    return { x: horizontal ? offset / width : 0, y: horizontal ? 0 : offset / height, width: tileWidth / width, height: tileHeight / height };
  });
  return windows.sort((a, b) => centerDistance(a) - centerDistance(b) || a.y - b.y || a.x - b.x);
}

/** Pad around physical cells, then fit an undistorted window within the source. */
export function getWordTrackingContext(group: WordTrackingGroup, width: number, height: number, expansion = 1): NormalizedBox {
  'worklet';
  const cellHeight = median(group.boxes.map((box) => box.height)) * height;
  const requiredWidth = Math.max(group.box.width * width + cellHeight * 3, (group.box.height * height + cellHeight) * WORD_MODEL_CONTEXT_ASPECT_RATIO) * expansion;
  const contextWidth = Math.min(width, height * WORD_MODEL_CONTEXT_ASPECT_RATIO, requiredWidth);
  const contextHeight = contextWidth / WORD_MODEL_CONTEXT_ASPECT_RATIO;
  const x = Math.max(0, Math.min(width - contextWidth, (group.box.x + group.box.width / 2) * width - contextWidth / 2));
  const y = Math.max(0, Math.min(height - contextHeight, (group.box.y + group.box.height / 2) * height - contextHeight / 2));
  return { x: x / width, y: y / height, width: contextWidth / width, height: contextHeight / height };
}

function matches(a: WordTrackingGroup, b: WordTrackingGroup): boolean {
  'worklet';
  const cellHeight = median([...a.boxes, ...b.boxes].map((box) => box.height));
  const rowDistance = Math.abs(a.box.y + a.box.height / 2 - b.box.y - b.box.height / 2);
  return wordBoxIou(a.box, b.box) >= 0.3 && rowDistance <= cellHeight * 0.55;
}

function smoothBox(previous: NormalizedBox, next: NormalizedBox): NormalizedBox {
  'worklet';
  const alpha = 0.35;
  return {
    x: previous.x + alpha * (next.x - previous.x), y: previous.y + alpha * (next.y - previous.y),
    width: previous.width + alpha * (next.width - previous.width), height: previous.height + alpha * (next.height - previous.height),
  };
}

export function getWordTrackingTarget(state: WordTrackingState, nowMs: number): (WordTrackedTarget & { readonly ready: boolean }) | undefined {
  'worklet';
  const target = state.target;
  if (!target || nowMs < target.lastSeenMs || nowMs - target.lastSeenMs >= WORD_TRACKING_CLEAR_AGE_MS) return undefined;
  return { ...target, ready: target.raw.complete && nowMs - target.lastSeenMs <= WORD_TRACKING_CAPTURE_AGE_MS };
}

export function advanceWordTracking(state: WordTrackingState, groups: readonly WordTrackingGroup[], nowMs: number): WordTrackingState {
  'worklet';
  if (!Number.isFinite(nowMs) || (state.target && nowMs < state.target.lastSeenMs)) return state;
  let target: WordTrackedTarget | undefined = getWordTrackingTarget(state, nowMs);
  const ordered = [...groups].sort((a, b) => centerDistance(a.box) - centerDistance(b.box) || b.score - a.score);
  const associated = target && ordered.filter((group) => matches(target!.raw, group)).sort((a, b) => wordBoxIou(target!.raw.box, b.box) - wordBoxIou(target!.raw.box, a.box))[0];
  if (target && associated) {
    const sameCount = target.display.boxes.length === associated.boxes.length;
    target = {
      ...target, raw: associated, lastSeenMs: nowMs,
      display: { ...associated, box: sameCount ? smoothBox(target.display.box, associated.box) : associated.box, boxes: sameCount ? associated.boxes.map((box, index) => smoothBox(target!.display.boxes[index]!, box)) : associated.boxes },
    };
  }
  const nearest = ordered[0];
  const candidate = nearest && (!target || (!matches(target.raw, nearest) && centerDistance(nearest.box) < centerDistance(target.raw.box) * 0.8)) ? nearest : undefined;
  if (!candidate) return { target, observations: 0, nextId: state.nextId };
  const observations = state.pending && state.pendingSeenMs !== undefined && nowMs - state.pendingSeenMs <= WORD_TRACKING_CAPTURE_AGE_MS && matches(state.pending, candidate) ? state.observations + 1 : 1;
  if (observations < 2) return { target, pending: candidate, pendingSeenMs: nowMs, observations, nextId: state.nextId };
  return { target: { id: state.nextId, raw: candidate, display: candidate, lastSeenMs: nowMs }, observations: 0, nextId: state.nextId + 1 };
}
