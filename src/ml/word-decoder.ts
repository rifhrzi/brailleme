import {
  isValidWordModelThresholds,
  WORD_MAX_CELLS,
  WORD_MODEL_GRID_HEIGHT,
  WORD_MODEL_GRID_WIDTH,
  WORD_MODEL_INPUT_HEIGHT,
  WORD_MODEL_INPUT_WIDTH,
  WORD_MODEL_OUTPUT_CHANNELS,
  WORD_MODEL_OUTPUT_LENGTH,
  type WordModelThresholds,
} from '@/ml/word-model-contract';
import { parseWordMasks } from '@/ml/word-symbols';
import type { NormalizedBox } from '@/types/live-scan';
import { isNormalizedWordBox, type WordCell, type WordImageResult, type WordRejectionReason } from '@/types/word-scan';

export type WordProposal = {
  readonly score: number;
  readonly box: NormalizedBox;
  readonly offset: number;
  readonly mask: number;
  readonly confidence: number;
  readonly dotConfidence: number;
};

/** Also used for loader warmup; it asserts only the tensor contract. */
export function assertWordModelOutput(output: ArrayLike<number>): void {
  'worklet';
  if (output.length !== WORD_MODEL_OUTPUT_LENGTH) throw new Error('Dimensi output model kata tidak sesuai.');
  for (let index = 0; index < output.length; index += 1) {
    const probability = output[index];
    if (probability === undefined || !Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new Error('Output model kata harus berupa probabilitas yang valid.');
    }
  }
}

export function wordBoxIou(a: NormalizedBox, b: NormalizedBox): number {
  'worklet';
  const width = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const height = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const intersection = width * height;
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function median(values: readonly number[]): number {
  'worklet';
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function visibleBox(box: NormalizedBox): NormalizedBox | undefined {
  'worklet';
  const x = Math.max(0, Math.min(1, box.x));
  const y = Math.max(0, Math.min(1, box.y));
  const width = Math.min(1, box.x + box.width) - x;
  const height = Math.min(1, box.y + box.height) - y;
  return width > 0 && height > 0 ? { x, y, width, height } : undefined;
}

function rejected(reason: WordRejectionReason, proposals: readonly WordProposal[]): WordImageResult {
  'worklet';
  const boxes: NormalizedBox[] = [];
  for (const proposal of proposals) {
    const box = visibleBox(proposal.box);
    if (box) boxes.push(box);
  }
  return { status: 'rejected', reason, cells: [], boxes };
}

/** Position-only NMS keeps weak, indicator and unsupported cells for tracking. */
export function detectWordProposals(
  output: ArrayLike<number>,
  thresholds: WordModelThresholds,
  selection: NormalizedBox = { x: 0, y: 0, width: 1, height: 1 },
  projectedViewSelection = false,
): readonly WordProposal[] {
  'worklet';
  assertWordModelOutput(output);
  if (!isValidWordModelThresholds(thresholds)) throw new Error('Ambang model kata tidak valid.');
  // Center zoom may project the user's selection beyond the image. Only the
  // internal view matcher uses this; public decoding still requires normalized ROI.
  const validProjected = [selection.x, selection.y, selection.width, selection.height].every(Number.isFinite) && selection.width > 0 && selection.height > 0;
  if (projectedViewSelection ? !validProjected : !isNormalizedWordBox(selection)) throw new Error('Area pilihan kata tidak valid.');

  const proposals: WordProposal[] = [];
  for (let row = 0; row < WORD_MODEL_GRID_HEIGHT; row += 1) {
    for (let column = 0; column < WORD_MODEL_GRID_WIDTH; column += 1) {
      const offset = (row * WORD_MODEL_GRID_WIDTH + column) * WORD_MODEL_OUTPUT_CHANNELS;
      const score = output[offset]!;
      if (score < thresholds.proposalScore) continue;
      const width = output[offset + 3]!;
      const height = output[offset + 4]!;
      const box = {
          x: (column + output[offset + 1]!) / WORD_MODEL_GRID_WIDTH - width / 2,
          y: (row + output[offset + 2]!) / WORD_MODEL_GRID_HEIGHT - height / 2,
          width, height,
      };
      // Neighboring rows are useful model context but are not part of the selected word.
      // A proposal intersecting the guide is retained and later rejected if clipped.
      if (box.x + box.width <= selection.x || box.x >= selection.x + selection.width ||
          box.y + box.height <= selection.y || box.y >= selection.y + selection.height) continue;
      let mask = 0;
      let dotConfidence = 1;
      for (let dot = 0; dot < 6; dot += 1) {
        const probability = output[offset + 5 + dot]!;
        dotConfidence = Math.min(dotConfidence, Math.max(probability, 1 - probability));
        if (probability >= 0.5) mask |= 1 << dot;
      }
      proposals.push({ score, offset, box, mask, dotConfidence, confidence: Math.min(score, dotConfidence) });
    }
  }
  proposals.sort((a, b) => b.score - a.score || a.offset - b.offset);
  const retained: WordProposal[] = [];
  for (const proposal of proposals) {
    if (!retained.some((other) => wordBoxIou(proposal.box, other.box) > thresholds.nmsIou)) retained.push(proposal);
  }
  retained.sort((a, b) => (a.box.x + a.box.width / 2) - (b.box.x + b.box.width / 2));
  return retained;
}

/** Any uncertain physical cell, including an indicator, rejects the token. */
export function decodeWordOutput(
  output: ArrayLike<number>,
  thresholds: WordModelThresholds,
  selection: NormalizedBox = { x: 0, y: 0, width: 1, height: 1 },
): WordImageResult {
  'worklet';
  const retained = detectWordProposals(output, thresholds, selection);

  if (retained.length === 0) return rejected('no_grid', retained);
  if (retained.length > WORD_MAX_CELLS) return rejected('too_many_cells', retained);
  if (retained.some(({ box }) => box.width < thresholds.minCellWidth || box.height < thresholds.minCellHeight)) return rejected('too_far', retained);
  if (retained.some(({ box }) => box.x < thresholds.edgeMargin || box.y < thresholds.edgeMargin ||
      box.x + box.width > 1 - thresholds.edgeMargin || box.y + box.height > 1 - thresholds.edgeMargin ||
      box.x < selection.x || box.y < selection.y ||
      box.x + box.width > selection.x + selection.width || box.y + box.height > selection.y + selection.height)) return rejected('clipped', retained);
  if (retained.some(({ score }) => score < thresholds.acceptScore)) return rejected('low_confidence', retained);

  const medianHeight = median(retained.map(({ box }) => box.height));
  const centerYs = retained.map(({ box }) => box.y + box.height / 2);
  if (Math.max(...centerYs) - Math.min(...centerYs) > thresholds.maxRowDeviation * medianHeight) return rejected('multiple_rows', retained);
  for (let index = 1; index < retained.length; index += 1) {
    const previous = retained[index - 1]!.box;
    const current = retained[index]!.box;
    const centerGap = (current.x + current.width / 2 - previous.x - previous.width / 2) * WORD_MODEL_INPUT_WIDTH;
    if (centerGap > thresholds.maxGapRatio * medianHeight * WORD_MODEL_INPUT_HEIGHT) return rejected('gap', retained);
    const intersection = Math.max(0, Math.min(previous.x + previous.width, current.x + current.width) - Math.max(previous.x, current.x));
    if (centerGap <= 0 || intersection > thresholds.maxOverlapRatio * Math.min(previous.width, current.width)) return rejected('overlapping_cells', retained);
  }

  if (retained.some(({ dotConfidence }) => dotConfidence < thresholds.dotConfidence)) return rejected('low_confidence', retained);
  const parsed = parseWordMasks(retained.map(({ mask }) => mask));
  if (parsed.status === 'rejected') return rejected(parsed.reason, retained);
  const cells: WordCell[] = retained.map((proposal, index) => ({
    ...parsed.symbols[index]!, confidence: proposal.confidence, box: proposal.box,
  }));
  return {
    status: 'success',
    text: parsed.text,
    confidence: Math.min(...cells.map(({ confidence }) => confidence)),
    cells,
    boxes: cells.map(({ box }) => box),
  };
}
