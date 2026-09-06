import {
  isValidWordModelThresholds,
  WORD_LATIN_MASKS,
  WORD_MAX_LETTERS,
  WORD_MODEL_GRID_HEIGHT,
  WORD_MODEL_GRID_WIDTH,
  WORD_MODEL_INPUT_HEIGHT,
  WORD_MODEL_INPUT_WIDTH,
  WORD_MODEL_OUTPUT_CHANNELS,
  WORD_MODEL_OUTPUT_LENGTH,
  type WordModelThresholds,
} from '@/ml/word-model-contract';
import type { NormalizedBox } from '@/types/live-scan';
import { isNormalizedWordBox, type WordCell, type WordImageResult, type WordRejectionReason } from '@/types/word-scan';

type Proposal = {
  readonly score: number;
  readonly box: NormalizedBox;
  readonly offset: number;
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

function rejected(reason: WordRejectionReason, proposals: readonly Proposal[]): WordImageResult {
  'worklet';
  const boxes: NormalizedBox[] = [];
  for (const proposal of proposals) {
    const box = visibleBox(proposal.box);
    if (box) boxes.push(box);
  }
  return { status: 'rejected', reason, cells: [], boxes };
}

/** Pure six-dot decoder. Any retained uncertain cell rejects the entire word. */
export function decodeWordOutput(
  output: ArrayLike<number>,
  thresholds: WordModelThresholds,
  selection: NormalizedBox = { x: 0, y: 0, width: 1, height: 1 },
): WordImageResult {
  'worklet';
  assertWordModelOutput(output);
  if (!isValidWordModelThresholds(thresholds)) throw new Error('Ambang model kata tidak valid.');
  if (!isNormalizedWordBox(selection)) throw new Error('Area pilihan kata tidak valid.');

  const proposals: Proposal[] = [];
  for (let row = 0; row < WORD_MODEL_GRID_HEIGHT; row += 1) {
    for (let column = 0; column < WORD_MODEL_GRID_WIDTH; column += 1) {
      const offset = (row * WORD_MODEL_GRID_WIDTH + column) * WORD_MODEL_OUTPUT_CHANNELS;
      const score = output[offset]!;
      if (score < thresholds.proposalScore) continue;
      let isMaximum = true;
      for (let dy = -1; dy <= 1 && isMaximum; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const y = row + dy;
          const x = column + dx;
          if (y < 0 || y >= WORD_MODEL_GRID_HEIGHT || x < 0 || x >= WORD_MODEL_GRID_WIDTH || (dx === 0 && dy === 0)) continue;
          const neighborOffset = (y * WORD_MODEL_GRID_WIDTH + x) * WORD_MODEL_OUTPUT_CHANNELS;
          const neighborScore = output[neighborOffset]!;
          if (neighborScore > score || (neighborScore === score && neighborOffset < offset)) {
            isMaximum = false;
            break;
          }
        }
      }
      if (!isMaximum) continue;
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
      proposals.push({ score, offset, box });
    }
  }
  proposals.sort((a, b) => b.score - a.score || a.offset - b.offset);
  const retained: Proposal[] = [];
  for (const proposal of proposals) {
    if (!retained.some((other) => wordBoxIou(proposal.box, other.box) > thresholds.nmsIou)) retained.push(proposal);
  }
  retained.sort((a, b) => (a.box.x + a.box.width / 2) - (b.box.x + b.box.width / 2));

  if (retained.length === 0) return rejected('no_grid', retained);
  if (retained.length > WORD_MAX_LETTERS) return rejected('too_many_cells', retained);
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

  const cells: WordCell[] = [];
  for (const proposal of retained) {
    let mask = 0;
    let confidence = proposal.score;
    for (let dot = 0; dot < 6; dot += 1) {
      const probability = output[proposal.offset + 5 + dot]!;
      const selectedProbability = Math.max(probability, 1 - probability);
      if (selectedProbability < thresholds.dotConfidence) return rejected('low_confidence', retained);
      if (probability >= 0.5) mask |= 1 << dot;
      confidence = Math.min(confidence, selectedProbability);
    }
    const letterIndex = WORD_LATIN_MASKS.findIndex((candidate) => candidate === mask);
    if (letterIndex < 0) return rejected('unsupported_pattern', retained);
    cells.push({ letter: String.fromCharCode(65 + letterIndex), mask, confidence, box: proposal.box });
  }
  return {
    status: 'success',
    text: cells.map(({ letter }) => letter).join(''),
    confidence: Math.min(...cells.map(({ confidence }) => confidence)),
    cells,
    boxes: cells.map(({ box }) => box),
  };
}
