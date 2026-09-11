/** Port of the frozen v5 NumPy runtime. Routing has no reference text or boxes. */
import { assertWordModelOutput, decodeWordOutput, detectWordProposals, type WordProposal } from '@/ml/word-decoder';
import {
  WORD_MODEL_INPUT_HEIGHT as H, WORD_MODEL_INPUT_WIDTH as W, WORD_MODEL_INPUT_LENGTH,
  WORD_MODEL_GRID_HEIGHT as GH, WORD_MODEL_GRID_WIDTH as GW, WORD_MODEL_OUTPUT_LENGTH,
  type WordModelThresholds,
} from '@/ml/word-model-contract';
import type { NormalizedBox } from '@/types/live-scan';
import type { WordImageResult } from '@/types/word-scan';

export type WordV5View = 'identity' | 'horizontal' | 'vertical' | 'half_turn' | 'zoom125' | 'zoom150';
export type WordV5Route = 'identity' | 'matched_logit_four' | 'matched_prob_four' | 'matched_prob_vertical' |
  'matched_prob_half_turn' | 'matched_prob_horizontal' | 'zoom_logit_mean' | 'zoom_probability_mean';
export const WORD_V5_VIEWS: readonly WordV5View[] = ['identity', 'horizontal', 'vertical', 'half_turn', 'zoom125', 'zoom150'];
const FLIPS = ['horizontal', 'vertical', 'half_turn'] as const;
type Tensor = Float32Array<ArrayBuffer>;
export type WordV5Predict = (input: Tensor) => Tensor;
export interface WordV5Result {
  readonly result: WordImageResult;
  readonly route: WordV5Route;
  /** Final dots at ORIGINAL detector offsets; geometry and objectness unchanged. */
  readonly output: Tensor;
  readonly forwardCount: number;
}

/** Float32 operations follow NumPy's four ordered bilinear accumulations. */
export function transformWordV5Image(input: Tensor, view: WordV5View): Tensor {
  'worklet';
  if (input.length !== WORD_MODEL_INPUT_LENGTH) throw new Error('Dimensi pixel v5 tidak sesuai.');
  if (view === 'identity') return input;
  const result = new Float32Array(WORD_MODEL_INPUT_LENGTH);
  if (view === 'zoom125' || view === 'zoom150') {
    const scale = view === 'zoom125' ? 1.25 : 1.5;
    // Both frozen scales zoom inward: all four samples are inside the image.
    // Therefore the reference RGB-mean exterior fill is never sampled.
    const x0 = new Int32Array(W), y0 = new Int32Array(H);
    const wx = new Float32Array(W), wy = new Float32Array(H);
    for (let x = 0; x < W; x += 1) {
      const coordinate = (x + 0.5 - W / 2) / scale + W / 2 - 0.5;
      x0[x] = Math.floor(coordinate); wx[x] = coordinate - x0[x]!;
    }
    for (let y = 0; y < H; y += 1) {
      const coordinate = (y + 0.5 - H / 2) / scale + H / 2 - 0.5;
      y0[y] = Math.floor(coordinate); wy[y] = coordinate - y0[y]!;
    }
    for (let dx = 0; dx < 2; dx += 1) {
      for (let dy = 0; dy < 2; dy += 1) {
        for (let y = 0; y < H; y += 1) {
          const yw = dy ? wy[y]! : Math.fround(1 - wy[y]!);
          for (let x = 0; x < W; x += 1) {
            const xw = dx ? wx[x]! : Math.fround(1 - wx[x]!);
            const weight = Math.fround(yw * xw);
            const source = ((y0[y]! + dy) * W + x0[x]! + dx) * 3;
            const target = (y * W + x) * 3;
            for (let channel = 0; channel < 3; channel += 1) {
              result[target + channel] = result[target + channel]! + Math.fround(input[source + channel]! * weight);
            }
          }
        }
      }
    }
    return result;
  }
  const horizontal = view !== 'vertical', vertical = view !== 'horizontal';
  for (let y = 0; y < H; y += 1) {
    for (let x = 0; x < W; x += 1) {
      const source = ((vertical ? H - 1 - y : y) * W + (horizontal ? W - 1 - x : x)) * 3;
      const target = (y * W + x) * 3;
      result[target] = input[source]!; result[target + 1] = input[source + 1]!; result[target + 2] = input[source + 2]!;
    }
  }
  return result;
}

export function invertWordV5Output(output: Tensor, view: typeof FLIPS[number]): Tensor {
  'worklet';
  assertWordModelOutput(output);
  const horizontal = view !== 'vertical', vertical = view !== 'horizontal';
  const bits = view === 'horizontal' ? [3, 4, 5, 0, 1, 2] : view === 'vertical' ? [2, 1, 0, 5, 4, 3] : [5, 4, 3, 2, 1, 0];
  const result = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
  for (let y = 0; y < GH; y += 1) {
    for (let x = 0; x < GW; x += 1) {
      const source = ((vertical ? GH - 1 - y : y) * GW + (horizontal ? GW - 1 - x : x)) * 11;
      const target = (y * GW + x) * 11;
      result[target] = output[source]!;
      result[target + 1] = horizontal ? 1 - output[source + 1]! : output[source + 1]!;
      result[target + 2] = vertical ? 1 - output[source + 2]! : output[source + 2]!;
      result[target + 3] = output[source + 3]!; result[target + 4] = output[source + 4]!;
      for (let bit = 0; bit < 6; bit += 1) result[target + 5 + bit] = output[source + 5 + bits[bit]!]!;
    }
  }
  return result;
}

type DotCell = { readonly offset: number; readonly box: NormalizedBox; readonly dots: readonly number[] };
function dotCells(output: Tensor, proposals: readonly WordProposal[]): DotCell[] {
  'worklet';
  return proposals.map(({ offset, box }) => ({ offset, box, dots: Array.from(output.slice(offset + 5, offset + 11)) }));
}

function projectBox(box: NormalizedBox, scale: number): NormalizedBox {
  'worklet';
  const x = (box.x - 0.5) * scale + 0.5;
  const y = (box.y - 0.5) * scale + 0.5;
  return { x, y, width: (box.x + box.width - 0.5) * scale + 0.5 - x, height: (box.y + box.height - 0.5) * scale + 0.5 - y };
}

function complete(box: NormalizedBox, bounds: NormalizedBox): boolean {
  'worklet';
  return box.x >= bounds.x && box.y >= bounds.y && box.x + box.width <= bounds.x + bounds.width && box.y + box.height <= bounds.y + bounds.height;
}

function inverseZoomCells(output: Tensor, scale: number, thresholds: WordModelThresholds, selection: NormalizedBox): DotCell[] {
  'worklet';
  const projected = projectBox(selection, scale), margin = thresholds.edgeMargin;
  const x = Math.max(margin, projected.x), y = Math.max(margin, projected.y);
  const bounds = { x, y, width: Math.min(1 - margin, projected.x + projected.width) - x, height: Math.min(1 - margin, projected.y + projected.height) - y };
  return dotCells(output, detectWordProposals(output, thresholds, projected, true))
    .filter((cell) => complete(cell.box, bounds))
    .map((cell) => {
      // Use division, not multiplication by a reciprocal, matching NumPy.
      const x = (cell.box.x - 0.5) / scale + 0.5, y = (cell.box.y - 0.5) / scale + 0.5;
      return { ...cell, box: { x, y, width: (cell.box.x + cell.box.width - 0.5) / scale + 0.5 - x, height: (cell.box.y + cell.box.height - 0.5) / scale + 0.5 - y } };
    });
}

/** Matching uses xyxy-derived areas like the reference, distinct from detector NMS. */
function matchIou(a: NormalizedBox, b: NormalizedBox): number {
  'worklet';
  const width = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const height = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const overlap = width * height;
  const total = Math.max(0, a.x + a.width - a.x) * Math.max(0, a.y + a.height - a.y) + Math.max(0, b.x + b.width - b.x) * Math.max(0, b.y + b.height - b.y);
  return total > overlap ? overlap / (total - overlap) : 0;
}

function combineDots(base: Tensor, baseCells: readonly DotCell[], views: readonly { readonly cells: readonly DotCell[]; readonly scale?: number }[], thresholds: WordModelThresholds, logits: boolean): Tensor {
  'worklet';
  const gathered = baseCells.map((cell) => [cell.dots]);
  const margin = thresholds.edgeMargin;
  const bounds = { x: margin, y: margin, width: 1 - margin * 2, height: 1 - margin * 2 };
  for (const view of views) {
    const pairs: { overlap: number; base: number; other: number }[] = [];
    for (let i = 0; i < baseCells.length; i += 1) {
      if (view.scale && !complete(projectBox(baseCells[i]!.box, view.scale), bounds)) continue;
      for (let j = 0; j < view.cells.length; j += 1) {
        const overlap = matchIou(baseCells[i]!.box, view.cells[j]!.box);
        if (overlap >= 0.5) pairs.push({ overlap, base: i, other: j });
      }
    }
    pairs.sort((a, b) => b.overlap - a.overlap || a.base - b.base || a.other - b.other);
    const usedBase: number[] = [], usedOther: number[] = [];
    for (const pair of pairs) {
      if (pair.overlap < 0.5) break;
      if (usedBase.includes(pair.base) || usedOther.includes(pair.other)) continue;
      gathered[pair.base]!.push(view.cells[pair.other]!.dots);
      usedBase.push(pair.base); usedOther.push(pair.other);
    }
  }
  const result = new Float32Array(base);
  for (let i = 0; i < baseCells.length; i += 1) {
    const parts = gathered[i]!;
    if (parts.length === 1) continue;
    for (let bit = 0; bit < 6; bit += 1) {
      let sum = 0;
      for (const part of parts) {
        const value = logits ? Math.max(1e-6, Math.min(1 - 1e-6, part[bit]!)) : part[bit]!;
        sum += logits ? Math.log(value / (1 - value)) : value;
      }
      const mean = sum / parts.length;
      result[baseCells[i]!.offset + 5 + bit] = logits ? 1 / (1 + Math.exp(-mean)) : mean;
    }
  }
  return result;
}

/** Called after the original image quality gate. getOutput lazily owns inference. */
export function decodeWordV5Views(getOutput: (view: WordV5View) => Tensor, thresholds: WordModelThresholds,
  selection: NormalizedBox = { x: 0, y: 0, width: 1, height: 1 }, qualityRejected = false): WordV5Result {
  'worklet';
  if (qualityRejected) {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    return { result: decodeWordOutput(output, thresholds, selection), route: 'identity', output, forwardCount: 0 };
  }
  // Snapshot: native predictors may reuse their output buffer on the next call.
  const base = new Float32Array(getOutput('identity'));
  const original = decodeWordOutput(base, thresholds, selection);
  let forwardCount = 1;
  const identity: WordV5Result = { result: original, route: 'identity', output: base, forwardCount };
  if (original.status === 'success') return identity;
  const proposals = detectWordProposals(base, thresholds, selection);
  // Dots alone can never change these rejections. Bound mobile work before
  // pairing noisy detections; this preserves the frozen result and route.
  if (['too_many_cells', 'too_far', 'clipped', 'multiple_rows', 'gap', 'overlapping_cells'].includes(original.reason) ||
      proposals.some((cell) => cell.score < thresholds.acceptScore)) return identity;
  const baseCells = dotCells(base, proposals);
  if (!baseCells.length) return identity;
  const flips = FLIPS.map((view) => {
    const output = invertWordV5Output(getOutput(view), view);
    forwardCount += 1;
    return { cells: dotCells(output, detectWordProposals(output, thresholds, selection)) };
  });
  const routes: readonly WordV5Route[] = ['matched_logit_four', 'matched_prob_four', 'matched_prob_vertical', 'matched_prob_half_turn', 'matched_prob_horizontal'];
  for (let index = 0; index < routes.length; index += 1) {
    const views = index < 2 ? flips : [flips[index === 2 ? 1 : index === 3 ? 2 : 0]!];
    const output = combineDots(base, baseCells, views, thresholds, index === 0);
    const result = decodeWordOutput(output, thresholds, selection);
    if (result.status === 'success') return { result, route: routes[index]!, output, forwardCount };
  }
  const zooms = [{ view: 'zoom125' as const, scale: 1.25 }, { view: 'zoom150' as const, scale: 1.5 }].map(({ view, scale }) => {
    const cells = inverseZoomCells(getOutput(view), scale, thresholds, selection);
    forwardCount += 1;
    return { cells, scale };
  });
  for (const logits of [true, false]) {
    const output = combineDots(base, baseCells, zooms, thresholds, logits);
    const result = decodeWordOutput(output, thresholds, selection);
    if (result.status === 'success') return { result, route: logits ? 'zoom_logit_mean' : 'zoom_probability_mean', output, forwardCount };
  }
  return { ...identity, forwardCount };
}

/** Background-worklet prediction of the fixed selected policy; no image copies persist. */
export function recognizeWordV5Tensor(input: Tensor, predict: WordV5Predict, thresholds: WordModelThresholds,
  selection?: NormalizedBox, identityOutput?: Tensor): WordV5Result {
  'worklet';
  return decodeWordV5Views((view) => {
    'worklet';
    return view === 'identity' && identityOutput ? identityOutput : predict(transformWordV5Image(input, view));
  }, thresholds, selection);
}
