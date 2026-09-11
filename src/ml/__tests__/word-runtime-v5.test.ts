/* Node I/O is restricted to this Jest evidence harness; no Node globals in app types. */
/* eslint-disable @typescript-eslint/no-require-imports */
const { createHash } = require('node:crypto') as { createHash(name: string): { update(value: Uint8Array | string): { digest(encoding: 'hex'): string } } };
const { mkdirSync, readFileSync, writeFileSync } = require('node:fs') as {
  mkdirSync(path: string, options: { recursive: boolean }): void;
  readFileSync(path: string): Uint8Array;
  readFileSync(path: string, encoding: 'utf8'): string;
  writeFileSync(path: string, content: string): void;
};
const path = require('node:path') as { resolve(path: string): string; dirname(path: string): string };
/* eslint-enable @typescript-eslint/no-require-imports */

import { detectWordProposals } from '@/ml/word-decoder';
import { WORD_DECODER_VERSION, WORD_RUNTIME_MODULE, WORD_RUNTIME_POLICY, WORD_MODEL_INPUT_LENGTH, WORD_MODEL_OUTPUT_LENGTH, type WordModelThresholds } from '@/ml/word-model-contract';
import { decodeWordV5Views, invertWordV5Output, transformWordV5Image, type WordV5View } from '@/ml/word-runtime-v5';
import type { NormalizedBox } from '@/types/live-scan';

const fixtureFile = 'src/ml/__tests__/fixtures/word-v5-runtime-reference.json';
type Signature = [string, string, [number | null, number, string | null, string | null][]];
type Reference = {
  scope: string; runtimeModule: string; modelSha256: string; methodSha256: string; selectedPolicy: string; decoderVersion: string;
  thresholds: WordModelThresholds; sampleCount: number; rawViews: number; views: WordV5View[]; oracleSha256: string; oracleMetadataSha256: string;
  validationSha256: string; referenceSourceSha256: Record<string, string>; transformProbe: { sha256: Record<WordV5View, string> }; qualityRejectedCount: number;
  cases: { source: string; expected: string; group: string; selection: number[] | null; qualityRejected: boolean;
    outputs: Record<WordV5View, number[][]>; signature: Signature; route: string }[];
};
const reference = JSON.parse(readFileSync(path.resolve(fixtureFile), 'utf8')) as Reference;
const thresholds = reference.thresholds;
const hash = (value: Uint8Array | string) => createHash('sha256').update(value).digest('hex');
const fileHash = (name: string) => hash(readFileSync(path.resolve(name)));

function tensor(cells: number[][]): Float32Array<ArrayBuffer> {
  const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
  for (const [offset, ...values] of cells) output.set(values, offset!);
  return output;
}

function addCell(output: Float32Array, x: number, dots: number[], width = 0.06, height = 0.25) {
  const column = Math.floor(x * 120), offset = (24 * 120 + column) * 11;
  output.set([0.95, x * 120 - column, 0, width, height, ...dots], offset);
  return offset;
}

describe('frozen six-view v5 mobile runtime', () => {
  it('matches all 540 validation decisions and every float32 pixel in six deterministic transformed views', () => {
    expect(reference.scope).toBe('VALIDATION');
    expect(reference.runtimeModule).toBe(WORD_RUNTIME_MODULE);
    expect(reference.selectedPolicy).toBe(WORD_RUNTIME_POLICY);
    expect(reference.decoderVersion).toBe(WORD_DECODER_VERSION);
    expect(reference.sampleCount).toBe(540);
    expect(reference.cases).toHaveLength(540);
    expect(reference.rawViews).toBe(6);
    expect(reference.qualityRejectedCount).toBe(8);
    for (const [name, expected] of Object.entries(reference.referenceSourceSha256)) {
      expect(fileHash(`training/evidence/braille-word-reader-id-v5-20260911/source/${name}`)).toBe(expected);
    }
    const pixels = new Float32Array(WORD_MODEL_INPUT_LENGTH);
    for (let index = 0; index < pixels.length; index += 1) pixels[index] = ((index * 37) % 251) / 250;
    for (const view of reference.views) {
      const transformed = transformWordV5Image(pixels, view);
      expect(hash(new Uint8Array(transformed.buffer))).toBe(reference.transformProbe.sha256[view]);
    }
    const mismatches = { text: 0, reason: 0, masks: 0, roles: 0, outputs: 0, route: 0 };
    const failures: unknown[] = [];
    const routes: Record<string, number> = {};
    let exactCorrect = 0;
    for (const item of reference.cases) {
      const selection: NormalizedBox | undefined = item.selection ? {
        x: item.selection[0]!, y: item.selection[1]!, width: item.selection[2]! - item.selection[0]!, height: item.selection[3]! - item.selection[1]!,
      } : undefined;
      const calls: WordV5View[] = [];
      const identity = item.qualityRejected ? new Float32Array(WORD_MODEL_OUTPUT_LENGTH) : tensor(item.outputs.identity);
      const actual = decodeWordV5Views((view) => { calls.push(view); return view === 'identity' ? identity : tensor(item.outputs[view]); }, thresholds, selection, item.qualityRejected);
      const proposals = detectWordProposals(actual.output, thresholds, selection);
      const result = actual.result;
      const actualText = result.status === 'success' ? result.text : '';
      const actualReason = result.status === 'success' ? '' : result.reason;
      const actualMasks = proposals.map((cell) => cell.mask);
      const actualRoles = proposals.map((_, index) => result.status === 'success' ? result.cells[index]!.role : null);
      const actualOutputs = proposals.map((_, index) => result.status === 'success' ? result.cells[index]!.output : null);
      const expected = item.signature;
      const checks = {
        text: actualText === expected[0], reason: actualReason === expected[1],
        masks: JSON.stringify(actualMasks) === JSON.stringify(expected[2].map((cell) => cell[1])),
        roles: JSON.stringify(actualRoles) === JSON.stringify(expected[2].map((cell) => cell[2])),
        outputs: JSON.stringify(actualOutputs) === JSON.stringify(expected[2].map((cell) => cell[3])),
        route: actual.route === item.route,
      };
      for (const key of Object.keys(mismatches) as (keyof typeof mismatches)[]) if (!checks[key]) mismatches[key] += 1;
      if (Object.values(checks).some((value) => !value)) failures.push({ source: item.source, checks, actual: [actualText, actualReason, actualMasks, actual.route], expected });
      const immutableRejection = result.status !== 'success' && (['too_many_cells', 'too_far', 'clipped', 'multiple_rows', 'gap', 'overlapping_cells'].includes(result.reason) || proposals.some((cell) => cell.score < thresholds.acceptScore));
      const count = item.qualityRejected ? 0 : actual.route === 'identity' && (result.status === 'success' || !proposals.length || immutableRejection) ? 1 : actual.route.startsWith('zoom_') || result.status !== 'success' ? 6 : 4;
      expect(calls).toEqual(reference.views.slice(0, count));
      expect(actual.forwardCount).toBe(calls.length);
      for (let offset = 0; offset < actual.output.length; offset += 11) {
        for (let channel = 0; channel < 5; channel += 1) {
          if (actual.output[offset + channel] !== identity[offset + channel]) throw new Error(`Original geometry changed at ${item.source}:${offset + channel}`);
        }
      }
      if (item.expected && actualText === item.expected) exactCorrect += 1;
      routes[actual.route] = (routes[actual.route] ?? 0) + 1;
    }
    expect(failures.slice(0, 8)).toEqual([]);
    expect(mismatches).toEqual({ text: 0, reason: 0, masks: 0, roles: 0, outputs: 0, route: 0 });
    expect(exactCorrect).toBe(407);
    // Opt-in evidence creation; ordinary tests never alter evidence or manifests.
    if (process.env.WORD_V5_PARITY_REPORT) {
      const sourceFiles = ['src/ml/word-runtime-v5.ts', 'src/ml/word-symbols.ts', 'src/ml/word-decoder.ts',
        'src/ml/word-image-processing.ts', 'src/ml/word-model-contract.ts', 'src/types/word-scan.ts'];
      const report = {
        version: 1, createdUtc: new Date().toISOString(), parityPassed: true, scope: reference.scope,
        runtimeModule: reference.runtimeModule, modelSha256: reference.modelSha256, methodSha256: reference.methodSha256,
        selectedPolicy: reference.selectedPolicy, decoderVersion: reference.decoderVersion, thresholds,
        sampleCount: reference.sampleCount, rawViews: reference.views, views: reference.views, mismatches,
        fixtureFile, fixtureSha256: fileHash(fixtureFile), oracleSha256: reference.oracleSha256,
        oracleMetadataSha256: reference.oracleMetadataSha256, validationSha256: reference.validationSha256,
        sourceSha256: Object.fromEntries(sourceFiles.map((name) => [name, fileHash(name)])),
        testFile: 'src/ml/__tests__/word-runtime-v5.test.ts', testSha256: fileHash('src/ml/__tests__/word-runtime-v5.test.ts'),
        transformParity: { passed: true, fullPixelsPerView: WORD_MODEL_INPUT_LENGTH, views: reference.views, sha256: reference.transformProbe.sha256 },
        originalGeometryPreserved: true, originalObjectnessPreserved: true, fixedLazyRoutingVerified: true,
        optimizations: ['Skip fallback for immutable geometry/objectness rejections', 'Allocate matching pairs only at IoU >= 0.5', 'Skip inference on original quality failure'],
        qualityBoundary: 'Eight original quality failures use canonical no_grid only in runtime parity; actual image processing preserves existing dark/glare/low_confidence user feedback and skips model execution.',
        validationExactCorrect: exactCorrect, validationPositiveTotal: 500, routingCounts: routes,
        physicalCameraValidated: false, nativeRuntimeValidated: false,
        limitation: 'CPU TypeScript replay of all frozen old-validation Torch raw views, whose TFLite decisions were separately qualified; no phone camera accuracy or Android latency claim.',
      };
      const destination = path.resolve(process.env.WORD_V5_PARITY_REPORT);
      mkdirSync(path.dirname(destination), { recursive: true });
      writeFileSync(destination, `${JSON.stringify(report, null, 2)}\n`);
    }
  }, 120000);

  it('requests only identity when accepted, and preserves an empty detector rejection', () => {
    for (const present of [true, false]) {
      const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
      if (present) addCell(output, 0.5, [0.99, 0.01, 0.01, 0.01, 0.01, 0.01]);
      const predict = jest.fn(() => output);
      const result = decodeWordV5Views(predict, thresholds);
      expect(predict).toHaveBeenCalledTimes(1);
      expect(result.route).toBe('identity');
      expect(result.result).toMatchObject(present ? { text: 'a' } : { reason: 'no_grid' });
    }
  });

  it('inverts physical dot slots and subcell geometry exactly once for each flip', () => {
    const output = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    const offset = addCell(output, 0.25, [0.91, 0.82, 0.73, 0.24, 0.15, 0.06]);
    output[offset + 1] = 0.25; output[offset + 2] = 0.75;
    for (const view of ['horizontal', 'vertical', 'half_turn'] as const) {
      const transformed = invertWordV5Output(output, view);
      const restored = invertWordV5Output(transformed, view);
      expect(Array.from(restored.slice(offset, offset + 11))).toEqual(Array.from(output.slice(offset, offset + 11)));
    }
  });

  it('never rescues a projected clipped original cell using an apparently complete zoom prediction', () => {
    const base = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(base, 0.05, [0.6, 0.01, 0.01, 0.01, 0.01, 0.01], 0.06);
    const zoom = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(zoom, 0.04, [0.99, 0.01, 0.01, 0.01, 0.01, 0.01], 0.06);
    const empty = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    const result = decodeWordV5Views((view) => view === 'identity' ? base : view.startsWith('zoom') ? zoom : empty, thresholds);
    expect(result.result).toMatchObject({ status: 'rejected', reason: 'low_confidence' });
    expect(result.route).toBe('identity');
    expect(result.output).toEqual(base);
  });

  it('preserves a geometry rejection even when every transformed view has confident letters', () => {
    const base = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(base, 0.01, [0.6, 0.01, 0.01, 0.01, 0.01, 0.01]);
    const view = new Float32Array(WORD_MODEL_OUTPUT_LENGTH);
    addCell(view, 0.5, [0.99, 0.01, 0.01, 0.01, 0.01, 0.01]);
    const result = decodeWordV5Views((name) => name === 'identity' ? base : view, thresholds);
    expect(result.result).toMatchObject({ status: 'rejected', reason: 'clipped' });
    expect(result.route).toBe('identity');
  });
});
