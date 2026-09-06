import manifest from '../../assets/models/braille-word-reader-v1/manifest.json';
import { WORD_DECODER_VERSION } from '@/ml/word-model-contract';

// Literal Metro assets are added only after the new model passes its benchmark.
// A missing candidate must remain a normal unavailable state, not a broken import.
export const WORD_MODEL_ASSET: number | null = null;
export const WORD_NATIVE_FIXTURE_ASSET: number | null = null;
export const WORD_MODEL_MANIFEST = manifest;
export const WORD_MODEL_VERSION = manifest.artifact.version;
export const WORD_MODEL_BUNDLED = WORD_MODEL_ASSET !== null &&
  WORD_NATIVE_FIXTURE_ASSET !== null && manifest.internalAcceptance.passed &&
  manifest.decoderVersion === WORD_DECODER_VERSION &&
  ['unverified_candidate', 'verified'].includes(manifest.artifact.status);

// The word reader is a new pipeline; classifier-v2 evidence cannot verify it.
export const WORD_MODEL_RELEASE_VERIFIED = false;
