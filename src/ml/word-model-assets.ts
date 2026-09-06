import manifest from '../../assets/models/braille-word-reader-v1/manifest.json';
import { WORD_DECODER_VERSION } from '@/ml/word-model-contract';

// This explicitly authorized local experiment retains its failed accuracy gates.
// eslint-disable-next-line @typescript-eslint/no-require-imports -- Metro requires a literal binary asset.
export const WORD_MODEL_ASSET: number | null = require('../../assets/models/braille-word-reader-v1/word-reader.tflite');
export const WORD_NATIVE_FIXTURE_ASSET: number | null = require('../../assets/models/braille-word-reader-v1/native-word-fixture.png');
export const WORD_MODEL_MANIFEST = manifest;
export const WORD_MODEL_VERSION = manifest.artifact.version;
export const WORD_MODEL_EXPERIMENTAL = manifest.artifact.status === 'experimental_candidate' &&
  manifest.experimentalUse?.enabled === true &&
  manifest.experimentalUse.scope === 'local_internal_only' &&
  manifest.experimentalUse.authorizedBy === 'user' && !manifest.internalAcceptance.passed;
export const WORD_MODEL_BUNDLED = WORD_MODEL_ASSET !== null &&
  WORD_NATIVE_FIXTURE_ASSET !== null &&
  manifest.decoderVersion === WORD_DECODER_VERSION &&
  (WORD_MODEL_EXPERIMENTAL || (manifest.internalAcceptance.passed &&
    ['unverified_candidate', 'verified'].includes(manifest.artifact.status)));

// The word reader is a new pipeline; classifier-v2 evidence cannot verify it.
export const WORD_MODEL_RELEASE_VERIFIED = false;
