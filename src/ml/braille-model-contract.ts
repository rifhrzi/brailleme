import labelsJson from '../../assets/models/braille-cell-classifier-v2/label.json';
import modelManifestJson from '../../assets/models/braille-cell-classifier-v2/manifest.json';

export const BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS = Object.freeze([
  'tensorContractInspected',
  'labelCountMatchesOutput',
  'labelOrderConfirmedAgainstTrainingCode',
  'preprocessingConfirmedAgainstTrainingCode',
  'kerasTfliteConversionParityPassed',
  'nativePostCropFixturePassed',
  'nativeCameraGoldenParityPassed',
  'independentPhysicalHoldoutPassed',
  'negativeCorpusPassed',
  'deviceBenchmarkPassed',
  'datasetRedistributionRightsConfirmed',
] as const);

type ModelReleaseManifest = {
  readonly artifact?: {
    readonly status?: string;
    readonly version?: string;
  };
  readonly nativeSelfTest?: {
    readonly expectedClassIndex?: number;
    readonly expectedLabel?: string;
    readonly minimumConfidence?: number;
  };
  readonly verification?: Readonly<Record<string, boolean>>;
};

const modelReleaseManifest = modelManifestJson as ModelReleaseManifest;
const modelVerification = modelReleaseManifest.verification;
const modelVerificationKeys = Object.keys(modelVerification ?? {});
const hasExactVerificationSchema =
  modelVerification !== null &&
  typeof modelVerification === 'object' &&
  !Array.isArray(modelVerification) &&
  modelVerificationKeys.length ===
    BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS.length &&
  modelVerificationKeys.every((key) =>
    BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS.includes(
      key as (typeof BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS)[number],
    ),
  );

const modelVersion = modelReleaseManifest.artifact?.version;
if (typeof modelVersion !== 'string' || modelVersion.trim().length === 0) {
  throw new Error('Versi model Braille tidak tersedia pada manifest.');
}

export const BRAILLE_MODEL_VERSION = modelVersion;
export const BRAILLE_DECODER_VERSION = 'latin-letter-labels-v1';
export const BRAILLE_MODEL_INPUT_WIDTH = 28;
export const BRAILLE_MODEL_INPUT_HEIGHT = 28;
export const BRAILLE_MODEL_INPUT_CHANNELS = 3;
export const BRAILLE_MODEL_INPUT_LENGTH =
  BRAILLE_MODEL_INPUT_WIDTH *
  BRAILLE_MODEL_INPUT_HEIGHT *
  BRAILLE_MODEL_INPUT_CHANNELS;
export const BRAILLE_MODEL_OUTPUT_LENGTH = 26;
export const BRAILLE_MODEL_RELEASE_VERIFIED =
  modelReleaseManifest.artifact?.status === 'verified' &&
  hasExactVerificationSchema &&
  BRAILLE_MODEL_REQUIRED_VERIFICATION_KEYS.every(
    (key) => modelVerification?.[key] === true,
  );

/** Kontrak preprocessing yang dikonfirmasi terhadap script training kandidat aktif. */
export const BRAILLE_MODEL_INPUT_SCALE = 'zero_to_one' as const;

export type BrailleInputScale =
  | typeof BRAILLE_MODEL_INPUT_SCALE
  | 'zero_to_255';

export const BRAILLE_LABELS = Object.freeze([...labelsJson]);
const EXPECTED_BRAILLE_LABELS = 'abcdefghijklmnopqrstuvwxyz';

if (BRAILLE_LABELS.length !== BRAILLE_MODEL_OUTPUT_LENGTH) {
  throw new Error('Jumlah label Braille tidak sesuai dengan output model.');
}
if (BRAILLE_LABELS.join('') !== EXPECTED_BRAILLE_LABELS) {
  throw new Error('Urutan label Braille tidak sesuai dengan kontrak decoder.');
}

const nativeSelfTest = modelReleaseManifest.nativeSelfTest;
const nativeSelfTestIndex = nativeSelfTest?.expectedClassIndex;
const nativeSelfTestLabel = nativeSelfTest?.expectedLabel;
const nativeSelfTestMinimumConfidence = nativeSelfTest?.minimumConfidence;

if (
  !Number.isInteger(nativeSelfTestIndex) ||
  nativeSelfTestIndex === undefined ||
  nativeSelfTestIndex < 0 ||
  nativeSelfTestIndex >= BRAILLE_MODEL_OUTPUT_LENGTH ||
  typeof nativeSelfTestLabel !== 'string' ||
  BRAILLE_LABELS[nativeSelfTestIndex] !== nativeSelfTestLabel ||
  typeof nativeSelfTestMinimumConfidence !== 'number' ||
  !Number.isFinite(nativeSelfTestMinimumConfidence) ||
  nativeSelfTestMinimumConfidence < 0 ||
  nativeSelfTestMinimumConfidence > 1
) {
  throw new Error('Kontrak self-test native model tidak valid.');
}

export const BRAILLE_NATIVE_SELF_TEST_EXPECTED_INDEX = nativeSelfTestIndex;
export const BRAILLE_NATIVE_SELF_TEST_EXPECTED_LABEL = nativeSelfTestLabel;
export const BRAILLE_NATIVE_SELF_TEST_MIN_CONFIDENCE =
  nativeSelfTestMinimumConfidence;
