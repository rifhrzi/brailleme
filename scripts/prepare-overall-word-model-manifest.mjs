// Writes a reviewable manifest outside assets; activation is a separate action.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { V5_IDENTITY } from './verify-overall-word-model-assets.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)),'..');
const option = (name) => { const index = process.argv.indexOf(name); return index < 0 ? null : process.argv[index+1]; };
const destination = resolve(root, option('--output') ?? 'output/apk-id-v5-20260911/manifest.template.json');
const rel = relative(root,destination).replaceAll('\\','/');
assert.ok(rel && !rel.startsWith('..') && !isAbsolute(rel) && !rel.startsWith('assets/'), 'Prepare manifest outside assets only');
const allowPending = process.argv.includes('--allow-pending');
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const read = (file) => JSON.parse(readFileSync(resolve(root,file),'utf8'));
const proof = (file) => {
  if (!existsSync(resolve(root,file))) {
    assert.ok(allowPending, `Required evidence is not ready: ${file}`);
    return { file, sha256: 'PENDING_EVIDENCE' };
  }
  return { file, sha256: hash(readFileSync(resolve(root,file))) };
};
const archived = 'training/evidence/braille-word-reader-id-v5-20260911';
const checkpoint = 'training/checkpoints/braille-word-reader-id-v5';
const old = read('assets/models/braille-word-reader-v1/manifest.json');
const method = read(archived+'/final-method-selection.json');
const final = read(archived+'/final-test-evaluation.json');
const summary = final.results.primary.summary;
const smokeFile = option('--startup-proof') ?? 'output/apk-id-v5-20260911/startup-fixture-tflite.json';
const smoke = read(smokeFile);
const evidence = Object.fromEntries(Object.entries({ finalMethod:'final-method-selection.json',finalTest:'final-test-evaluation.json',
  confirmation:'final-test-torch-confirmation.json',protocol:'protocol.json',sourceSeparation:'source-separation-audit.json',
  conversion:'export-parity.json',originalConversion:'export-parity-original-failed.json',validationSelection:'tta-validation.json',
  regression:'historical-regression.json' }).map(([key,file]) => [key,proof(archived+'/'+file)]));
evidence.qualification = proof(checkpoint+'/qualification.json');
evidence.inferencePolicy = proof(checkpoint+'/inference-policy.json');
evidence.runtimeParity = proof(option('--runtime-parity') ?? 'output/qa/word-v5-runtime-parity.json');
evidence.nativeSelfTest = proof(smokeFile);
const model = readFileSync(resolve(root,checkpoint+'/word-reader.tflite'));
assert.equal(hash(model),V5_IDENTITY.model); assert.equal(evidence.finalMethod.sha256,V5_IDENTITY.method);
assert.equal(smoke.expected_text,'Yys'); assert.equal(smoke.model_sha256,V5_IDENTITY.model);
const manifest = {
  schemaVersion:1, experimentVersion:V5_IDENTITY.experiment, decoderVersion:V5_IDENTITY.decoder,
  artifact:{name:'braille-word-reader',version:'1.2.0-experimental.id-v5',status:'experimental_candidate',
    file:'word-reader.tflite',sha256:hash(model),sizeBytes:model.length},
  imageNormalization:old.imageNormalization,input:old.input,output:old.output,thresholds:method.thresholds,latinMasks:old.latinMasks,
  experimentalUse:{enabled:true,scope:'local_internal_only',authorizedBy:'user',
    reason:'User-authorized v5 capital, number, and punctuation APK; physical camera acceptance remains unverified.'},
  runtimeContract:{module:V5_IDENTITY.module,selectedPolicy:V5_IDENTITY.policy,methodSha256:V5_IDENTITY.method,method:method.tta_method},
  symbolContract:{locale:'id-basic',eventSchemaVersion:4,capitalMask:32,numberMask:60,maxCharacters:12,maxPhysicalCells:24,
    preserveLeadingZeros:true,mixedAlphanumeric:false,protocol:'indonesian-punctuation-token-v1',
    punctuationMasks:{'.':50,',':2,'?':38,'!':22,':':18,';':6,'-':36,'(':54,')':54},
    parentheses:'paired_non_nested',questionMark:'terminal_after_token',standaloneAmbiguousSymbolsRejected:true},
  internalAcceptance:{passed:false,exactWordSuccessRate:summary.overall_exact_accuracy,acceptedWordAccuracy:summary.accepted_precision,
    evaluatedWordCount:summary.positive_total,acceptedWordCount:summary.accepted_positive_total,
    negativeImageCount:summary.invalid_total,negativeFalseAccepts:summary.invalid_false_accepts},
  benchmarkAcceptance:{scope:'fresh_source_disjoint_recomposed_public_flatbed_scans_not_physical_phone_camera',
    metric:'overall_exact_complete_text_including_rejected_positives',target:.8,passed:summary.target80_met,
    positiveCount:summary.positive_total,exactCorrect:summary.exact_correct,exactAccuracy:summary.overall_exact_accuracy,
    acceptedPositiveCount:summary.accepted_positive_total,acceptedPrecision:summary.accepted_precision,
    precision95Passed:summary.precision95_met,invalidCount:summary.invalid_total,invalidFalseAccepts:summary.invalid_false_accepts,
    selectedPolicy:V5_IDENTITY.policy},
  physicalCameraValidated:false,
  nativeSelfTest:{file:'native-word-fixture.png',sha256:smoke.fixture_sha256,expectedText:smoke.expected_text,width:480,height:192,
    roi:{x:0,y:0,width:1,height:1},native_validated:false,sourceSplit:smoke.source_split,validationRowIndex:101,
    referenceScript:proof(option('--startup-script') ?? 'output/apk-id-v5-20260911/evaluate-startup.py'),
    purpose:'Prepared validation-input startup smoke; not camera geometry evidence'},
  provenance:{datasets:old.provenance?.datasets ?? [],retainedModel:'training/checkpoints/braille-word-reader-id-v5/RETENTION-COPY.json'},
  verification:{datasetRedistributionRightsConfirmed:false},evidence,
};
writeFileSync(destination,JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({template:destination,pendingEvidence:Object.entries(evidence).filter(([,entry]) => entry.sha256==='PENDING_EVIDENCE').map(([key]) => key),assetsChanged:false}));
