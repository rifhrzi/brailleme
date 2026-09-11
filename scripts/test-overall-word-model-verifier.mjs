// In-memory mutations of retained evidence plus a clearly synthetic parity
// report exercise the guards; these tests do not claim application parity.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyExperimentalWordModel } from './verify-experimental-word-model-assets.mjs';
import { V5_IDENTITY } from './verify-overall-word-model-assets.mjs';
import { activeWordModelDirectory } from './word-model-active-path.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)),'..');
const directory = resolve(root,'assets/models/braille-word-reader-id-v5');
const original = JSON.parse(readFileSync(resolve(directory,'manifest.json'),'utf8'));
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const inside = (base,file) => {
  assert.equal(typeof file,'string');
  const target = resolve(base,file), rel = relative(base,target);
  assert.ok(rel && !rel.startsWith('..') && !rel.includes(':')); return target;
};
const sourceNames = ['src/ml/word-runtime-v5.ts','src/ml/word-symbols.ts','src/ml/word-decoder.ts',
  'src/ml/word-image-processing.ts','src/ml/word-model-contract.ts','src/types/word-scan.ts'];
const reports = Object.fromEntries(Object.entries(original.evidence).filter(([name]) => name !== 'runtimeParity')
  .map(([name,entry]) => [name,JSON.parse(readFileSync(inside(root,entry.file),'utf8'))]));
const fixtureFile = 'assets/models/braille-word-reader-id-v5/native-word-fixture.png';
const syntheticParity = { parityPassed:true,modelSha256:V5_IDENTITY.model,methodSha256:V5_IDENTITY.method,
  runtimeModule:V5_IDENTITY.module,selectedPolicy:V5_IDENTITY.policy,decoderVersion:V5_IDENTITY.decoder,
  thresholds:original.thresholds,scope:'VALIDATION',sampleCount:540,
  rawViews:['identity','horizontal','vertical','half_turn','zoom125','zoom150'],
  mismatches:{text:0,reason:0,masks:0,roles:0,outputs:0,route:0},
  sourceSha256:Object.fromEntries(sourceNames.map((file) => [file,hash(readFileSync(inside(root,file)))])),
  oracleSha256:reports.conversion.oracle_sha256,fixtureFile,fixtureSha256:hash(readFileSync(inside(root,fixtureFile))),
  validationSha256:reports.conversion.validation_sha256,
  transformParity:{passed:true,views:['identity','horizontal','vertical','half_turn','zoom125','zoom150'],fullPixelsPerView:276480,
    sha256:Object.fromEntries(['identity','horizontal','vertical','half_turn','zoom125','zoom150'].map((name)=>[name,'a'.repeat(64)]))},
  originalGeometryPreserved:true,originalObjectnessPreserved:true,fixedLazyRoutingVerified:true,
  physicalCameraValidated:false,nativeRuntimeValidated:false,
  testFile:'scripts/test-overall-word-model-verifier.mjs',testSha256:hash(readFileSync(resolve(root,'scripts/test-overall-word-model-verifier.mjs'))),
  scopeNote:'SYNTHETIC GUARD TEST ONLY; not actual runtime parity' };
let passed = 0;
function check(name, change, pattern) {
  const manifest = structuredClone(original), values = structuredClone({...reports,runtimeParity:syntheticParity}), environment = {};
  change({manifest,reports:values,environment});
  const verify = () => verifyExperimentalWordModel({root,directory,manifest,hash,inside,environment,
    evidence(entry) {
      const name = Object.entries(manifest.evidence).find(([,value]) => value === entry)?.[0];
      assert.ok(name); return values[name];
    }});
  if (pattern) assert.throws(verify,pattern,name);
  else { const actual = verify(); assert.equal(actual.correct,438); assert.equal(actual.total,500); assert.equal(actual.accepted,474); }
  passed++; console.log('PASS '+name);
}
assert.equal(activeWordModelDirectory(root),directory);
check('v5 evidence arithmetic and method binding with synthetic guard inputs',()=>{});
check('identity-only model cannot claim cascade accuracy',({manifest})=>{manifest.runtimeContract.selectedPolicy='identity';},/deep|cascade/);
check('different model rejected',({manifest})=>{manifest.artifact.sha256='0'.repeat(64);},/exact qualified/);
check('frozen method cannot be replaced',({manifest})=>{manifest.evidence.finalMethod.sha256='0'.repeat(64);},/Frozen method/);
check('fabricated physical release approval rejected',({manifest})=>{manifest.internalAcceptance.passed=true;},/physical release/);
check('production blocked',({environment})=>{environment.APP_ENV='production';},/cannot enter production/);
check('cloud preview blocked',({environment})=>{environment.EAS_BUILD_PROFILE='preview';},/cloud builds/);
check('benchmark denominator cannot exclude rejected words',({reports})=>{reports.finalTest.results.primary.decisions.pop();},/Invalid rows|denominator/);
check('duplicate test source rejected',({reports})=>{reports.finalTest.results.primary.decisions[1].source=reports.finalTest.results.primary.decisions[0].source;},/Duplicate/);
check('claimed exact accuracy independently recounted',({manifest})=>{manifest.benchmarkAcceptance.exactAccuracy=.99;},/deep/);
check('precision cannot be labeled 95%',({manifest})=>{manifest.benchmarkAcceptance.precision95Passed=true;},/deep/);
check('missing application parity blocks build',({manifest})=>{delete manifest.evidence.runtimeParity;},/Missing v5 evidence/);
check('wrong application cascade policy rejected',({reports})=>{reports.runtimeParity.selectedPolicy='identity';},/cascade_zoom_fallback/);
check('six-view coverage mandatory',({reports})=>{reports.runtimeParity.rawViews.pop();},/deep/);
check('all540 validation rows required',({reports})=>{reports.runtimeParity.sampleCount=24;},/540/);
check('route mismatch rejected despite same text',({reports})=>{reports.runtimeParity.mismatches.route=1;},/deep/);
check('physical mask mismatch rejected',({reports})=>{reports.runtimeParity.mismatches.masks=1;},/deep/);
check('stale TS helper proof rejected',({reports})=>{reports.runtimeParity.sourceSha256['src/ml/word-runtime-v5.ts']='0'.repeat(64);},/Stale v5 runtime/);
check('original numerical failure remains visible',({reports})=>{reports.conversion.original_numerical_gate_passed=true;},/original numerical/);
check('startup fixture cannot use finalTEST',({manifest})=>{manifest.nativeSelfTest.sourceSplit='test';},/previously_observed_validation/);
check('startup model identity fixed',({reports})=>{reports.nativeSelfTest.model_sha256='0'.repeat(64);},/6847c91/);
console.log(`V5 experimental integrity guards: ${passed}/${passed} passed; synthetic parity input, no model inference.`);
