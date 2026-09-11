import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createRequire } from 'node:module';
import ts from 'typescript';

const { load } = createRequire(import.meta.url)('./word-native-parity/geometry-reference.cjs');
export const V5_IDENTITY = Object.freeze({
  experiment: 'overall-exact-text-v5-20260911', decoder: 'six-dot-id-v4-experimental',
  model: '6847c91fe186169ba5208a7444f9c8bedc753228bc42b78c691471a85453a6cb',
  checkpoint: 'ac9e30cd27cb22ca4f97db42637d3a9e2a27ed47d13a962a661c3bb01c0e66bb',
  method: 'd7f900120359d1dcd32c8e3bd586663439a9bd31a776630571fbcef0c50ef9c4',
  finalTest: '1afa4f10b013c93e48927d4c23e6cb04f3015c12f2699253152fb5f9bd9685f3',
  confirmation: '91be170361249bd55eb5b381f3adda04169da47122a968108355881b15b039da',
  module: 'word_overall_v5_zoom', policy: 'cascade_zoom_fallback',
});
const views = ['identity','horizontal','vertical','half_turn','zoom125','zoom150'];
const groups = ['lowercase','capital_single','capital_word','numbers','punctuation'];
const digest = (value) => assert.match(value, /^[a-f0-9]{64}$/, 'Missing SHA-256 identity');

export function summarizeV5(rows, primary = false) {
  assert.ok(Array.isArray(rows) && rows.length > 0, 'Missing complete evaluation decisions');
  assert.equal(new Set(rows.map((row) => row.source)).size, rows.length, 'Duplicate evaluation rows');
  for (const row of rows) {
    assert.equal(typeof row.expected, 'string'); assert.equal(typeof row.predicted, 'string');
    assert.equal(typeof row.reason, 'string'); assert.ok(row.expected.length <= 12 && row.predicted.length <= 12);
    assert.equal(Boolean(row.predicted), !row.reason, 'Accepted text and rejection must be exclusive');
    assert.equal(typeof row.route, 'string');
  }
  const positive = rows.filter((row) => row.expected), invalid = rows.filter((row) => !row.expected);
  const correct = positive.filter((row) => row.expected === row.predicted).length;
  const accepted = positive.filter((row) => row.predicted).length;
  if (primary) {
    assert.equal(positive.length, 500, 'Primary denominator must include all 500 positives');
    assert.equal(invalid.length, 40, 'Invalid rows stay outside the accuracy denominator');
    assert.equal(positive.filter((row) => row.expected.length > 1).length, 480, 'Preserve 96% multi-character scope');
    for (const group of groups) assert.equal(positive.filter((row) => row.group === group).length, 100, 'Balanced group denominator changed');
    assert.ok(invalid.every((row) => row.group === 'rejection'));
  }
  return { total: positive.length, correct, accepted, exact: correct / positive.length,
    precision: accepted ? correct / accepted : null, invalid: invalid.length,
    falseAccepts: invalid.filter((row) => row.predicted).length };
}

export function verifyOverallWordModel({ root, directory, manifest, evidence, inside, hash, environment = process.env }) {
  assert.equal(manifest.experimentVersion, V5_IDENTITY.experiment);
  assert.equal(manifest.artifact.status, 'experimental_candidate');
  assert.equal(manifest.artifact.sha256, V5_IDENTITY.model, 'Use the exact qualified v5 artifact');
  assert.equal(manifest.decoderVersion, V5_IDENTITY.decoder);
  assert.deepEqual(manifest.experimentalUse, { enabled: true, scope: 'local_internal_only', authorizedBy: 'user',
    reason: 'User-authorized v5 capital, number, and punctuation APK; physical camera acceptance remains unverified.' });
  assert.notEqual(environment.APP_ENV, 'production', 'Experimental model cannot enter production');
  assert.ok(!environment.EAS_BUILD_PROFILE && environment.EAS_BUILD !== 'true', 'Experimental model cannot enter cloud builds');
  assert.equal(manifest.internalAcceptance.passed, false, 'Benchmark success cannot promote physical release acceptance');
  assert.equal(manifest.physicalCameraValidated, false);
  assert.deepEqual(manifest.symbolContract, { locale: 'id-basic', eventSchemaVersion: 4, capitalMask: 32, numberMask: 60,
    maxCharacters: 12, maxPhysicalCells: 24, preserveLeadingZeros: true, mixedAlphanumeric: false,
    protocol: 'indonesian-punctuation-token-v1', punctuationMasks: { '.':50, ',':2, '?':38, '!':22, ':':18, ';':6, '-':36, '(':54, ')':54 },
    parentheses: 'paired_non_nested', questionMark: 'terminal_after_token', standaloneAmbiguousSymbolsRejected: true });
  const required = ['finalMethod','finalTest','confirmation','protocol','sourceSeparation','conversion','originalConversion',
    'validationSelection','regression','qualification','inferencePolicy','runtimeParity','nativeSelfTest'];
  const reports = Object.fromEntries(required.map((name) => {
    assert.ok(manifest.evidence?.[name], `Missing v5 evidence: ${name}`);
    return [name,evidence(manifest.evidence[name])];
  }));
  const method = reports.finalMethod, test = reports.finalTest, conversion = reports.conversion;
  assert.equal(manifest.evidence.finalMethod.sha256, V5_IDENTITY.method, 'Frozen method evidence changed');
  assert.equal(manifest.evidence.finalTest.sha256,V5_IDENTITY.finalTest, 'Frozen final TEST evidence changed');
  assert.equal(manifest.evidence.confirmation.sha256,V5_IDENTITY.confirmation, 'Frozen backend confirmation changed');
  assert.equal(method.protocol_version, V5_IDENTITY.experiment);
  assert.equal(method.checkpoint_sha256, V5_IDENTITY.checkpoint); assert.equal(method.tflite_sha256, V5_IDENTITY.model);
  assert.equal(method.primary_engine, 'tflite'); assert.equal(method.selected_policy, V5_IDENTITY.policy);
  assert.equal(method.runtime_module, V5_IDENTITY.module); assert.equal(method.selection_scope, 'validation_only');
  assert.equal(method.test_opened, false); assert.deepEqual(method.thresholds, manifest.thresholds);
  assert.deepEqual(manifest.runtimeContract, { module: V5_IDENTITY.module, selectedPolicy: V5_IDENTITY.policy,
    methodSha256: V5_IDENTITY.method, method: method.tta_method });
  assert.deepEqual(method.tta_method.views, views);
  assert.equal(reports.protocol.version, V5_IDENTITY.experiment); assert.equal(reports.sourceSeparation.status, 'passed');
  for (const report of [test,reports.confirmation]) {
    assert.equal(report.method_sha256, V5_IDENTITY.method); assert.equal(report.tflite_sha256, V5_IDENTITY.model);
    assert.equal(report.checkpoint_sha256, V5_IDENTITY.checkpoint); assert.equal(report.selected_policy, V5_IDENTITY.policy);
    assert.deepEqual(report.thresholds, manifest.thresholds); assert.equal(report.test_used_for_selection, false);
    assert.equal(report.physical_camera_validated, false);
  }
  assert.equal(test.runtime.engine, 'tensorflow_lite', 'Benchmark must measure the actual TFLite artifact');
  const counts = summarizeV5(test.results.primary.decisions, true), summary = test.results.primary.summary;
  for (const [field,value] of Object.entries({ positive_total:counts.total, exact_correct:counts.correct,
    accepted_positive_total:counts.accepted, overall_exact_accuracy:counts.exact, accepted_precision:counts.precision,
    invalid_total:counts.invalid, invalid_false_accepts:counts.falseAccepts })) assert.equal(summary[field],value, 'Misreported final benchmark arithmetic');
  assert.equal(test.target80_met, counts.exact >= .8); assert.equal(summary.target80_met, counts.exact >= .8);
  assert.ok(counts.exact >= .8); assert.equal(summary.precision95_met, counts.precision >= .95);
  assert.deepEqual(manifest.benchmarkAcceptance, { scope: 'fresh_source_disjoint_recomposed_public_flatbed_scans_not_physical_phone_camera',
    metric: 'overall_exact_complete_text_including_rejected_positives', target: .8, passed: true,
    positiveCount:counts.total, exactCorrect:counts.correct, exactAccuracy:counts.exact, acceptedPositiveCount:counts.accepted,
    acceptedPrecision:counts.precision, precision95Passed:counts.precision >= .95, invalidCount:counts.invalid,
    invalidFalseAccepts:counts.falseAccepts, selectedPolicy:V5_IDENTITY.policy });
  for (const [field,value] of Object.entries({ evaluatedWordCount:counts.total, acceptedWordCount:counts.accepted,
    exactWordSuccessRate:counts.exact, acceptedWordAccuracy:counts.precision, negativeImageCount:counts.invalid,
    negativeFalseAccepts:counts.falseAccepts })) assert.equal(manifest.internalAcceptance[field],value);
  const { parseWordMasks } = load(resolve(root,'src/ml/word-symbols.ts'));
  const currentContract = load(resolve(root,'src/ml/word-model-contract.ts'));
  assert.equal(currentContract.WORD_RUNTIME_MODULE,V5_IDENTITY.module, 'App runtime module differs from measured method');
  assert.equal(currentContract.WORD_RUNTIME_POLICY,V5_IDENTITY.policy, 'App runtime policy differs from measured method');
  assert.equal(currentContract.WORD_SYMBOL_PROTOCOL,manifest.symbolContract.protocol);
  for (const row of test.results.primary.decisions) {
    assert.equal(row.semantic_signature[0],row.predicted); assert.equal(row.semantic_signature[1],row.reason);
    if (row.predicted) {
      const parsed = parseWordMasks(row.semantic_signature[2].map((cell) => cell[1]));
      assert.equal(parsed.status,'success', 'Current parser cannot reproduce accepted physical masks');
      assert.equal(parsed.text,row.predicted);
    }
  }
  assert.equal(test.results.native_context.decisions.length,200);
  assert.equal(reports.confirmation.status,'passed');
  for (const name of ['primary','native_context']) {
    assert.deepEqual(reports.confirmation.results[name].decisions,test.results[name].decisions, 'Backend semantic decisions differ');
    assert.equal(reports.confirmation.parity[name].status,'passed');
    assert.deepEqual(reports.confirmation.parity[name].semantic_mismatches,[]);
    assert.deepEqual(reports.confirmation.parity[name].routing_differences,[]);
  }
  assert.equal(conversion.status,'passed'); assert.equal(conversion.allclose,true);
  assert.equal(conversion.tflite_sha256,V5_IDENTITY.model); assert.equal(conversion.checkpoint_sha256,V5_IDENTITY.checkpoint);
  assert.equal(conversion.runtime_module,V5_IDENTITY.module); assert.deepEqual(conversion.method,method.tta_method);
  assert.deepEqual(conversion.thresholds,manifest.thresholds); assert.deepEqual(conversion.views,views);
  assert.equal(conversion.samples,540); assert.equal(conversion.inference_count,3240);
  assert.deepEqual(conversion.pipeline_decision_mismatches,[]);
  assert.equal(conversion.original_numerical_gate_passed,false, 'Preserve the original numerical gate failure');
  assert.equal(conversion.original_report_sha256,manifest.evidence.originalConversion.sha256);
  assert.equal(reports.originalConversion.status,'failed'); assert.equal(reports.originalConversion.allclose,false);
  assert.equal(conversion.atol,.0003); assert.equal(conversion.rtol,0);
  assert.ok(Number.isFinite(conversion.torch_tflite_max_abs) && conversion.torch_tflite_max_abs <= .0003);
  assert.equal(reports.qualification.primary_overall_exact_accuracy,counts.exact);
  assert.equal(reports.qualification.accepted_precision,counts.precision);
  assert.equal(reports.qualification.physical_camera_validated,false);
  assert.equal(reports.inferencePolicy.selected_policy,V5_IDENTITY.policy);
  assert.deepEqual(reports.inferencePolicy.tta_method,method.tta_method);
  assert.equal(reports.validationSelection.checkpoint_sha256,V5_IDENTITY.checkpoint);
  assert.equal(reports.validationSelection.selected_validation_policy,V5_IDENTITY.policy);
  assert.deepEqual(reports.validationSelection.method,method.tta_method);
  assert.equal(reports.validationSelection.selection_scope,'validation_only'); assert.equal(reports.validationSelection.test_opened,false);
  assert.equal(reports.regression.method_sha256,V5_IDENTITY.method); assert.equal(reports.regression.runtime.engine,'tensorflow_lite');
  for (const [name,total] of [['external607',607],['legacy405',405],['negative44',44]]) {
    assert.equal(reports.regression.results[name].candidate.total,total, 'Historical regression must remain separate');
  }
  const parity = reports.runtimeParity;
  assert.equal(parity.parityPassed,true, 'Current application cascade parity must pass');
  assert.equal(parity.modelSha256,V5_IDENTITY.model); assert.equal(parity.methodSha256,V5_IDENTITY.method);
  assert.equal(parity.runtimeModule,V5_IDENTITY.module); assert.equal(parity.selectedPolicy,V5_IDENTITY.policy);
  assert.equal(parity.decoderVersion,V5_IDENTITY.decoder); assert.deepEqual(parity.thresholds,manifest.thresholds);
  assert.equal(parity.scope,'VALIDATION'); assert.equal(parity.sampleCount,540);
  assert.deepEqual(parity.rawViews,views); assert.deepEqual(parity.mismatches,{text:0,reason:0,masks:0,roles:0,outputs:0,route:0});
  assert.equal(parity.validationSha256,conversion.validation_sha256);
  assert.equal(parity.transformParity.passed,true); assert.deepEqual(parity.transformParity.views,views);
  assert.equal(parity.transformParity.fullPixelsPerView,276480);
  assert.deepEqual(Object.keys(parity.transformParity.sha256).sort(),[...views].sort());
  for (const value of Object.values(parity.transformParity.sha256)) digest(value);
  assert.equal(parity.originalGeometryPreserved,true); assert.equal(parity.originalObjectnessPreserved,true);
  assert.equal(parity.fixedLazyRoutingVerified,true); assert.equal(parity.physicalCameraValidated,false);
  assert.equal(parity.nativeRuntimeValidated,false);
  for (const name of ['src/ml/word-runtime-v5.ts','src/ml/word-symbols.ts','src/ml/word-decoder.ts','src/ml/word-image-processing.ts',
    'src/ml/word-model-contract.ts','src/types/word-scan.ts']) assert.ok(parity.sourceSha256?.[name], `Missing current runtime source: ${name}`);
  for (const [file,expected] of Object.entries(parity.sourceSha256)) assert.equal(hash(readFileSync(inside(root,file))),expected, `Stale v5 runtime parity: ${file}`);
  digest(parity.oracleSha256); digest(parity.fixtureSha256);
  assert.equal(parity.oracleSha256,conversion.oracle_sha256);
  assert.equal(hash(readFileSync(inside(root,parity.fixtureFile))),parity.fixtureSha256);
  assert.equal(hash(readFileSync(inside(root,parity.testFile))),parity.testSha256, 'Parity test source changed');
  const fixture = manifest.nativeSelfTest, smoke = reports.nativeSelfTest;
  const png = readFileSync(inside(directory,fixture.file)); assert.equal(hash(png),fixture.sha256);
  assert.equal(png.subarray(0,8).toString('hex'),'89504e470d0a1a0a');
  assert.equal(png.readUInt32BE(16),480); assert.equal(png.readUInt32BE(20),192);
  assert.equal(fixture.width,480); assert.equal(fixture.height,192);
  assert.deepEqual(fixture.roi,{x:0,y:0,width:1,height:1}); assert.equal(fixture.native_validated,false);
  assert.equal(fixture.sourceSplit,'previously_observed_validation'); assert.equal(fixture.validationRowIndex,101);
  assert.equal(smoke.status,'passed'); assert.equal(smoke.model_sha256,V5_IDENTITY.model);
  assert.equal(smoke.checkpoint_sha256,V5_IDENTITY.checkpoint); assert.equal(smoke.fixture_sha256,fixture.sha256);
  assert.equal(smoke.source_split,fixture.sourceSplit); assert.equal(smoke.test_used_for_selection,false);
  assert.equal(smoke.runtime.engine,'tensorflow_lite'); assert.equal(smoke.selected_policy,V5_IDENTITY.policy);
  assert.deepEqual(smoke.thresholds,manifest.thresholds); assert.equal(smoke.decision.text,fixture.expectedText);
  assert.equal(smoke.decision.reason,''); assert.equal(smoke.expected_text,fixture.expectedText);
  assert.equal(smoke.native_camera_validated,false); digest(smoke.input_float32_sha256);
  assert.equal(fixture.referenceScript.sha256,smoke.script_sha256);
  assert.equal(hash(readFileSync(inside(root,fixture.referenceScript.file))),smoke.script_sha256, 'Startup evidence script changed');
  assert.equal(reports.validationSelection.variants.identity.decisions[101].expected,fixture.expectedText);
  assert.equal(parseWordMasks(smoke.decision.cells.map((cell) => cell.mask)).text,fixture.expectedText);
  const syntax = ts.createSourceFile('word-model-assets.ts',readFileSync(resolve(root,'src/ml/word-model-assets.ts'),'utf8'),ts.ScriptTarget.Latest,true);
  const manifestImport = syntax.statements.find((statement) => ts.isImportDeclaration(statement) &&
    statement.importClause?.name?.text === 'manifest');
  assert.ok(manifestImport && ts.isStringLiteral(manifestImport.moduleSpecifier));
  assert.equal(manifestImport.moduleSpecifier.text,'../../assets/models/braille-word-reader-id-v5/manifest.json', 'App must load the verified v5 manifest');
  const declarations = syntax.statements.filter((s) => ts.isVariableStatement(s) && s.modifiers?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword))
    .flatMap((s) => s.declarationList.declarations);
  for (const [name,file] of [['WORD_MODEL_ASSET',manifest.artifact.file],['WORD_NATIVE_FIXTURE_ASSET',fixture.file]]) {
    const value = declarations.find((d) => ts.isIdentifier(d.name) && d.name.text === name)?.initializer;
    assert.ok(value && ts.isCallExpression(value) && ts.isIdentifier(value.expression) && value.expression.text === 'require' &&
      value.arguments.length === 1 && ts.isStringLiteral(value.arguments[0]), 'Use actual literal Metro assets');
    assert.equal(value.arguments[0].text,`../../assets/models/braille-word-reader-id-v5/${file}`);
  }
  assert.equal(declarations.find((d) => ts.isIdentifier(d.name) && d.name.text === 'WORD_MODEL_RELEASE_VERIFIED')?.initializer?.kind,
    ts.SyntaxKind.FalseKeyword, 'Experimental v5 must remain unverified for production');
  return counts;
}
