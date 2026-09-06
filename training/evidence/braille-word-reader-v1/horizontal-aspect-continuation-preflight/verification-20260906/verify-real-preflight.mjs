// Actual archived provenance replay only; no model, fixtures, thresholds or inference.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

const work = '.braille-training-work/word-scan-v1-20260905';
const archive = 'training/evidence/braille-word-reader-v1/horizontal-aspect-continuation-preflight';
const folder = 'training/evidence/braille-word-reader-v1/dsbi-training-preparation';
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const link = (file) => ({ file, sha256: hash(readFileSync(file)) });
const verifiedBytes = (entry) => {
  const bytes = readFileSync(entry.file);
  assert.equal(hash(bytes), entry.sha256, `Changed proof ${entry.file}`);
  return bytes;
};
const evidence = (entry) => JSON.parse(verifiedBytes(entry));
const source = readFileSync('scripts/verify-word-model-assets.mjs', 'utf8');
assert.equal(hash(source), '6e0a30e1fe50915f53000fa455b6d71d6d4b6c6ae5341d986f24da5285992d92');
const recordFile = `${archive}/preflight.json`, recordRef = link(recordFile);
assert.equal(recordRef.sha256, 'c02018b78bb54d3eda81ba0a4e9339455483e34a7ac8f6a68ecff87fb0f6463b');
const run = evidence(recordRef), config = evidence(run.configuration), parentRun = run.ancestors[0];
const archiveRef = link(`${archive}/archive-index.json`), archiveIndex = evidence(archiveRef);
assert.equal(archiveRef.sha256, 'cb5cdefd10027d705431d93842fba0e2e1fec7c20b56560843a9102ca24948b3');
for (const entry of archiveIndex.files) {
  const bytes = verifiedBytes({ ...entry, file: resolve(archive, entry.file) });
  assert.equal(bytes.length, entry.bytes);
}
for (const entry of archiveIndex.externalReferences) verifiedBytes(entry);
const indexLink = link(`${folder}/source-index.json`), index = evidence(indexLink);
const snapshot = evidence(link(`${folder}/source-archive.json`));
const preparation = { dataset: 'DSBI', revision: index.revision, sourceIndex: indexLink,
  generatedManifestSha256: index.generatedManifest.sha256,
  sources: snapshot.files.map((entry) => link(`${folder}/${entry.archived}`)),
  ...Object.fromEntries(['source-archive.json', 'preparation-protocol.json', 'independent-context-audit.json'].map((name) => [name, link(`${folder}/${name}`)])),
  sourceAnnotations: index.sources.map((row, ordinal) => ({ source: row.source,
    file: `training/evidence/braille-word-reader-v1/dsbi-visible-partial-preflight/source-annotations/${String(ordinal).padStart(2, '0')}.txt`, sha256: row.annotation_sha256 })) };
const retainedDataBytes = new Map();
const dataDirectory = 'training/evidence/braille-word-reader-v1/data-manifests';
for (const entry of evidence(link(`${dataDirectory}/index.json`)).files) {
  const compressed = readFileSync(resolve(dataDirectory, entry.file));
  assert.equal(hash(compressed), entry.gzipSha256);
  assert.equal(compressed.length, entry.gzipSizeBytes);
  const bytes = gunzipSync(compressed);
  assert.equal(hash(bytes), entry.sha256);
  assert.equal(bytes.length, entry.sizeBytes);
  retainedDataBytes.set(entry.sha256, bytes);
}
const olderFolder = 'training/evidence/braille-word-reader-v1/failed-selected-v22/run20';
const grandparentRun = { run: 'run20', configuration: link(`${olderFolder}/training-config.json`),
  'sampling-history.json': link(`${olderFolder}/sampling-history.json`),
  'sampling-lineage-final.json': link(`${olderFolder}/sampling-lineage-final.json`) };
const verifyTree = (value) => {
  if (!value || typeof value !== 'object') return;
  if (typeof value.file === 'string' && typeof value.sha256 === 'string') verifiedBytes(value);
  for (const child of Object.values(value)) verifyTree(child);
};
for (const value of [run, grandparentRun, preparation]) verifyTree(value);
const verifySourceSet = source.slice(source.indexOf('const verifySourceSet = '), source.indexOf('verifySourceSet(provenance.selectionSources'));
const preparations = source.slice(source.indexOf('const archivedData = '), source.indexOf('assert.ok(Array.isArray(provenance.trainingRuns)'));
const configChecks = source.slice(source.indexOf('  const indexes = config.supplemental_source_indexes'), source.indexOf("  if (!run['sampling-lineage-final.json'])"));
const context = { assert, basename, hash, evidence, verifiedBytes, retainedDataBytes,
  imageNormalization: config.imageNormalization,
  provenance: { supplementalPreparations: [preparation], trainingRuns: [grandparentRun, parentRun, run] },
  manifest: { datasets: [{ name: 'DSBI', revision: index.revision, repository: 'https://github.com/yeluo1994/DSBI' }],
    artifact: { sha256: '406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230' } },
  validation: { oracle: { checkpoint_sha256: run.horizontalAspectContinuation.parentCheckpointSha256 } },
  actualRuns: [parentRun, run] };
new Function(...Object.keys(context), `${verifySourceSet}\n${preparations}\nconst usedSupplementalHashes = new Set();\nfor (const run of actualRuns) { const config = evidence(run.configuration); verifySourceSet(run.sources, config.source_sha256);\n${configChecks}\n}\nassert.equal(usedSupplementalHashes.size, 1);`)(...Object.values(context));
const report = { status: 'passed', scope: 'Actual retained run27 continuation plus inherited run26 horizontal aspect and DSBI point/source provenance. The current verifier source was executed against exact archived configuration/protocol/guard/HDF5 parent report/source snapshots, original run26 transform proof, and retained run20/run26 eligible ancestry. No numerical model or release acceptance is claimed. Run27 preflight archive has no final sampling history; final selected-model lineage still must retain/verify its completed training history.',
  verifier_sha256: hash(source), script_sha256: hash(readFileSync(new URL(import.meta.url))),
  preflight: recordRef, archive_index: archiveRef, verified_archive_payloads: archiveIndex.files.length,
  training_configuration: run.configuration, continuation: run.horizontalAspectContinuation,
  parent_configuration: parentRun.configuration,
  parent_sampling_history: parentRun['sampling-history.json'], parent_sampling_eligibility: parentRun['sampling-lineage-final.json'],
  inherited_original_transform: parentRun.horizontalAspectPreflight.protocol,
  grandparent_sampling_eligibility: grandparentRun,
  dsbi_source_index: indexLink, dsbi_point_metadata: run.dsbiPointMetadata,
  source_annotation_count: index.sources.length, source_cells: index.sources.reduce((sum, row) => sum + row.positive_cell_count, 0),
  all_proof_paths_archived_in_repository: true, source_photos_read: false, model_execution: false,
  synthetic_unit_suite_repeated: false, original_archive_modified: false };
const output = `${work}/real-horizontal-aspect-continuation-preflight-verifier-check.json`;
writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ status: report.status, report: output, sha256: hash(readFileSync(output)), config: run.configuration.sha256, payloads: report.verified_archive_payloads }));
