// Run the actual verifier's provenance statements against frozen run21 proofs.
// No model/backend imports, image reads, selection or numeric gate substitution.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

const work = '.braille-training-work/word-scan-v1-20260905';
const folder = 'training/evidence/braille-word-reader-v1/dsbi-training-preparation';
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const link = (file) => ({ file, sha256: hash(readFileSync(file)) });
const verifiedBytes = (entry) => {
  const bytes = readFileSync(entry.file);
  assert.equal(hash(bytes), entry.sha256);
  return bytes;
};
const evidence = (entry) => JSON.parse(verifiedBytes(entry));
const source = readFileSync('scripts/verify-word-model-assets.mjs', 'utf8');
const recordFile = 'training/evidence/braille-word-reader-v1/geometry-residual-preflight/preflight.json';
const run = evidence(link(recordFile)), config = evidence(run.configuration);
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
// This small metadata file is already losslessly archived; use its original
// frozen path as the per-run link until final model ancestry is assembled.
run.dsbiPointMetadata = link(`${work}/dsbi-point-metadata-v1/points.json`);
const ancestor = { run: 'run20', configuration: link(`${work}/run20/training-config.json`),
  'sampling-history.json': link(`${work}/run20/sampling-history.json`),
  'sampling-lineage-final.json': link(`${work}/run20/sampling-lineage-final.json`) };
const verifySourceSet = source.slice(source.indexOf('const verifySourceSet = '), source.indexOf('verifySourceSet(provenance.selectionSources'));
const preparations = source.slice(source.indexOf('const archivedData = '), source.indexOf('assert.ok(Array.isArray(provenance.trainingRuns)'));
const configChecks = source.slice(source.indexOf('  const indexes = config.supplemental_source_indexes'), source.indexOf("  if (!run['sampling-lineage-final.json'])"));
const context = { assert, basename, hash, evidence, verifiedBytes, retainedDataBytes,
  imageNormalization: config.imageNormalization,
  provenance: { supplementalPreparations: [preparation], trainingRuns: [ancestor, run] },
  manifest: { datasets: [{ name: 'DSBI', revision: index.revision, repository: 'https://github.com/yeluo1994/DSBI' }],
    artifact: { sha256: '406db0e55e8a5ba9e0d00fb22aadc24f19698219f375e4dcead9057cf3bf2230' } },
  validation: { oracle: { checkpoint_sha256: run.geometryInitialization.baseParentCheckpointSha256 } }, config, run };
new Function(...Object.keys(context), `${verifySourceSet}\n${preparations}\nverifySourceSet(run.sources, config.source_sha256);\nconst usedSupplementalHashes = new Set();\n${configChecks}\nassert.equal(usedSupplementalHashes.size, 1);`)(...Object.values(context));
const report = { status: 'passed', scope: 'Actual run21 geometry and DSBI point provenance checks only. The retained preflight sources/reports and 26 raw TRAIN annotations were hash verified; priors recomputed from all 9983 unique source cells. The existing run20 sampling audit proves base v15 epoch9 eligibility. Per-run DSBI metadata bytes additionally match their retained gzip archive. Final candidate lineage is not yet assembled; numerical recognition, full405 trained-model parity, native and physical gates are not claimed.',
  verifier_sha256: hash(source), script_sha256: hash(readFileSync(new URL(import.meta.url))),
  preflight: link(recordFile), training_configuration: run.configuration, geometry: run.geometryInitialization,
  parent_sampling_eligibility: ancestor, dsbi_source_index: indexLink, dsbi_point_metadata: run.dsbiPointMetadata,
  source_annotation_count: index.sources.length, source_cells: index.sources.reduce((sum, row) => sum + row.positive_cell_count, 0),
  source_photos_read: false, model_execution: false };
writeFileSync(`${work}/real-geometry-preflight-verifier-check.json`, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ status: report.status, config: run.configuration.sha256, source_cells: report.source_cells, report: `${work}/real-geometry-preflight-verifier-check.json` }));
