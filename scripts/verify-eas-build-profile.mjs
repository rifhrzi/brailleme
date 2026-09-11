import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { activeWordModelManifest } from './word-model-active-path.mjs';

const modelManifest = activeWordModelManifest(resolve(dirname(fileURLToPath(import.meta.url)), '..'));

const buildProfile = process.env.EAS_BUILD_PROFILE;
const appEnvironment = process.env.APP_ENV;
const productionBuild =
  buildProfile === 'production' || appEnvironment === 'production';
const supportedProfiles = new Set(['development', 'preview', 'production']);
const provenanceDatasets = modelManifest.provenance?.datasets;
const allDatasetRightsConfirmed =
  Array.isArray(provenanceDatasets) &&
  provenanceDatasets.length > 0 &&
  provenanceDatasets.every(
    (dataset) => dataset?.redistributionRightsConfirmed === true,
  );

if (buildProfile && !supportedProfiles.has(buildProfile)) {
  throw new Error(`Build profile ${buildProfile} tidak didukung.`);
}
if (
  buildProfile &&
  modelManifest.verification?.datasetRedistributionRightsConfirmed !==
    allDatasetRightsConfirmed
) {
  throw new Error(
    'Build EAS ditolak: flag hak redistribusi tidak konsisten dengan provenance dataset.',
  );
}
if (buildProfile && !allDatasetRightsConfirmed) {
  throw new Error(
    'Build EAS ditolak: hak redistribusi seluruh dataset model belum dikonfirmasi. Gunakan build development lokal saja.',
  );
}
if (buildProfile && buildProfile !== 'production' && productionBuild) {
  throw new Error(
    `Konfigurasi EAS tidak konsisten: profile ${buildProfile} memakai APP_ENV production.`,
  );
}

if (productionBuild && appEnvironment !== 'production') {
  throw new Error('Build profile production wajib memakai APP_ENV production.');
}
const npmExecPath = process.env.npm_execpath;
if (!npmExecPath) {
  throw new Error('npm_execpath tidak tersedia untuk menjalankan release gate.');
}

const verificationScript = productionBuild ? 'release:verify' : 'model:verify';
const verification = spawnSync(
  process.execPath,
  [npmExecPath, 'run', verificationScript],
  { stdio: 'inherit', env: process.env },
);

if (verification.error) {
  throw verification.error;
}
if (verification.status !== 0) {
  process.exit(verification.status ?? 1);
}
