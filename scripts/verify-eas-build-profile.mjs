import { spawnSync } from 'node:child_process';

const buildProfile = process.env.EAS_BUILD_PROFILE;
const appEnvironment = process.env.APP_ENV;
const unverifiedModelOverride = process.env.ALLOW_UNVERIFIED_MODEL === 'true';
const experimentalProductionBuild =
  buildProfile === 'production-experimental';
const productionBuild =
  buildProfile === 'production' ||
  experimentalProductionBuild ||
  appEnvironment === 'production';

if (
  buildProfile &&
  buildProfile !== 'production' &&
  buildProfile !== 'production-experimental'
) {
  if (productionBuild) {
    throw new Error(
      `Konfigurasi EAS tidak konsisten: profile ${buildProfile} memakai APP_ENV production.`,
    );
  }
}

if (productionBuild && appEnvironment !== 'production') {
  throw new Error('Build profile production wajib memakai APP_ENV production.');
}
if (experimentalProductionBuild && !unverifiedModelOverride) {
  throw new Error(
    'Profile production-experimental wajib memakai ALLOW_UNVERIFIED_MODEL=true.',
  );
}
if (!experimentalProductionBuild && unverifiedModelOverride) {
  throw new Error(
    'ALLOW_UNVERIFIED_MODEL hanya boleh dipakai profile production-experimental.',
  );
}

const npmExecPath = process.env.npm_execpath;
if (!npmExecPath) {
  throw new Error('npm_execpath tidak tersedia untuk menjalankan release gate.');
}

const verificationScript = experimentalProductionBuild
  ? 'release:experimental:verify'
  : productionBuild
    ? 'release:verify'
    : 'model:verify';
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
