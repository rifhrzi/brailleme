import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { activeWordModelManifest } from './word-model-active-path.mjs';

const word = activeWordModelManifest(resolve(dirname(fileURLToPath(import.meta.url)), '..'));
if (word.artifact.status !== 'verified' || word.physicalCameraValidated !== true) {
  throw new Error('Production ditolak: pengenal kata belum memiliki validasi kertas timbul, parity kamera native, holdout independen, aksesibilitas, dan benchmark perangkat pada model yang sama. Bukti classifier lama tidak memverifikasi pengenal kata.');
}
await import('./verify-word-model-assets.mjs');
// The physical evidence verifier binds every report to the active word model;
// classifier-v2 evidence cannot satisfy these checks.
await import('./verify-model-release-evidence.mjs');
