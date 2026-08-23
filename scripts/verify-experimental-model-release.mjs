import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));

function readJson(...segments) {
  return JSON.parse(readFileSync(join(projectRoot, ...segments), "utf8"));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(`Experimental production ditolak: ${message}`);
  }
}

const manifest = readJson(
  "assets",
  "models",
  "braille-cell-classifier-v1",
  "manifest.json",
);
const acceptance = readJson(
  "docs",
  "release",
  "EXPERIMENTAL-MODEL-RISK-ACCEPTANCE.json",
);
const liveRoute = readFileSync(join(projectRoot, "app", "live.tsx"), "utf8");
const homeScreen = readFileSync(
  join(projectRoot, "src", "features", "home", "home-screen.tsx"),
  "utf8",
);
const storeListing = readFileSync(
  join(projectRoot, "docs", "release", "PLAY-STORE-LISTING-ID.md"),
  "utf8",
);

assert(acceptance.schemaVersion === 1, "schema risk acceptance tidak dikenal");
assert(acceptance.enabled === true, "risk acceptance tidak aktif");
assert(
  acceptance.decision === "allow_unverified_model_in_production",
  "keputusan risk acceptance tidak cocok",
);
assert(
  manifest.artifact?.status === "unverified_candidate",
  "jalur eksperimental hanya untuk model berstatus unverified_candidate",
);
assert(
  acceptance.modelStatus === manifest.artifact.status,
  "status model berbeda dari risk acceptance",
);
assert(
  acceptance.modelSha256 === manifest.artifact?.sha256,
  "SHA-256 model berbeda dari risk acceptance",
);

for (const key of [
  "tensorContractInspected",
  "labelCountMatchesOutput",
  "labelOrderConfirmedAgainstTrainingCode",
  "preprocessingConfirmedAgainstTrainingCode",
  "kerasTfliteConversionParityPassed",
]) {
  assert(manifest.verification?.[key] === true, `verifikasi dasar belum lulus: ${key}`);
}

assert(
  acceptance.requirements?.modelEvidenceMustNotBeMarkedVerified === true,
  "risk acceptance harus mempertahankan status evidence yang jujur",
);
assert(
  liveRoute.includes("hasilnya dapat salah"),
  "disclosure risiko wajib tampil sebelum kamera aktif",
);
assert(
  homeScreen.includes("Hasil masih eksperimental dan wajib diverifikasi"),
  "Beranda wajib menyatakan status eksperimental",
);
assert(
  storeListing.includes("Kandidat dapat salah dan harus diverifikasi manusia"),
  "listing wajib mengungkap keterbatasan model",
);

console.warn(
  "PERINGATAN: experimental production diizinkan untuk model unverified_candidate; ini bukan production GO terverifikasi.",
);
