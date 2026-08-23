import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const easPackagePath = require.resolve("eas-cli/package.json");
const easPackage = JSON.parse(readFileSync(easPackagePath, "utf8"));
const easBin =
  typeof easPackage.bin === "string" ? easPackage.bin : easPackage.bin?.eas;
const mode = process.argv[2];
const buildId = process.env.EAS_BUILD_ID;
const modeConfig =
  mode === "internal"
    ? { buildProfile: "preview", submitProfile: "internal" }
    : mode === "production-experimental"
      ? {
          buildProfile: "production-experimental",
          submitProfile: "production",
          releaseClass: "unverified-model-exception",
        }
    : mode === "production"
      ? {
          buildProfile: "production",
          submitProfile: "production",
          releaseClass: "verified-model",
        }
      : null;

function fail(message) {
  throw new Error(`Submission ditolak: ${message}`);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: projectRoot,
    encoding: "utf8",
    ...options,
  });
  if (result.status !== 0) {
    fail(result.stderr?.trim() || result.stdout?.trim() || `${command} gagal`);
  }
  return result.stdout.trim();
}

if (!modeConfig) {
  fail("mode harus internal, production, atau production-experimental");
}
if (
  typeof buildId !== "string" ||
  !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu.test(
    buildId,
  )
) {
  fail("set EAS_BUILD_ID ke ID build EAS yang spesifik");
}
if (!easBin) fail("binary EAS CLI lokal tidak ditemukan");

const easCli = resolve(dirname(easPackagePath), easBin);
const currentCommit = run("git", ["rev-parse", "HEAD"]);
if (!/^[0-9a-f]{40}$/u.test(currentCommit)) {
  fail("repository belum memiliki commit HEAD yang valid");
}
if (run("git", ["status", "--porcelain"]).length > 0) {
  fail("working tree harus bersih agar source dan AAB dapat diikat");
}

const expoConfig = JSON.parse(
  run(
    process.execPath,
    [
      join(dirname(require.resolve("expo/package.json")), "bin", "cli"),
      "config",
      "--type",
      "public",
      "--json",
    ],
    {
      env: {
        ...process.env,
        APP_ENV: modeConfig.buildProfile,
        EAS_BUILD_PROFILE: modeConfig.buildProfile,
      },
    },
  ),
);
const projectId = expoConfig.extra?.eas?.projectId;
if (typeof projectId !== "string" || projectId.length === 0) {
  fail("extra.eas.projectId belum dikonfigurasi oleh eas init");
}

const build = JSON.parse(
  run(process.execPath, [easCli, "build:view", buildId, "--json"]),
);
if (build.id !== buildId) fail("respons EAS memakai build ID berbeda");
if (build.project?.id !== projectId) fail("build berasal dari proyek EAS berbeda");
if (build.platform !== "ANDROID") fail("build bukan Android");
if (build.status !== "FINISHED") fail(`status build ${build.status ?? "unknown"}`);
if (build.buildProfile !== modeConfig.buildProfile) {
  fail(`build profile harus ${modeConfig.buildProfile}`);
}
if (build.gitCommitHash !== currentCommit) {
  fail("commit build EAS berbeda dari working tree terverifikasi");
}
if (build.applicationId && build.applicationId !== "com.brailleme.app") {
  fail("applicationId build tidak cocok");
}

const artifactUrl =
  build.artifacts?.applicationArchiveUrl ?? build.artifacts?.buildUrl;
if (typeof artifactUrl !== "string" || !artifactUrl.startsWith("https://")) {
  fail("URL artefak HTTPS tidak tersedia pada build EAS");
}

const response = await fetch(artifactUrl, { redirect: "follow" });
if (!response.ok || !response.body || !response.url.startsWith("https://")) {
  fail(`artefak EAS tidak dapat diunduh (${response.status})`);
}
const maximumArtifactBytes = 500 * 1024 * 1024;
const hash = createHash("sha256");
let artifactBytes = 0;
for await (const chunk of Readable.fromWeb(response.body)) {
  artifactBytes += chunk.length;
  if (artifactBytes > maximumArtifactBytes) {
    fail("artefak EAS melewati batas 500 MiB");
  }
  hash.update(chunk);
}
if (artifactBytes === 0) fail("artefak EAS kosong");

const modelManifestPath = resolve(
  projectRoot,
  "assets",
  "models",
  "braille-cell-classifier-v1",
  "manifest.json",
);
if (!existsSync(modelManifestPath)) fail("manifest model aktif tidak ditemukan");
const modelManifest = JSON.parse(readFileSync(modelManifestPath, "utf8"));
const attestation = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  source: { gitCommit: currentCommit, cleanWorkingTree: true },
  model: {
    version: modelManifest.artifact?.version,
    sha256: modelManifest.artifact?.sha256,
    status: modelManifest.artifact?.status,
    releaseClass: modeConfig.releaseClass ?? "internal-preview",
  },
  eas: {
    projectId,
    buildId,
    buildProfile: build.buildProfile,
    platform: build.platform,
    status: build.status,
  },
  android: {
    applicationId: build.applicationId ?? "com.brailleme.app",
    appVersion: build.appVersion ?? null,
    appBuildVersion: build.appBuildVersion ?? null,
  },
  artifact: { sha256: hash.digest("hex"), sizeBytes: artifactBytes },
  submission: {
    profile: modeConfig.submitProfile,
    exactBuildIdRequired: true,
  },
};
const attestationDirectory = resolve(projectRoot, "artifacts", "release");
mkdirSync(attestationDirectory, { recursive: true });
const attestationPath = resolve(
  attestationDirectory,
  `eas-${modeConfig.submitProfile}-${buildId}.json`,
);
writeFileSync(attestationPath, `${JSON.stringify(attestation, null, 2)}\n`, {
  encoding: "utf8",
  flag: "wx",
});

console.log(
  `Attestation dibuat: ${attestationPath} (${attestation.artifact.sha256})`,
);
const submit = spawnSync(
  process.execPath,
  [
    easCli,
    "submit",
    "--platform",
    "android",
    "--profile",
    modeConfig.submitProfile,
    "--id",
    buildId,
    "--non-interactive",
    "--wait",
  ],
  { cwd: projectRoot, stdio: "inherit" },
);
if (submit.status !== 0) {
  fail(`EAS Submit gagal; attestation tetap tersedia di ${attestationPath}`);
}
