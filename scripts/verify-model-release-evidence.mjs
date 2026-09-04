import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const modelDirectory = resolve(
  projectRoot,
  "assets",
  "models",
  "braille-cell-classifier-v2",
);

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function resolveProjectPath(path, label) {
  if (typeof path !== "string" || path.length === 0) {
    throw new Error(`${label} tidak tersedia.`);
  }
  const resolved = resolve(projectRoot, path);
  const projectRelative = relative(projectRoot, resolved);
  if (projectRelative.startsWith("..") || projectRelative === "") {
    throw new Error(`${label} harus menunjuk file di dalam repository.`);
  }
  return resolved;
}

const manifest = readJson(resolve(modelDirectory, "manifest.json"));
const evidenceManifestPath = resolveProjectPath(
  manifest.provenance?.evidenceManifest,
  "provenance.evidenceManifest",
);
const evidenceDirectory = dirname(evidenceManifestPath);
const evidenceManifest = readJson(evidenceManifestPath);
const blockers = [];
const requiredVerificationKeys = [
  "tensorContractInspected",
  "labelCountMatchesOutput",
  "labelOrderConfirmedAgainstTrainingCode",
  "preprocessingConfirmedAgainstTrainingCode",
  "kerasTfliteConversionParityPassed",
  "nativePostCropFixturePassed",
  "nativeCameraGoldenParityPassed",
  "independentPhysicalHoldoutPassed",
  "negativeCorpusPassed",
  "deviceBenchmarkPassed",
  "datasetRedistributionRightsConfirmed",
];

function addBlocker(condition, message) {
  if (!condition) blockers.push(message);
}

function finiteAtLeast(value, minimum) {
  return Number.isFinite(value) && value >= minimum;
}

function finiteAtMost(value, maximum) {
  return Number.isFinite(value) && value <= maximum;
}

addBlocker(
  manifest.artifact?.status === "verified",
  `status model ${manifest.artifact?.status ?? "missing"}, bukan verified`,
);

const verification = manifest.verification;
const verificationIsRecord =
  verification !== null &&
  typeof verification === "object" &&
  !Array.isArray(verification);
const verificationKeys = verificationIsRecord ? Object.keys(verification) : [];
const missingChecks = requiredVerificationKeys.filter(
  (name) => !verificationKeys.includes(name),
);
const unknownChecks = verificationKeys.filter(
  (name) => !requiredVerificationKeys.includes(name),
);
const incompleteChecks = requiredVerificationKeys.filter(
  (name) => verification?.[name] !== true,
);
addBlocker(
  verificationIsRecord,
  "bagian verification model harus berupa object",
);
addBlocker(
  missingChecks.length === 0,
  `verification wajib hilang: ${missingChecks.join(", ")}`,
);
addBlocker(
  unknownChecks.length === 0,
  `verification tidak dikenal: ${unknownChecks.join(", ")}`,
);
addBlocker(
  incompleteChecks.length === 0,
  `verification belum lulus: ${incompleteChecks.join(", ")}`,
);

const provenanceDatasets = manifest.provenance?.datasets;
addBlocker(
  Array.isArray(provenanceDatasets) && provenanceDatasets.length > 0,
  "provenance dataset model belum tersedia",
);
if (Array.isArray(provenanceDatasets)) {
  for (const dataset of provenanceDatasets) {
    addBlocker(
      dataset?.redistributionRightsConfirmed === true,
      `hak redistribusi dataset belum dikonfirmasi: ${dataset?.name ?? "unknown"}`,
    );
  }
  addBlocker(
    verification?.datasetRedistributionRightsConfirmed ===
      provenanceDatasets.every(
        (dataset) => dataset?.redistributionRightsConfirmed === true,
      ),
    "flag hak redistribusi tidak konsisten dengan provenance dataset",
  );
}

addBlocker(
  evidenceManifest.model?.sha256 === manifest.artifact?.sha256,
  "SHA model pada evidence manifest berbeda dari manifest aplikasi",
);
addBlocker(
  evidenceManifest.model?.version === manifest.artifact?.version,
  "versi model pada evidence manifest berbeda dari manifest aplikasi",
);
for (const [name, expectedHash] of Object.entries(
  evidenceManifest.evidenceFiles ?? {},
)) {
  const path = resolve(evidenceDirectory, name);
  const evidenceRelative = relative(evidenceDirectory, path);
  if (evidenceRelative.startsWith("..") || evidenceRelative === "") {
    blockers.push(`path evidence tidak aman: ${name}`);
    continue;
  }
  if (!existsSync(path)) {
    blockers.push(`evidence tidak ditemukan: ${name}`);
    continue;
  }
  addBlocker(sha256(path) === expectedHash, `hash evidence tidak cocok: ${name}`);
}

const trainingSources = manifest.provenance?.trainingSources;
if (Array.isArray(trainingSources) && trainingSources.length > 0) {
  const evidenceSources = evidenceManifest.trainingRun?.sourceFiles;
  addBlocker(
    Array.isArray(evidenceSources) &&
      evidenceSources.length === trainingSources.length,
    "daftar source training pada evidence tidak konsisten",
  );
  for (const source of trainingSources) {
    const sourcePath = resolveProjectPath(source?.path, "trainingSources.path");
    const evidenceSource = Array.isArray(evidenceSources)
      ? evidenceSources.find((item) => item?.path === source?.path)
      : undefined;
    addBlocker(
      typeof source?.sha256 === "string" && source.sha256 === evidenceSource?.sha256,
      `hash source training tidak konsisten: ${source?.path ?? "missing"}`,
    );
    addBlocker(
      typeof source?.sha256 === "string" && sha256(sourcePath) === source.sha256,
      `source training berubah: ${source?.path ?? "missing"}`,
    );
  }
} else {
  addBlocker(
    evidenceManifest.trainingRun?.codeCellsSha256 ===
      manifest.provenance?.trainingCodeCellsSha256,
    "fingerprint code cell training tidak konsisten",
  );
  const notebookPath = resolveProjectPath(
    manifest.provenance?.trainingNotebook,
    "provenance.trainingNotebook",
  );
  const notebook = readJson(notebookPath);
  const codeCellSources = notebook.cells
    .filter((cell) => cell.cell_type === "code")
    .map((cell) =>
      Array.isArray(cell.source) ? cell.source.join("") : cell.source,
    );
  const codeCellHash = createHash("sha256")
    .update(JSON.stringify(codeCellSources))
    .digest("hex");
  addBlocker(
    codeCellHash === evidenceManifest.trainingRun?.codeCellsSha256,
    "code cell notebook berubah dari training run yang dipromosikan",
  );
}

const trainingReportPath = resolveProjectPath(
  evidenceManifest.trainingRun?.reportFile,
  "trainingRun.reportFile",
);
addBlocker(existsSync(trainingReportPath), "training report belum tersedia");
if (existsSync(trainingReportPath)) {
  addBlocker(
    sha256(trainingReportPath) === evidenceManifest.trainingRun?.reportSha256,
    "hash training report tidak cocok",
  );
  const trainingReport = readJson(trainingReportPath);
  addBlocker(
    trainingReport.artifact?.tflite_sha256 === manifest.artifact?.sha256,
    "training report memakai model yang berbeda",
  );
}

const evaluationSources = evidenceManifest.evaluationRun?.sourceFiles;
addBlocker(
  Array.isArray(evaluationSources) && evaluationSources.length > 0,
  "source evaluator final belum tersedia",
);
if (Array.isArray(evaluationSources)) {
  for (const source of evaluationSources) {
    const sourcePath = resolveProjectPath(source?.path, "evaluationRun.sourceFiles.path");
    addBlocker(
      typeof source?.sha256 === "string" && sha256(sourcePath) === source.sha256,
      `source evaluator berubah: ${source?.path ?? "missing"}`,
    );
  }
}
const evaluationRequirementsPath = resolveProjectPath(
  evidenceManifest.evaluationRun?.requirementsFile,
  "evaluationRun.requirementsFile",
);
addBlocker(
  existsSync(evaluationRequirementsPath),
  "dependency evaluator final belum tersedia",
);
const externalEvaluationPath = resolveProjectPath(
  evidenceManifest.evaluationRun?.outputFile,
  "evaluationRun.outputFile",
);
addBlocker(
  existsSync(externalEvaluationPath),
  "output evaluator final belum tersedia",
);

const evidenceFiles = evidenceManifest.evidenceFiles ?? {};
const evaluation = readJson(resolve(evidenceDirectory, "evaluation_report.json"));
const nativeDevelopmentEvidenceName = "native-development-verification.json";
const nativeDevelopmentEvidencePath = resolve(
  evidenceDirectory,
  nativeDevelopmentEvidenceName,
);
const nativeDevelopmentEvidenceIsBound =
  typeof evidenceFiles[nativeDevelopmentEvidenceName] === "string" &&
  existsSync(nativeDevelopmentEvidencePath);
addBlocker(
  nativeDevelopmentEvidenceIsBound,
  "bukti self-test native post-crop belum diikat",
);
if (nativeDevelopmentEvidenceIsBound) {
  const nativeDevelopment = readJson(nativeDevelopmentEvidencePath);
  const nativeFixture = nativeDevelopment.nativePostCropFixture;
  const evaluationFixture = evaluation.native_post_crop_fixture;
  addBlocker(
    nativeDevelopment.productionEvidence === false,
    "bukti native development salah diklasifikasikan sebagai production evidence",
  );
  addBlocker(
    nativeDevelopment.loadedAssets?.model?.sha256 === manifest.artifact?.sha256,
    "self-test native memuat model yang berbeda",
  );
  addBlocker(
    nativeDevelopment.loadedAssets?.nativeSelfTestFixture?.sha256 ===
      manifest.nativeSelfTest?.sha256,
    "self-test native memuat fixture yang berbeda",
  );
  addBlocker(nativeFixture?.passed === true, "fixture native post-crop belum lulus");
  addBlocker(
    nativeFixture?.expectedLabel === manifest.nativeSelfTest?.expectedLabel &&
      nativeFixture?.actualLabel === manifest.nativeSelfTest?.expectedLabel,
    "label fixture native post-crop tidak cocok",
  );
  addBlocker(
    finiteAtLeast(
      nativeFixture?.actualConfidence,
      manifest.nativeSelfTest?.minimumConfidence,
    ),
    "confidence fixture native post-crop di bawah ambang",
  );
  addBlocker(
    evaluationFixture?.passed === nativeFixture?.passed &&
      evaluationFixture?.actual_label === nativeFixture?.actualLabel &&
      evaluationFixture?.confidence === nativeFixture?.actualConfidence,
    "ringkasan fixture native tidak konsisten dengan bukti runtime",
  );
}
const criteria = evidenceManifest.acceptanceCriteria ?? {};
const holdoutCriteria = criteria.physicalHoldout ?? {};
const holdout = evaluation.physical_holdout;
addBlocker(holdout !== null && typeof holdout === "object", "physical holdout belum tersedia");
addBlocker(
  evaluation.holdout_manifest_validation?.valid === true,
  "manifest physical holdout belum valid",
);
if (holdout) {
  addBlocker(
    finiteAtLeast(holdout.sample_count, holdoutCriteria.minimumSampleCount),
    "jumlah physical holdout di bawah ambang",
  );
  addBlocker(
    finiteAtLeast(holdout.accuracy, holdoutCriteria.minimumAccuracy),
    "akurasi physical holdout di bawah ambang",
  );
  addBlocker(
    finiteAtLeast(holdout.macro_f1, holdoutCriteria.minimumMacroF1),
    "macro-F1 physical holdout di bawah ambang",
  );

  const reportName = "physical_holdout_classification_report.json";
  const reportPath = resolve(evidenceDirectory, reportName);
  addBlocker(
    typeof evidenceFiles[reportName] === "string" && existsSync(reportPath),
    `${reportName} belum diikat ke evidence manifest`,
  );
  if (typeof evidenceFiles[reportName] === "string" && existsSync(reportPath)) {
    const report = readJson(reportPath);
    for (const label of "abcdefghijklmnopqrstuvwxyz") {
      addBlocker(
        finiteAtLeast(report[label]?.support, holdoutCriteria.minimumSamplesPerClass),
        `support physical holdout kelas ${label} di bawah ambang`,
      );
      addBlocker(
        finiteAtLeast(report[label]?.recall, holdoutCriteria.minimumPerClassRecall),
        `recall physical holdout kelas ${label} di bawah ambang`,
      );
    }
  }
}

const negativeCriteria = criteria.negativeCorpus ?? {};
const negative = evaluation.negative_guardrail;
addBlocker(negative !== null && typeof negative === "object", "negative corpus belum tersedia");
if (negative) {
  addBlocker(
    finiteAtLeast(negative.sample_count, negativeCriteria.minimumSceneCount),
    "jumlah negative scene di bawah ambang",
  );
  addBlocker(
    finiteAtMost(
      negative.false_accept_count,
      negativeCriteria.maximumFalseAcceptCount,
    ),
    "false accept negative corpus melewati ambang",
  );
  addBlocker(
    negative.confidence_threshold === negativeCriteria.confidenceThreshold,
    "confidence threshold negative corpus berbeda dari kontrak release",
  );
}

addBlocker(
  evaluation.keras_tflite_golden_parity?.passed === true &&
    evaluation.keras_tflite_golden_parity?.sample_count >= 26,
  "golden conversion parity A-Z belum lulus",
);
addBlocker(
  evaluation.native_post_crop_fixture?.passed === true,
  "self-test native post-crop belum lulus",
);

const expectedCommit = evidenceManifest.trainingRun?.gitCommit;
addBlocker(
  typeof expectedCommit === "string" && /^[0-9a-f]{40}$/u.test(expectedCommit),
  "commit training belum diikat ke evidence",
);
const gitResult = spawnSync("git", ["rev-parse", "HEAD"], {
  cwd: projectRoot,
  encoding: "utf8",
});
const currentCommit = gitResult.status === 0 ? gitResult.stdout.trim() : null;
addBlocker(currentCommit !== null, "repository belum memiliki commit HEAD");
if (currentCommit && typeof expectedCommit === "string") {
  addBlocker(currentCommit === expectedCommit, "commit release berbeda dari commit evidence");
}

const deviceEvidence = evidenceManifest.requiredReleaseEvidence ?? {};
const devicePath = resolveProjectPath(
  deviceEvidence.deviceValidationFile,
  "requiredReleaseEvidence.deviceValidationFile",
);
addBlocker(
  typeof deviceEvidence.deviceValidationSha256 === "string",
  "SHA device validation belum diikat",
);
addBlocker(existsSync(devicePath), "device validation belum tersedia");
if (
  existsSync(devicePath) &&
  typeof deviceEvidence.deviceValidationSha256 === "string"
) {
  addBlocker(
    sha256(devicePath) === deviceEvidence.deviceValidationSha256,
    "hash device validation tidak cocok",
  );
  const device = readJson(devicePath);
  const runtimeCriteria = criteria.nativeRuntime ?? {};
  addBlocker(device.modelSha256 === manifest.artifact.sha256, "device test memakai model berbeda");
  addBlocker(device.gitCommit === expectedCommit, "device test memakai commit berbeda");
  addBlocker(
    new Set((device.devices ?? []).map((item) => item.id)).size >=
      runtimeCriteria.minimumPhysicalDeviceCount,
    "jumlah perangkat fisik di bawah ambang",
  );
  addBlocker(
    device.nativeGoldenParity?.sampleCount >= 26 &&
      device.nativeGoldenParity?.top1MatchRate === 1,
    "native camera golden parity A-Z belum lulus",
  );
  addBlocker(
    finiteAtLeast(device.runtime?.minimumPreviewFps, runtimeCriteria.minimumPreviewFps),
    "preview FPS perangkat di bawah ambang",
  );
  addBlocker(
    finiteAtMost(
      device.runtime?.p95StableLatencyMs,
      runtimeCriteria.maximumP95StableLatencyMs,
    ),
    "p95 latency perangkat melewati ambang",
  );
  addBlocker(
    finiteAtLeast(device.runtime?.soakMinutes, runtimeCriteria.minimumSoakMinutes),
    "durasi soak di bawah ambang",
  );
  addBlocker(device.runtime?.crashCount === 0, "crash ditemukan pada device validation");
  addBlocker(device.runtime?.anrCount === 0, "ANR ditemukan pada device validation");
  addBlocker(
    device.runtime?.falseUtteranceCount === 0,
    "false utterance ditemukan pada device validation",
  );
  addBlocker(device.offline?.airplaneModePassed === true, "uji mode pesawat belum lulus");
  addBlocker(device.offline?.coldStartPassed === true, "uji cold start offline belum lulus");
  addBlocker(device.accessibility?.talkBackPassed === true, "uji TalkBack belum lulus");
  addBlocker(device.accessibility?.fontScale200Passed === true, "uji font 200% belum lulus");
  addBlocker(
    device.accessibility?.nonVisualRecoveryPassed === true,
    "recovery nonvisual belum lulus",
  );
}

if (blockers.length > 0) {
  throw new Error(
    `Production ditolak oleh evidence gate:\n- ${blockers.join("\n- ")}`,
  );
}

console.log(
  `Model release verified: ${manifest.artifact.version}, commit ${currentCommit}, seluruh evidence gate lulus.`,
);
