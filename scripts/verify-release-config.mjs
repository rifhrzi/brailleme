import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const expoCli = join(dirname(require.resolve("expo/package.json")), "bin", "cli");

const expectedBlockedPermissions = [
  "android.permission.INTERNET",
  "android.permission.ACCESS_NETWORK_STATE",
  "android.permission.READ_EXTERNAL_STORAGE",
  "android.permission.WRITE_EXTERNAL_STORAGE",
  "android.permission.SYSTEM_ALERT_WINDOW",
  "android.permission.READ_PHONE_STATE",
];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function loadExpoConfig(
  buildProfile,
  {
    appEnvironment = buildProfile,
    type = "public",
  } = {},
) {
  const result = spawnSync(
    process.execPath,
    [expoCli, "config", "--type", type, "--json"],
    {
      cwd: projectRoot,
      encoding: "utf8",
      env: {
        ...process.env,
        APP_ENV: appEnvironment,
        EAS_BUILD_PROFILE: buildProfile,
      },
    },
  );

  if (result.status !== 0) {
    throw new Error(result.stderr || `Expo config failed for ${buildProfile}`);
  }

  return JSON.parse(result.stdout);
}

const developmentConfig = loadExpoConfig("development");
const previewConfig = loadExpoConfig("preview");
const productionConfig = loadExpoConfig("production");
const previewIntrospection = loadExpoConfig("preview", { type: "introspect" });
const productionIntrospection = loadExpoConfig("production", {
  type: "introspect",
});
const easConfig = JSON.parse(
  readFileSync(join(projectRoot, "eas.json"), "utf8"),
);
const packageConfig = JSON.parse(
  readFileSync(join(projectRoot, "package.json"), "utf8"),
);
const packageLockConfig = JSON.parse(
  readFileSync(join(projectRoot, "package-lock.json"), "utf8"),
);
const doctorWrapper = readFileSync(
  join(projectRoot, "scripts", "run-expo-doctor.mjs"),
  "utf8",
);
const submitWrapper = readFileSync(
  join(projectRoot, "scripts", "submit-eas-build.mjs"),
  "utf8",
);
const easBuildVerifier = readFileSync(
  join(projectRoot, "scripts", "verify-eas-build-profile.mjs"),
  "utf8",
);
const qualityWorkflow = readFileSync(
  join(projectRoot, ".github", "workflows", "quality.yml"),
  "utf8",
);

for (const removedPath of [
  "app/debug.tsx",
  "app/model-probe.tsx",
  "src/features/debug/debug-screen.tsx",
  "src/features/debug/__tests__/debug-screen.test.tsx",
  "src/features/model-probe/model-probe-screen.tsx",
  "src/config/feature-flags.ts",
  "src/types/alpha-preferences.ts",
  "src/storage/alpha-preferences-schema.ts",
  "src/storage/alpha-preferences-store.ts",
  "assets/icon.png",
  "assets/models/braille-cell-classifier-v0/braille_model.tflite",
  "assets/models/braille-cell-classifier-v0/label.json",
  "assets/models/braille-cell-classifier-v0/manifest.json",
  "scripts/verify-experimental-model-release.mjs",
  "docs/release/EXPERIMENTAL-MODEL-RISK-ACCEPTANCE.json",
  "BrailleMe/PRD - BrailleMe.md",
  "BrailleMe/Roadmap - BrailleMe - Technical Alpha.canvas",
  "artifacts/development-build-report.md",
  "artifacts/readiness-audit-2026-08-20.md",
]) {
  assert(
    !existsSync(join(projectRoot, removedPath)),
    `Obsolete development artifact must stay removed: ${removedPath}`,
  );
}

assert(
  developmentConfig.android?.allowBackup === false,
  "Android backup must be disabled in development",
);
assert(
  previewConfig.android?.allowBackup === false,
  "Android backup must be disabled in preview",
);
assert(
  productionConfig.android?.allowBackup === false,
  "Android backup must be disabled in production",
);
assert(
  !developmentConfig.android?.blockedPermissions?.includes(
    "android.permission.INTERNET",
  ),
  "Development must retain INTERNET for Metro",
);
for (const permission of expectedBlockedPermissions) {
  assert(
    previewConfig.android?.blockedPermissions?.includes(permission),
    `Preview must block ${permission}`,
  );
  assert(
    productionConfig.android?.blockedPermissions?.includes(permission),
    `Production must block ${permission}`,
  );
}
const previewManifest =
  previewIntrospection._internal?.modResults?.android?.manifest?.manifest;
const productionManifest =
  productionIntrospection._internal?.modResults?.android?.manifest?.manifest;
for (const [buildName, manifest] of [
  ["Preview", previewManifest],
  ["Production", productionManifest],
]) {
  const manifestPermissions = manifest?.["uses-permission"] ?? [];
  for (const permission of expectedBlockedPermissions) {
    assert(
      manifestPermissions.some(
        (entry) =>
          entry.$?.["android:name"] === permission &&
          entry.$?.["tools:node"] === "remove",
      ),
      `${buildName} native manifest must remove ${permission}`,
    );
  }
  assert(
    manifest?.application?.[0]?.$?.["android:allowBackup"] === "false",
    `${buildName} native manifest must disable Android backup`,
  );
}
const mainActivity = productionManifest?.application?.[0]?.activity?.find(
  (activity) => activity.$?.["android:name"] === ".MainActivity",
);
assert(
  mainActivity?.$?.["android:screenOrientation"] === "unspecified",
  "Native production activity must keep orientation unlocked",
);
assert(
  previewConfig.android?.permissions?.includes("android.permission.CAMERA"),
  "Preview must retain CAMERA",
);
assert(
  productionConfig.android?.permissions?.includes("android.permission.CAMERA"),
  "Production must retain CAMERA",
);
assert(
  productionConfig.orientation === "default",
  "Production orientation must remain unlocked",
);
assert(
  developmentConfig.extra?.appEnvironment === "development" &&
    previewConfig.extra?.appEnvironment === "preview" &&
    productionConfig.extra?.appEnvironment === "production",
  "Runtime appEnvironment must identify every build profile",
);
assert(
  !Object.prototype.hasOwnProperty.call(
    productionConfig.extra ?? {},
    "allowUnverifiedModel",
  ),
  "Production runtime config must not expose an unverified-model override",
);
assert(
  productionConfig.icon === "./assets/icon-brand.png" &&
    productionConfig.android?.adaptiveIcon?.foregroundImage ===
      "./assets/adaptive-icon.png" &&
    productionConfig.android?.adaptiveIcon?.monochromeImage ===
      "./assets/monochrome-icon.png" &&
    productionConfig.android?.adaptiveIcon?.backgroundColor === "#006B63",
  "Production launcher and adaptive icons must use the approved BrailleMe assets",
);
assert(
  easConfig.cli?.version === "22.0.0" && easConfig.cli?.requireCommit === true,
  "EAS CLI must be pinned and require a committed source tree",
);
assert(
  packageConfig.scripts?.["eas-build-post-install"] ===
    "node scripts/verify-eas-build-profile.mjs",
  "EAS post-install must enforce the production release gate",
);
for (const [dependency, version] of [
  ["eas-cli", "22.0.0"],
  ["expo-doctor", "1.20.2"],
  ["metro", "0.84.5"],
  ["metro-config", "0.84.5"],
  ["metro-transform-worker", "0.84.5"],
]) {
  assert(
    packageConfig.devDependencies?.[dependency] === version &&
      packageLockConfig.packages?.[""]?.devDependencies?.[dependency] ===
        version &&
      packageLockConfig.packages?.[`node_modules/${dependency}`]?.version ===
        version,
    `${dependency} must be installed and locked exactly to ${version}`,
  );
}
for (const scriptName of [
  "build:dev:android",
  "build:preview:android",
  "build:production:android",
]) {
  const script = packageConfig.scripts?.[scriptName] ?? "";
  assert(
    /(?:^|&& )eas (?:build|submit) /.test(script) &&
      !/\bnpx(?:\.cmd)?\b|eas-cli@/.test(script),
    `${scriptName} must execute the locked local EAS CLI`,
  );
}
assert(
  packageConfig.scripts?.["submit:internal:android"]?.endsWith(
    "node scripts/submit-eas-build.mjs internal",
  ) &&
    packageConfig.scripts?.["submit:production:android"]?.endsWith(
      "node scripts/submit-eas-build.mjs production",
    ) &&
    submitWrapper.includes('"--id",') &&
    !submitWrapper.includes('"--latest"') &&
    submitWrapper.includes("build.gitCommitHash !== currentCommit") &&
    submitWrapper.includes("build.project?.id !== projectId"),
  "Submissions must bind an exact EAS build ID, project, and source commit",
);
const packageScripts = packageConfig.scripts ?? {};
assert(
  !Object.prototype.hasOwnProperty.call(
    easConfig.build ?? {},
    "production-experimental",
  ) &&
    !Object.prototype.hasOwnProperty.call(
      packageScripts,
      "build:production-experimental:android",
    ) &&
    !Object.prototype.hasOwnProperty.call(
      packageScripts,
      "submit:production-experimental:android",
    ) &&
    !Object.prototype.hasOwnProperty.call(
      packageScripts,
      "release:experimental:verify",
    ) &&
    !JSON.stringify(easConfig).includes("ALLOW_UNVERIFIED_MODEL") &&
    !submitWrapper.includes("production-experimental") &&
    !easBuildVerifier.includes("release:experimental"),
  "Release configuration must not expose an unverified production path",
);
assert(
  packageConfig.scripts?.doctor === "node scripts/run-expo-doctor.mjs" &&
    doctorWrapper.includes('require.resolve("expo-doctor/package.json")') &&
    !/expo-doctor@latest|npm\s+exec|\bnpx(?:\.cmd)?\b/.test(doctorWrapper),
  "Expo Doctor must execute the locked local expo-doctor package",
);
const checkoutSha = "11bd71901bbe5b1630ceea73d27597364c9af683";
const setupNodeSha = "49933ea5288caeca8642d1e84afbd3f7d6820020";
assert(
  qualityWorkflow.includes(`uses: actions/checkout@${checkoutSha}`) &&
    qualityWorkflow.includes(`uses: actions/setup-node@${setupNodeSha}`) &&
    !/uses:\s*actions\/(?:checkout|setup-node)@v\d/.test(qualityWorkflow),
  "GitHub Actions dependencies must remain pinned to reviewed full SHAs",
);
assert(
  new RegExp(
    `uses: actions/checkout@${checkoutSha}[^\\r\\n]*\\r?\\n\\s+with:\\r?\\n\\s+persist-credentials:\\s*false`,
  ).test(qualityWorkflow),
  "Checkout credentials must not persist after the checkout step",
);
assert(
  packageConfig.scripts?.["expo:dependencies:check"] ===
      "expo install --check" &&
    qualityWorkflow.includes("run: npm run expo:dependencies:check") &&
    !/\brun:\s*npx\b/.test(qualityWorkflow),
  "CI dependency checks must execute the locked local Expo CLI",
);
assert(
  easConfig.build?.preview?.android?.buildType === "apk" &&
    easConfig.build?.preview?.distribution === "internal" &&
    easConfig.build?.preview?.env?.APP_ENV === "preview",
  "Preview must be an internally distributed APK",
);
assert(
  easConfig.build?.production?.android?.buildType === "app-bundle" &&
    easConfig.build?.production?.distribution === "store" &&
    easConfig.build?.production?.env?.APP_ENV === "production" &&
    !Object.prototype.hasOwnProperty.call(
      easConfig.build?.production?.env ?? {},
      "ALLOW_UNVERIFIED_MODEL",
    ),
  "Production must be a store-distributed AAB",
);
assert(
  easConfig.build?.development?.env?.APP_ENV === "development",
  "Development must select development app configuration",
);
assert(
  easConfig.submit?.internal?.android?.track === "internal" &&
    easConfig.submit?.internal?.android?.releaseStatus === "draft" &&
    easConfig.submit?.internal?.android?.changesNotSentForReview === true,
  "Internal submission must remain an unreviewed internal-track draft",
);
assert(
  easConfig.submit?.production?.android?.track === "production" &&
    easConfig.submit?.production?.android?.releaseStatus === "draft" &&
    easConfig.submit?.production?.android?.changesNotSentForReview === true,
  "Production submission must remain an unreviewed production-track draft",
);

console.log(
  "Release config valid: preview APK, fail-closed verified production AAB, least-privilege permissions, pinned tooling, and draft submits.",
);
