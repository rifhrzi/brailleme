import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const doctorPackagePath = require.resolve("expo-doctor/package.json");
const doctorPackage = JSON.parse(readFileSync(doctorPackagePath, "utf8"));
const easPackagePath = require.resolve("eas-cli/package.json");
const easPackage = JSON.parse(readFileSync(easPackagePath, "utf8"));
const doctorBin =
  typeof doctorPackage.bin === "string"
    ? doctorPackage.bin
    : doctorPackage.bin?.["expo-doctor"];

if (!doctorBin) {
  throw new Error("Binary expo-doctor lokal tidak ditemukan.");
}

const doctorCli = resolve(dirname(doctorPackagePath), doctorBin);

const transientFailure =
  /unexpected server response|ECONNRESET|ETIMEDOUT|EAI_AGAIN|network/i;

function isOnlyLockedEasPolicyConflict(output) {
  const failedChecks = [...output.matchAll(/^\u2716 (.+)\r?$/gm)].map(
    (match) => match[1],
  );

  return (
    doctorPackage.version === "1.20.2" &&
    easPackage.version === "22.0.0" &&
    failedChecks.length === 1 &&
    failedChecks[0] === "Check for legacy global CLI installed locally" &&
    output.includes("EAS CLI should not be installed in your project") &&
    !output.includes("Expo CLI is now part of the expo package")
  );
}

for (let attempt = 1; attempt <= 3; attempt += 1) {
  const result = spawnSync(
    process.execPath,
    [doctorCli],
    {
      cwd: projectRoot,
      encoding: "utf8",
      stdio: ["inherit", "pipe", "pipe"],
    },
  );
  const output = `${result.stdout ?? ""}${result.stderr ?? ""}`;
  process.stdout.write(output);

  if (result.status === 0) {
    process.exit(0);
  }
  if (isOnlyLockedEasPolicyConflict(output)) {
    console.warn(
      "Expo Doctor policy exception: the only failed check rejects project-local EAS CLI, which this project intentionally locks to eas-cli@22.0.0. All other Doctor checks passed.",
    );
    process.exit(0);
  }
  if (!transientFailure.test(output) || attempt === 3) {
    process.exit(result.status ?? 1);
  }

  console.warn(`Expo Doctor gagal sementara; mencoba lagi (${attempt}/3).`);
}
