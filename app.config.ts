import type { ConfigContext } from "expo/config";

declare const process: {
  env: Record<string, string | undefined>;
};

const standaloneBlockedPermissions = [
  "android.permission.INTERNET",
  "android.permission.ACCESS_NETWORK_STATE",
  "android.permission.READ_EXTERNAL_STORAGE",
  "android.permission.WRITE_EXTERNAL_STORAGE",
  "android.permission.SYSTEM_ALERT_WINDOW",
  "android.permission.READ_PHONE_STATE",
];

export default ({ config }: ConfigContext) => {
  const buildProfile = process.env.EAS_BUILD_PROFILE;
  const appEnvironment =
    process.env.APP_ENV === "production" || buildProfile === "production"
      ? "production"
      : process.env.APP_ENV === "preview" ||
          buildProfile === "preview"
        ? "preview"
        : "development";
  const isStandaloneBuild =
    appEnvironment === "preview" || appEnvironment === "production";

  return {
    ...config,
    extra: {
      ...config.extra,
      appEnvironment,
    },
    android: {
      ...config.android,
      allowBackup: false,
      ...(isStandaloneBuild
        ? { blockedPermissions: standaloneBlockedPermissions }
        : {}),
    },
  };
};
