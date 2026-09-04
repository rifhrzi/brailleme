export const APP_PREFERENCES_SCHEMA_VERSION = 4 as const;
export const APP_BRAILLE_PROFILE = "id-latin-6dot-v1" as const;
export const APP_SPEECH_MODE = "press" as const;

export type AppBrailleProfile = typeof APP_BRAILLE_PROFILE;
export type AppSpeechMode = typeof APP_SPEECH_MODE;

/** Preferensi pengguna yang dipersistensikan aplikasi. */
export interface AppPreferences {
  readonly schemaVersion: typeof APP_PREFERENCES_SCHEMA_VERSION;
  readonly onboardingCompleted: boolean;
  readonly profile: AppBrailleProfile;
  readonly speechMode: AppSpeechMode;
  readonly speechRate: number;
  readonly speechVolume: number;
  readonly muted: boolean;
  readonly hapticsEnabled: boolean;
}

/** Preferensi yang boleh diubah pengguna. */
export type AppPreferencesPatch = Partial<
  Pick<
    AppPreferences,
    | "onboardingCompleted"
    | "speechRate"
    | "speechVolume"
    | "muted"
    | "hapticsEnabled"
  >
>;
