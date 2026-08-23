export const ALPHA_PREFERENCES_SCHEMA_VERSION = 2 as const;
export const ALPHA_BRAILLE_PROFILE = "id-latin-6dot-alpha-v0" as const;
export const ALPHA_SPEECH_MODE = "press" as const;

export type AlphaBrailleProfile = typeof ALPHA_BRAILLE_PROFILE;
export type AlphaSpeechMode = typeof ALPHA_SPEECH_MODE;

/** Satu-satunya data aplikasi yang dipersistensikan pada milestone frontend. */
export interface AlphaPreferences {
  readonly schemaVersion: typeof ALPHA_PREFERENCES_SCHEMA_VERSION;
  readonly onboardingCompleted: boolean;
  readonly profile: AlphaBrailleProfile;
  readonly speechMode: AlphaSpeechMode;
  readonly speechRate: number;
  readonly speechVolume: number;
  readonly muted: boolean;
  readonly hapticsEnabled: boolean;
  readonly debugMenuEnabled: boolean;
}

/** Preferensi yang memang boleh diubah pengguna pada milestone ini. */
export type AlphaPreferencesPatch = Partial<
  Pick<
    AlphaPreferences,
    | "onboardingCompleted"
    | "speechRate"
    | "speechVolume"
    | "muted"
    | "hapticsEnabled"
    | "debugMenuEnabled"
  >
>;
