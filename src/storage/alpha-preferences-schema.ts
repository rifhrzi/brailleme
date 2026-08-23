import {
  ALPHA_BRAILLE_PROFILE,
  ALPHA_PREFERENCES_SCHEMA_VERSION,
  ALPHA_SPEECH_MODE,
  type AlphaPreferences,
} from "@/types";

export const ALPHA_PREFERENCES_DEFAULTS: AlphaPreferences = Object.freeze({
  schemaVersion: ALPHA_PREFERENCES_SCHEMA_VERSION,
  onboardingCompleted: false,
  profile: ALPHA_BRAILLE_PROFILE,
  speechMode: ALPHA_SPEECH_MODE,
  speechRate: 1.0,
  speechVolume: 1.0,
  muted: false,
  hapticsEnabled: true,
  debugMenuEnabled: false,
});

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function readFiniteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clampToTenth(value: number, minimum: number, maximum: number): number {
  const clamped = Math.min(maximum, Math.max(minimum, value));
  return Math.round(clamped * 10) / 10;
}

function normalizeCurrentSchema(value: Record<string, unknown>): AlphaPreferences {
  return Object.freeze({
    schemaVersion: ALPHA_PREFERENCES_SCHEMA_VERSION,
    onboardingCompleted: readBoolean(
      value.onboardingCompleted,
      ALPHA_PREFERENCES_DEFAULTS.onboardingCompleted,
    ),
    profile: ALPHA_BRAILLE_PROFILE,
    speechMode: ALPHA_SPEECH_MODE,
    speechRate: clampToTenth(
      readFiniteNumber(
        value.speechRate,
        ALPHA_PREFERENCES_DEFAULTS.speechRate,
      ),
      0.5,
      2.0,
    ),
    speechVolume: clampToTenth(
      readFiniteNumber(
        value.speechVolume,
        ALPHA_PREFERENCES_DEFAULTS.speechVolume,
      ),
      0,
      1,
    ),
    muted: readBoolean(value.muted, ALPHA_PREFERENCES_DEFAULTS.muted),
    hapticsEnabled: readBoolean(
      value.hapticsEnabled,
      ALPHA_PREFERENCES_DEFAULTS.hapticsEnabled,
    ),
    debugMenuEnabled: readBoolean(
      value.debugMenuEnabled,
      ALPHA_PREFERENCES_DEFAULTS.debugMenuEnabled,
    ),
  });
}

/**
 * Migrasi defensif untuk bentuk preferensi awal sebelum schemaVersion 1.
 * Alias hanya mencakup nama preference sederhana; field lain selalu dibuang.
 */
function migrateLegacySchema(value: Record<string, unknown>): AlphaPreferences {
  const legacyVolume = readFiniteNumber(
    value.speechVolume ?? value.volume ?? value.ttsVolume,
    ALPHA_PREFERENCES_DEFAULTS.speechVolume,
  );

  return normalizeCurrentSchema({
    onboardingCompleted:
      value.onboardingCompleted ?? value.hasCompletedOnboarding,
    speechRate: value.speechRate ?? value.rate ?? value.ttsRate,
    speechVolume:
      legacyVolume > 1 && legacyVolume <= 100
        ? legacyVolume / 100
        : legacyVolume,
    muted: value.muted ?? value.speechMuted,
    hapticsEnabled: value.hapticsEnabled ?? value.haptics,
    debugMenuEnabled: false,
  });
}

/** Memvalidasi, memigrasikan, membatasi rentang, dan membuang field asing. */
export function normalizeAlphaPreferences(value: unknown): AlphaPreferences {
  if (!isRecord(value)) {
    return ALPHA_PREFERENCES_DEFAULTS;
  }

  if (value.schemaVersion === ALPHA_PREFERENCES_SCHEMA_VERSION) {
    return normalizeCurrentSchema(value);
  }

  if (value.schemaVersion === 1) {
    return normalizeCurrentSchema({
      ...value,
      debugMenuEnabled: false,
    });
  }

  if (value.schemaVersion === undefined || value.schemaVersion === 0) {
    return migrateLegacySchema(value);
  }

  // Versi masa depan tidak ditafsirkan secara spekulatif.
  return ALPHA_PREFERENCES_DEFAULTS;
}

export function areAlphaPreferencesEqual(
  left: AlphaPreferences,
  right: AlphaPreferences,
): boolean {
  return (
    left.schemaVersion === right.schemaVersion &&
    left.onboardingCompleted === right.onboardingCompleted &&
    left.profile === right.profile &&
    left.speechMode === right.speechMode &&
    left.speechRate === right.speechRate &&
    left.speechVolume === right.speechVolume &&
    left.muted === right.muted &&
    left.hapticsEnabled === right.hapticsEnabled &&
    left.debugMenuEnabled === right.debugMenuEnabled
  );
}
