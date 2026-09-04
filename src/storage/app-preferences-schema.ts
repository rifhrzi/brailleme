import {
  APP_BRAILLE_PROFILE,
  APP_PREFERENCES_SCHEMA_VERSION,
  APP_SPEECH_MODE,
  type AppPreferences,
} from "@/types";

export const APP_PREFERENCES_DEFAULTS: AppPreferences = Object.freeze({
  schemaVersion: APP_PREFERENCES_SCHEMA_VERSION,
  onboardingCompleted: false,
  profile: APP_BRAILLE_PROFILE,
  speechMode: APP_SPEECH_MODE,
  speechRate: 1.0,
  speechVolume: 1.0,
  muted: false,
  hapticsEnabled: true,
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

function normalizeCurrentSchema(value: Record<string, unknown>): AppPreferences {
  return Object.freeze({
    schemaVersion: APP_PREFERENCES_SCHEMA_VERSION,
    onboardingCompleted: readBoolean(
      value.onboardingCompleted,
      APP_PREFERENCES_DEFAULTS.onboardingCompleted,
    ),
    profile: APP_BRAILLE_PROFILE,
    speechMode: APP_SPEECH_MODE,
    speechRate: clampToTenth(
      readFiniteNumber(
        value.speechRate,
        APP_PREFERENCES_DEFAULTS.speechRate,
      ),
      0.5,
      2.0,
    ),
    speechVolume: clampToTenth(
      readFiniteNumber(
        value.speechVolume,
        APP_PREFERENCES_DEFAULTS.speechVolume,
      ),
      0,
      1,
    ),
    muted: readBoolean(value.muted, APP_PREFERENCES_DEFAULTS.muted),
    hapticsEnabled: readBoolean(
      value.hapticsEnabled,
      APP_PREFERENCES_DEFAULTS.hapticsEnabled,
    ),
  });
}

/**
 * Migrasi defensif untuk bentuk preferensi awal sebelum schemaVersion 1.
 * Alias hanya mencakup nama preference sederhana; field lain selalu dibuang.
 */
function migrateLegacySchema(value: Record<string, unknown>): AppPreferences {
  const legacyVolume = readFiniteNumber(
    value.speechVolume ?? value.volume ?? value.ttsVolume,
    APP_PREFERENCES_DEFAULTS.speechVolume,
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
  });
}

/** Memvalidasi, memigrasikan, membatasi rentang, dan membuang field asing. */
export function normalizeAppPreferences(value: unknown): AppPreferences {
  if (!isRecord(value)) {
    return APP_PREFERENCES_DEFAULTS;
  }

  if (value.schemaVersion === APP_PREFERENCES_SCHEMA_VERSION) {
    return normalizeCurrentSchema(value);
  }

  if (
    value.schemaVersion === 1 ||
    value.schemaVersion === 2 ||
    value.schemaVersion === 3
  ) {
    return normalizeCurrentSchema(value);
  }

  if (value.schemaVersion === undefined || value.schemaVersion === 0) {
    return migrateLegacySchema(value);
  }

  // Versi masa depan tidak ditafsirkan secara spekulatif.
  return APP_PREFERENCES_DEFAULTS;
}

export function areAppPreferencesEqual(
  left: AppPreferences,
  right: AppPreferences,
): boolean {
  return (
    left.schemaVersion === right.schemaVersion &&
    left.onboardingCompleted === right.onboardingCompleted &&
    left.profile === right.profile &&
    left.speechMode === right.speechMode &&
    left.speechRate === right.speechRate &&
    left.speechVolume === right.speechVolume &&
    left.muted === right.muted &&
    left.hapticsEnabled === right.hapticsEnabled
  );
}
