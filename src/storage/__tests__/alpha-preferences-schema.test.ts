import {
  ALPHA_PREFERENCES_DEFAULTS,
  normalizeAlphaPreferences,
} from "@/storage/alpha-preferences-schema";

describe("normalizeAlphaPreferences", () => {
  it("menggunakan default yang dikunci untuk input rusak", () => {
    expect(normalizeAlphaPreferences(null)).toEqual(
      ALPHA_PREFERENCES_DEFAULTS,
    );
    expect(normalizeAlphaPreferences("bukan object")).toEqual(
      ALPHA_PREFERENCES_DEFAULTS,
    );
  });

  it("membatasi rate dan volume serta mengunci profil dan mode", () => {
    expect(
      normalizeAlphaPreferences({
        schemaVersion: 2,
        onboardingCompleted: true,
        profile: "profil-tidak-didukung",
        speechMode: "auto",
        speechRate: 5,
        speechVolume: -1,
        muted: true,
        hapticsEnabled: false,
        debugMenuEnabled: true,
        scanText: "tidak boleh dipersistensikan",
      }),
    ).toEqual({
      schemaVersion: 2,
      onboardingCompleted: true,
      profile: "id-latin-6dot-alpha-v0",
      speechMode: "press",
      speechRate: 2,
      speechVolume: 0,
      muted: true,
      hapticsEnabled: false,
      debugMenuEnabled: true,
    });
  });

  it("memigrasikan bentuk legacy dan volume persen", () => {
    expect(
      normalizeAlphaPreferences({
        hasCompletedOnboarding: true,
        rate: 0.74,
        volume: 60,
        speechMuted: true,
        haptics: false,
      }),
    ).toEqual({
      schemaVersion: 2,
      onboardingCompleted: true,
      profile: "id-latin-6dot-alpha-v0",
      speechMode: "press",
      speechRate: 0.7,
      speechVolume: 0.6,
      muted: true,
      hapticsEnabled: false,
      debugMenuEnabled: false,
    });
  });

  it("memigrasikan schema 1 dengan Debug Menu tetap nonaktif", () => {
    expect(
      normalizeAlphaPreferences({
        schemaVersion: 1,
        onboardingCompleted: true,
        speechRate: 1.2,
        speechVolume: 0.8,
        muted: false,
        hapticsEnabled: true,
      }),
    ).toEqual({
      schemaVersion: 2,
      onboardingCompleted: true,
      profile: "id-latin-6dot-alpha-v0",
      speechMode: "press",
      speechRate: 1.2,
      speechVolume: 0.8,
      muted: false,
      hapticsEnabled: true,
      debugMenuEnabled: false,
    });
  });

  it("tidak menafsirkan schema masa depan", () => {
    expect(
      normalizeAlphaPreferences({
        schemaVersion: 99,
        onboardingCompleted: true,
      }),
    ).toEqual(ALPHA_PREFERENCES_DEFAULTS);
  });
});
