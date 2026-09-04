import {
  APP_PREFERENCES_DEFAULTS,
  normalizeAppPreferences,
} from "@/storage/app-preferences-schema";

describe("normalizeAppPreferences", () => {
  it("menggunakan default yang dikunci untuk input rusak", () => {
    expect(normalizeAppPreferences(null)).toEqual(
      APP_PREFERENCES_DEFAULTS,
    );
    expect(normalizeAppPreferences("bukan object")).toEqual(
      APP_PREFERENCES_DEFAULTS,
    );
  });

  it("membatasi rate dan volume serta mengunci profil dan mode", () => {
    expect(
      normalizeAppPreferences({
        schemaVersion: 4,
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
      schemaVersion: 4,
      onboardingCompleted: true,
      profile: "id-latin-6dot-v1",
      speechMode: "press",
      speechRate: 2,
      speechVolume: 0,
      muted: true,
      hapticsEnabled: false,
    });
  });

  it("memigrasikan bentuk legacy dan volume persen", () => {
    expect(
      normalizeAppPreferences({
        hasCompletedOnboarding: true,
        rate: 0.74,
        volume: 60,
        speechMuted: true,
        haptics: false,
      }),
    ).toEqual({
      schemaVersion: 4,
      onboardingCompleted: true,
      profile: "id-latin-6dot-v1",
      speechMode: "press",
      speechRate: 0.7,
      speechVolume: 0.6,
      muted: true,
      hapticsEnabled: false,
    });
  });

  it.each([1, 2, 3])(
    "memigrasikan schema %s dan membuang field debug lama",
    (schemaVersion) => {
      expect(
        normalizeAppPreferences({
          schemaVersion,
          onboardingCompleted: true,
          speechRate: 1.2,
          speechVolume: 0.8,
          muted: false,
          hapticsEnabled: true,
          debugMenuEnabled: true,
        }),
      ).toEqual({
        schemaVersion: 4,
        onboardingCompleted: true,
        profile: "id-latin-6dot-v1",
        speechMode: "press",
        speechRate: 1.2,
        speechVolume: 0.8,
        muted: false,
        hapticsEnabled: true,
      });
    },
  );

  it("tidak menafsirkan schema masa depan", () => {
    expect(
      normalizeAppPreferences({
        schemaVersion: 99,
        onboardingCompleted: true,
      }),
    ).toEqual(APP_PREFERENCES_DEFAULTS);
  });
});
