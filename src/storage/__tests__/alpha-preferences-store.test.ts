describe('persistensi AlphaPreferences', () => {
  it('memuat kembali preferensi yang sama setelah module restart', () => {
    let firstStore!: typeof import('../alpha-preferences-store');
    jest.isolateModules(() => {
      firstStore = jest.requireActual('../alpha-preferences-store');
    });

    firstStore.updateAlphaPreferences({
      onboardingCompleted: true,
      speechRate: 1.7,
      speechVolume: 0.4,
      muted: true,
      hapticsEnabled: false,
      debugMenuEnabled: true,
    });

    const persistedBeforeRestart = globalThis.localStorage.getItem(
      firstStore.ALPHA_PREFERENCES_STORAGE_KEY,
    );
    expect(persistedBeforeRestart).not.toBeNull();

    let restartedStore!: typeof import('../alpha-preferences-store');
    jest.isolateModules(() => {
      restartedStore = jest.requireActual('../alpha-preferences-store');
    });

    expect(restartedStore.getAlphaPreferences()).toEqual({
      schemaVersion: 2,
      onboardingCompleted: true,
      profile: 'id-latin-6dot-alpha-v0',
      speechMode: 'press',
      speechRate: 1.7,
      speechVolume: 0.4,
      muted: true,
      hapticsEnabled: false,
      debugMenuEnabled: true,
    });
    expect(JSON.parse(persistedBeforeRestart ?? '{}')).not.toHaveProperty(
      'scanText',
    );
  });
});
