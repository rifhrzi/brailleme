describe('persistensi AppPreferences', () => {
  it('memuat kembali preferensi yang sama setelah module restart', () => {
    let firstStore!: typeof import('../app-preferences-store');
    jest.isolateModules(() => {
      firstStore = jest.requireActual('../app-preferences-store');
    });

    firstStore.updateAppPreferences({
      onboardingCompleted: true,
      speechRate: 1.7,
      speechVolume: 0.4,
      muted: true,
      hapticsEnabled: false,
    });

    const persistedBeforeRestart = globalThis.localStorage.getItem(
      firstStore.APP_PREFERENCES_STORAGE_KEY,
    );
    expect(persistedBeforeRestart).not.toBeNull();

    let restartedStore!: typeof import('../app-preferences-store');
    jest.isolateModules(() => {
      restartedStore = jest.requireActual('../app-preferences-store');
    });

    expect(restartedStore.getAppPreferences()).toEqual({
      schemaVersion: 4,
      onboardingCompleted: true,
      profile: 'id-latin-6dot-v1',
      speechMode: 'press',
      speechRate: 1.7,
      speechVolume: 0.4,
      muted: true,
      hapticsEnabled: false,
    });
    expect(JSON.parse(persistedBeforeRestart ?? '{}')).not.toHaveProperty(
      'scanText',
    );
  });

  it.each([1, 2, 3])(
    'memigrasikan schema %s dari key lama dan menghapus field debug',
    (schemaVersion) => {
      globalThis.localStorage.setItem(
        'brailleme.alpha-preferences.v1',
        JSON.stringify({
          schemaVersion,
          onboardingCompleted: true,
          profile: 'id-latin-6dot-alpha-v0',
          speechMode: 'press',
          speechRate: 1.4,
          speechVolume: 0.6,
          muted: false,
          hapticsEnabled: true,
          debugMenuEnabled: true,
        }),
      );

      let migratedStore!: typeof import('../app-preferences-store');
      jest.isolateModules(() => {
        migratedStore = jest.requireActual('../app-preferences-store');
      });

      expect(migratedStore.getAppPreferences()).toEqual({
        schemaVersion: 4,
        onboardingCompleted: true,
        profile: 'id-latin-6dot-v1',
        speechMode: 'press',
        speechRate: 1.4,
        speechVolume: 0.6,
        muted: false,
        hapticsEnabled: true,
      });
      expect(
        JSON.parse(
          globalThis.localStorage.getItem(
            migratedStore.APP_PREFERENCES_STORAGE_KEY,
          ) ?? '{}',
        ),
      ).not.toHaveProperty('debugMenuEnabled');
      expect(
        globalThis.localStorage.getItem('brailleme.alpha-preferences.v1'),
      ).toBeNull();
    },
  );

  it('memprioritaskan key baru dan membersihkan key lama tanpa menimpa data', () => {
    globalThis.localStorage.setItem(
      'brailleme.alpha-preferences.v1',
      JSON.stringify({
        schemaVersion: 3,
        onboardingCompleted: true,
        speechRate: 0.6,
        speechVolume: 0.9,
        muted: false,
        hapticsEnabled: true,
      }),
    );
    globalThis.localStorage.setItem(
      'brailleme.preferences.v1',
      JSON.stringify({
        schemaVersion: 4,
        onboardingCompleted: false,
        profile: 'id-latin-6dot-v1',
        speechMode: 'press',
        speechRate: 1.8,
        speechVolume: 0.3,
        muted: true,
        hapticsEnabled: false,
        debugMenuEnabled: true,
      }),
    );

    let migratedStore!: typeof import('../app-preferences-store');
    jest.isolateModules(() => {
      migratedStore = jest.requireActual('../app-preferences-store');
    });

    expect(migratedStore.getAppPreferences()).toEqual({
      schemaVersion: 4,
      onboardingCompleted: false,
      profile: 'id-latin-6dot-v1',
      speechMode: 'press',
      speechRate: 1.8,
      speechVolume: 0.3,
      muted: true,
      hapticsEnabled: false,
    });
    expect(
      globalThis.localStorage.getItem('brailleme.alpha-preferences.v1'),
    ).toBeNull();
    expect(
      JSON.parse(
        globalThis.localStorage.getItem(
          migratedStore.APP_PREFERENCES_STORAGE_KEY,
        ) ?? '{}',
      ),
    ).not.toHaveProperty('debugMenuEnabled');
  });

  it('memulihkan legacy saat JSON key baru rusak', () => {
    globalThis.localStorage.setItem(
      'brailleme.alpha-preferences.v1',
      JSON.stringify({
        schemaVersion: 3,
        onboardingCompleted: true,
        speechRate: 1.6,
        speechVolume: 0.4,
        muted: true,
        hapticsEnabled: false,
        debugMenuEnabled: true,
      }),
    );
    globalThis.localStorage.setItem(
      'brailleme.preferences.v1',
      '{json-rusak',
    );

    let migratedStore!: typeof import('../app-preferences-store');
    jest.isolateModules(() => {
      migratedStore = jest.requireActual('../app-preferences-store');
    });

    expect(migratedStore.getAppPreferences()).toEqual({
      schemaVersion: 4,
      onboardingCompleted: true,
      profile: 'id-latin-6dot-v1',
      speechMode: 'press',
      speechRate: 1.6,
      speechVolume: 0.4,
      muted: true,
      hapticsEnabled: false,
    });
    expect(
      globalThis.localStorage.getItem('brailleme.alpha-preferences.v1'),
    ).toBeNull();
  });

  it('tidak fallback ke legacy untuk schema masa depan yang valid', () => {
    globalThis.localStorage.setItem(
      'brailleme.alpha-preferences.v1',
      JSON.stringify({
        schemaVersion: 3,
        onboardingCompleted: true,
        speechRate: 1.6,
        speechVolume: 0.4,
        muted: true,
        hapticsEnabled: false,
      }),
    );
    globalThis.localStorage.setItem(
      'brailleme.preferences.v1',
      JSON.stringify({
        schemaVersion: 99,
        onboardingCompleted: true,
      }),
    );

    let migratedStore!: typeof import('../app-preferences-store');
    jest.isolateModules(() => {
      migratedStore = jest.requireActual('../app-preferences-store');
    });

    expect(migratedStore.getAppPreferences()).toEqual({
      schemaVersion: 4,
      onboardingCompleted: false,
      profile: 'id-latin-6dot-v1',
      speechMode: 'press',
      speechRate: 1,
      speechVolume: 1,
      muted: false,
      hapticsEnabled: true,
    });
    expect(
      globalThis.localStorage.getItem('brailleme.alpha-preferences.v1'),
    ).toBeNull();
  });
});
