import type { SpeechOptions } from 'expo-speech';

import {
  ExpoSpeechTtsPort,
  TTS_PLAYBACK_WATCHDOG_TIMEOUT_MS,
  TTS_STOP_TIMEOUT_MS,
  TTS_VOICE_PROBE_TIMEOUT_MS,
  type ExpoSpeechApi,
  type LocalVoiceProbe,
} from '@/services/expo-speech-tts-port';
import type { TtsPlaybackEvent } from '@/services/ports';

jest.mock(
  '../../../modules/braille-local-voice/src/BrailleLocalVoiceModule',
  () => ({ __esModule: true, default: null }),
);

const INDONESIAN_LOCAL_VOICE = {
  identifier: 'id-local',
  language: 'id-ID',
  name: 'Bahasa Indonesia lokal',
} as const;

function createSpeechHarness(
  voices: readonly { identifier: string; language: string }[] = [
    INDONESIAN_LOCAL_VOICE,
  ],
) {
  const speak = jest.fn();
  const stop = jest.fn(async () => undefined);
  const getAvailableVoicesAsync = jest.fn(async () => voices);
  const api = {
    getAvailableVoicesAsync,
    maxSpeechInputLength: 4_000,
    speak,
    stop,
  } as unknown as ExpoSpeechApi;

  return { api, getAvailableVoicesAsync, speak, stop };
}

function createLocalProbe(
  voices: readonly {
    identifier: string;
    language: string;
    name: string;
  }[] = [INDONESIAN_LOCAL_VOICE],
): LocalVoiceProbe {
  return {
    getLocalVoicesAsync: jest.fn(async () => voices),
  };
}

function speechOptionsFrom(
  speak: jest.Mock,
  callIndex = 0,
): SpeechOptions {
  const options = speak.mock.calls[callIndex]?.[1] as
    | SpeechOptions
    | undefined;
  if (!options) {
    throw new Error('Opsi Speech.speak tidak ditemukan.');
  }
  return options;
}

describe('ExpoSpeechTtsPort', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('hanya menyatakan siap ketika voice Indonesia cocok pada probe lokal dan Expo Speech', async () => {
    const speech = createSpeechHarness([
      { identifier: 'id-generic', language: 'id' },
      INDONESIAN_LOCAL_VOICE,
    ]);
    const probe = createLocalProbe([
      { ...INDONESIAN_LOCAL_VOICE, identifier: 'id-generic' },
      INDONESIAN_LOCAL_VOICE,
      { identifier: 'en-local', language: 'en-US', name: 'English local' },
    ]);
    const port = new ExpoSpeechTtsPort(async () => speech.api, probe);

    await expect(port.getAvailability()).resolves.toEqual({
      status: 'available',
      localVoiceAvailable: true,
      message: 'Suara Bahasa Indonesia tersedia secara lokal.',
    });

    await expect(port.speak(' A ', { rate: 1, volume: 0.8 })).resolves.toEqual({
      ok: true,
    });
    expect(speech.speak).toHaveBeenCalledWith(
      'A',
      expect.objectContaining({
        language: 'id',
        rate: 1,
        voice: 'id-local',
        volume: 0.8,
      }),
    );
    expect(probe.getLocalVoicesAsync).toHaveBeenCalledTimes(1);
    expect(speech.getAvailableVoicesAsync).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['modul native tidak tersedia', null],
    [
      'probe native gagal',
      {
        getLocalVoicesAsync: jest.fn(async () => {
          throw new Error('native probe failed');
        }),
      },
    ],
  ] as const)('gagal aman ketika %s', async (_caseName, probe) => {
    const port = new ExpoSpeechTtsPort(
      async () => createSpeechHarness().api,
      probe as LocalVoiceProbe | null,
    );

    await expect(port.getAvailability()).resolves.toEqual({
      status: 'unavailable',
      localVoiceAvailable: false,
      message: 'Status suara Bahasa Indonesia lokal tidak dapat diperiksa.',
    });
    await expect(port.speak('A', { rate: 1, volume: 1 })).resolves.toMatchObject({
      ok: false,
      code: 'unavailable',
    });
  });

  it('gagal aman ketika binary belum memuat native Expo Speech', async () => {
    const port = new ExpoSpeechTtsPort(
      async () => {
        throw new Error('Cannot find native module ExpoSpeech');
      },
      createLocalProbe(),
    );

    await expect(port.getAvailability()).resolves.toEqual({
      status: 'unavailable',
      localVoiceAvailable: false,
      message: 'Status suara Bahasa Indonesia lokal tidak dapat diperiksa.',
    });
    await expect(port.speak('A', { rate: 1, volume: 1 })).resolves.toEqual({
      ok: false,
      code: 'unavailable',
      message: 'Status suara Bahasa Indonesia lokal tidak dapat diperiksa.',
    });
  });

  it('mengakhiri pemeriksaan availability ketika probe voice tidak pernah selesai', async () => {
    const speech = createSpeechHarness();
    const probe: LocalVoiceProbe = {
      getLocalVoicesAsync: jest.fn(
        () => new Promise<never>(() => undefined),
      ),
    };
    const port = new ExpoSpeechTtsPort(async () => speech.api, probe);

    const availability = port.getAvailability();
    await Promise.resolve();
    await jest.advanceTimersByTimeAsync(TTS_VOICE_PROBE_TIMEOUT_MS);

    await expect(availability).resolves.toEqual({
      status: 'unavailable',
      localVoiceAvailable: false,
      message: 'Status suara Bahasa Indonesia lokal tidak dapat diperiksa.',
    });
    expect(jest.getTimerCount()).toBe(0);
  });

  it('mengakhiri speak ketika daftar voice Expo Speech tidak pernah selesai', async () => {
    const speech = createSpeechHarness();
    speech.getAvailableVoicesAsync.mockImplementation(
      () => new Promise(() => undefined),
    );
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );

    const result = port.speak('A', { rate: 1, volume: 1 });
    await Promise.resolve();
    await jest.advanceTimersByTimeAsync(TTS_VOICE_PROBE_TIMEOUT_MS);

    await expect(result).resolves.toEqual({
      ok: false,
      code: 'unavailable',
      message: 'Status suara Bahasa Indonesia lokal tidak dapat diperiksa.',
    });
    expect(speech.speak).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it.each([
    [
      'hanya voice lokal bahasa lain',
      [{ identifier: 'en-local', language: 'en-US', name: 'English local' }],
      [{ identifier: 'en-local', language: 'en-US' }],
    ],
    [
      'identifier lokal tidak ada pada Expo Speech',
      [INDONESIAN_LOCAL_VOICE],
      [{ identifier: 'id-other', language: 'id-ID' }],
    ],
  ] as const)(
    'tidak mengklaim voice Indonesia lokal ketika %s',
    async (_caseName, nativeVoices, speechVoices) => {
      const port = new ExpoSpeechTtsPort(
        async () => createSpeechHarness(speechVoices).api,
        createLocalProbe(nativeVoices),
      );

      await expect(port.getAvailability()).resolves.toEqual({
        status: 'unavailable',
        localVoiceAvailable: false,
        message: 'Suara Bahasa Indonesia offline belum terpasang.',
      });
    },
  );

  it.each([
    ['teks kosong', '   ', { rate: 1, volume: 1 }],
    ['teks terlalu panjang', 'A'.repeat(4_001), { rate: 1, volume: 1 }],
    ['rate NaN', 'A', { rate: Number.NaN, volume: 1 }],
    ['rate terlalu rendah', 'A', { rate: 0.4, volume: 1 }],
    ['rate terlalu tinggi', 'A', { rate: 2.1, volume: 1 }],
    ['volume NaN', 'A', { rate: 1, volume: Number.NaN }],
    ['volume negatif', 'A', { rate: 1, volume: -0.1 }],
    ['volume di atas satu', 'A', { rate: 1, volume: 1.1 }],
  ])(
    'menolak %s sebelum menyentuh native speech',
    async (_caseName, text, options) => {
      const speech = createSpeechHarness();
      const probe = createLocalProbe();
      const loadSpeech = jest.fn(async () => speech.api);
      const port = new ExpoSpeechTtsPort(loadSpeech, probe);

      await expect(port.speak(text, options)).resolves.toMatchObject({
        ok: false,
        code: 'failed',
      });
      expect(speech.speak).not.toHaveBeenCalled();
      expect(probe.getLocalVoicesAsync).not.toHaveBeenCalled();
      expect(loadSpeech).not.toHaveBeenCalled();
    },
  );

  it('mendeduplikasi dua permintaan identik yang masih aktif', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );

    await expect(
      Promise.all([
        port.speak('A', { rate: 1, volume: 1 }),
        port.speak('A', { rate: 1, volume: 1 }),
      ]),
    ).resolves.toEqual([{ ok: true }, { ok: true }]);
    expect(speech.speak).toHaveBeenCalledTimes(1);
  });

  it('menerbitkan lifecycle selesai sehingga state UI tidak stale', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('A', { rate: 1, volume: 1 });
    expect(listener).toHaveBeenLastCalledWith({ status: 'speaking', text: 'A' });

    const options = speechOptionsFrom(speech.speak);
    (options.onDone as (() => void) | undefined)?.();
    expect(listener).toHaveBeenLastCalledWith({
      status: 'idle',
      reason: 'done',
      text: 'A',
    });
    expect(jest.getTimerCount()).toBe(0);
  });

  it('menerbitkan error tanpa menyertakan isi hasil pada pesan', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('HASIL PRIVAT', { rate: 1, volume: 1 });
    const options = speechOptionsFrom(speech.speak);
    (options.onError as ((error: Error) => void) | undefined)?.(
      new Error('native details'),
    );

    expect(listener).toHaveBeenLastCalledWith({
      status: 'error',
      message: 'Suara gagal dibacakan.',
      text: 'HASIL PRIVAT',
    });
    expect(listener.mock.lastCall?.[0].status).toBe('error');
    expect(
      listener.mock.lastCall?.[0].status === 'error'
        ? listener.mock.lastCall[0].message
        : '',
    ).not.toContain('HASIL PRIVAT');
    expect(jest.getTimerCount()).toBe(0);
  });

  it('watchdog mengakhiri state speaking jika native tidak mengirim callback', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('A', { rate: 1, volume: 1 });
    expect(jest.getTimerCount()).toBe(1);

    await jest.advanceTimersByTimeAsync(TTS_PLAYBACK_WATCHDOG_TIMEOUT_MS - 1);
    expect(listener).toHaveBeenLastCalledWith({ status: 'speaking', text: 'A' });

    await jest.advanceTimersByTimeAsync(1);
    expect(listener).toHaveBeenLastCalledWith({
      status: 'error',
      message: 'Suara gagal dibacakan.',
      text: 'A',
    });
    expect(speech.stop).toHaveBeenCalledTimes(1);
    expect(jest.getTimerCount()).toBe(0);

    const options = speechOptionsFrom(speech.speak);
    const callsAfterTimeout = listener.mock.calls.length;
    (options.onDone as (() => void) | undefined)?.();
    expect(listener).toHaveBeenCalledTimes(callsAfterTimeout);
  });

  it('playback baru mengganti watchdog lama dan mengabaikan callback lama', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('A', { rate: 1, volume: 1 });
    const oldOptions = speechOptionsFrom(speech.speak);
    await port.speak('B', { rate: 1, volume: 1 });

    expect(speech.stop).toHaveBeenCalledTimes(1);
    expect(jest.getTimerCount()).toBe(1);
    const callsAfterReplacement = listener.mock.calls.length;
    (oldOptions.onDone as (() => void) | undefined)?.();
    expect(listener).toHaveBeenCalledTimes(callsAfterReplacement);

    await jest.advanceTimersByTimeAsync(TTS_PLAYBACK_WATCHDOG_TIMEOUT_MS);
    expect(listener).toHaveBeenLastCalledWith({
      status: 'error',
      message: 'Suara gagal dibacakan.',
      text: 'B',
    });
    expect(jest.getTimerCount()).toBe(0);
  });

  it('stop membatalkan callback lama dan repeat memulai utterance baru', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('A', { rate: 1, volume: 1 });
    const oldOptions = speechOptionsFrom(speech.speak);
    listener.mockClear();

    await expect(port.stop()).resolves.toEqual({ ok: true });
    expect(speech.stop).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenLastCalledWith({
      status: 'idle',
      reason: 'stopped',
      text: 'A',
    });
    expect(jest.getTimerCount()).toBe(0);
    const callsAfterStop = listener.mock.calls.length;
    (oldOptions.onDone as (() => void) | undefined)?.();
    expect(listener).toHaveBeenCalledTimes(callsAfterStop);

    await expect(port.repeat()).resolves.toEqual({ ok: true });
    expect(speech.speak).toHaveBeenCalledTimes(2);
    expect(jest.getTimerCount()).toBe(1);
  });

  it('tidak menggandakan event stopped ketika native callback mendahului promise stop', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('A', { rate: 1, volume: 1 });
    const options = speechOptionsFrom(speech.speak);
    speech.stop.mockImplementationOnce(async () => {
      (options.onStopped as (() => void) | undefined)?.();
    });
    listener.mockClear();

    await expect(port.stop()).resolves.toEqual({ ok: true });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith({
      status: 'idle',
      reason: 'stopped',
      text: 'A',
    });
    expect(jest.getTimerCount()).toBe(0);
  });

  it('membersihkan watchdog dan lifecycle ketika native stop gagal', async () => {
    const speech = createSpeechHarness();
    speech.stop.mockRejectedValueOnce(new Error('native stop hung up'));
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('A', { rate: 1, volume: 1 });
    expect(jest.getTimerCount()).toBe(1);

    await expect(port.stop()).resolves.toEqual({
      ok: false,
      code: 'failed',
      message: 'Suara tidak dapat dihentikan.',
    });
    expect(listener).toHaveBeenLastCalledWith({
      status: 'error',
      message: 'Suara tidak dapat dihentikan.',
      text: 'A',
    });
    expect(jest.getTimerCount()).toBe(0);

    await jest.advanceTimersByTimeAsync(TTS_PLAYBACK_WATCHDOG_TIMEOUT_MS);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('membatasi native stop yang menggantung agar antrean suara tetap berjalan', async () => {
    const speech = createSpeechHarness();
    speech.stop.mockImplementationOnce(
      () => new Promise<undefined>(() => undefined),
    );
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );
    const listener = jest.fn<void, [TtsPlaybackEvent]>();
    port.subscribePlayback(listener);

    await port.speak('A', { rate: 1, volume: 1 });
    const stopResult = port.stop();
    const repeatResult = port.repeat();

    await jest.advanceTimersByTimeAsync(TTS_STOP_TIMEOUT_MS);

    await expect(stopResult).resolves.toEqual({
      ok: false,
      code: 'failed',
      message: 'Suara tidak dapat dihentikan.',
    });
    await expect(repeatResult).resolves.toEqual({ ok: true });
    expect(listener).toHaveBeenCalledWith({
      status: 'error',
      message: 'Suara tidak dapat dihentikan.',
      text: 'A',
    });
    expect(speech.speak).toHaveBeenCalledTimes(2);
  });

  it('repeat gagal aman sebelum ada utterance yang berhasil dijadwalkan', async () => {
    const speech = createSpeechHarness();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      createLocalProbe(),
    );

    await expect(port.repeat()).resolves.toEqual({
      ok: false,
      code: 'failed',
      message: 'Belum ada hasil suara yang dapat diulangi.',
    });
    expect(speech.speak).not.toHaveBeenCalled();
  });

  it('membuka pengaturan TTS dan memetakan kegagalan intent', async () => {
    const launcher = jest
      .fn<Promise<void>, []>()
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(new Error('settings missing'));
    const port = new ExpoSpeechTtsPort(
      async () => createSpeechHarness().api,
      createLocalProbe(),
      launcher,
    );

    await expect(port.openTtsSettings()).resolves.toEqual({ ok: true });
    await expect(port.openTtsSettings()).resolves.toMatchObject({
      ok: false,
      code: 'failed',
    });
  });

  it('menginvalidasi cache voice ketika pengaturan TTS dibuka', async () => {
    const speech = createSpeechHarness();
    const probe = createLocalProbe();
    const port = new ExpoSpeechTtsPort(
      async () => speech.api,
      probe,
      async () => undefined,
    );

    await port.getAvailability();
    await port.getAvailability();
    expect(probe.getLocalVoicesAsync).toHaveBeenCalledTimes(1);

    await port.openTtsSettings();
    await port.getAvailability();
    expect(probe.getLocalVoicesAsync).toHaveBeenCalledTimes(2);
  });

  it('mengabaikan hasil probe lama setelah cache voice diinvalidasi', async () => {
    const speech = createSpeechHarness([INDONESIAN_LOCAL_VOICE]);
    let resolveOldProbe!: (
      voices: readonly {
        identifier: string;
        language: string;
        name: string;
      }[],
    ) => void;
    const oldProbe = new Promise<
      readonly { identifier: string; language: string; name: string }[]
    >((resolve) => {
      resolveOldProbe = resolve;
    });
    const probe: LocalVoiceProbe = {
      getLocalVoicesAsync: jest
        .fn()
        .mockReturnValueOnce(oldProbe)
        .mockResolvedValueOnce([INDONESIAN_LOCAL_VOICE]),
    };
    const port = new ExpoSpeechTtsPort(async () => speech.api, probe);

    const staleRequest = port.getAvailability();
    await Promise.resolve();
    port.invalidateAvailabilityCache();
    const freshRequest = port.getAvailability();
    const concurrentFreshRequest = port.getAvailability();

    await expect(Promise.all([freshRequest, concurrentFreshRequest])).resolves.toEqual([
      {
        status: 'available',
        localVoiceAvailable: true,
        message: 'Suara Bahasa Indonesia tersedia secara lokal.',
      },
      {
        status: 'available',
        localVoiceAvailable: true,
        message: 'Suara Bahasa Indonesia tersedia secara lokal.',
      },
    ]);

    resolveOldProbe([
      { identifier: 'id-old', language: 'id-ID', name: 'Voice lama' },
    ]);
    await expect(staleRequest).resolves.toEqual({
      status: 'available',
      localVoiceAvailable: true,
      message: 'Suara Bahasa Indonesia tersedia secara lokal.',
    });
    expect(probe.getLocalVoicesAsync).toHaveBeenCalledTimes(2);
    expect(speech.getAvailableVoicesAsync).toHaveBeenCalledTimes(2);
  });
});
