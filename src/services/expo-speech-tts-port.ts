import {
  ActivityAction,
  startActivityAsync,
} from 'expo-intent-launcher';
import type * as Speech from 'expo-speech';

import BrailleLocalVoiceModule from '../../modules/braille-local-voice/src/BrailleLocalVoiceModule';
import type { BrailleLocalVoice } from '../../modules/braille-local-voice/src/BrailleLocalVoice.types';
import type {
  TtsPlaybackEvent,
  TtsPlaybackListener,
  TtsPort,
} from '@/services/ports';
import type { PortResult, TtsAvailability, TtsSpeakOptions } from '@/types';

const RATE_MIN = 0.5;
const RATE_MAX = 2;
const VOLUME_MIN = 0;
const VOLUME_MAX = 1;
const ANDROID_SPEECH_FALLBACK_LIMIT = 4_000;
export const TTS_VOICE_PROBE_TIMEOUT_MS = 7_000;
export const TTS_PLAYBACK_WATCHDOG_TIMEOUT_MS = 15_000;
export const TTS_STOP_TIMEOUT_MS = 3_000;

const LOCAL_VOICE_READY: TtsAvailability = Object.freeze({
  status: 'available',
  localVoiceAvailable: true,
  message: 'Suara Bahasa Indonesia tersedia secara lokal.',
});

const LOCAL_VOICE_MISSING: TtsAvailability = Object.freeze({
  status: 'unavailable',
  localVoiceAvailable: false,
  message: 'Suara Bahasa Indonesia offline belum terpasang.',
});

const LOCAL_VOICE_PROBE_FAILED: TtsAvailability = Object.freeze({
  status: 'unavailable',
  localVoiceAvailable: false,
  message: 'Status suara Bahasa Indonesia lokal tidak dapat diperiksa.',
});

type SpeechVoice = {
  readonly identifier: string;
  readonly language: string;
};

type PreferredVoice = SpeechVoice;

type ActivePlayback = {
  readonly requestKey: string;
  readonly text: string;
  readonly token: number;
};

type SpeechRequest = {
  readonly options: TtsSpeakOptions;
  readonly text: string;
};

export type ExpoSpeechApi = Pick<
  typeof Speech,
  | 'getAvailableVoicesAsync'
  | 'maxSpeechInputLength'
  | 'speak'
  | 'stop'
>;

export type ExpoSpeechLoader = () => Promise<ExpoSpeechApi>;

export type LocalVoiceProbe = {
  getLocalVoicesAsync(): Promise<readonly BrailleLocalVoice[]>;
};

type TtsSettingsLauncher = () => Promise<unknown>;

const loadExpoSpeech: ExpoSpeechLoader = async () => import('expo-speech');

function failed(message: string): PortResult {
  return { ok: false, code: 'failed', message };
}

function unavailable(message: string): PortResult {
  return { ok: false, code: 'unavailable', message };
}

function normalizeLanguage(language: string): string {
  return language.trim().replaceAll('_', '-').toLowerCase();
}

function isIndonesian(language: string): boolean {
  const normalized = normalizeLanguage(language);
  return normalized === 'id' || normalized.startsWith('id-');
}

function isSpeechVoice(value: unknown): value is SpeechVoice {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const voice = value as Record<string, unknown>;
  return (
    typeof voice.identifier === 'string' &&
    voice.identifier.trim().length > 0 &&
    typeof voice.language === 'string' &&
    voice.language.trim().length > 0
  );
}

function isLocalVoice(value: unknown): value is BrailleLocalVoice {
  return isSpeechVoice(value);
}

function voiceRank(voice: SpeechVoice): number {
  const language = normalizeLanguage(voice.language);
  if (language === 'id-id') {
    return 2;
  }
  return isIndonesian(language) ? 1 : 0;
}

function requestKey(text: string, options: TtsSpeakOptions): string {
  return `${text}\u0000${options.rate}\u0000${options.volume}`;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Operasi Text-to-Speech melewati batas waktu.'));
    }, timeoutMs);

    void promise.then(
      (value) => {
        clearTimeout(timeout);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timeout);
        reject(error);
      },
    );
  });
}

export class ExpoSpeechTtsPort implements TtsPort {
  private readonly listeners = new Set<TtsPlaybackListener>();
  private commandQueue: Promise<void> = Promise.resolve();
  private playbackSequence = 0;
  private activePlayback: ActivePlayback | undefined;
  private lastRequest: SpeechRequest | undefined;
  private playbackWatchdog: ReturnType<typeof setTimeout> | undefined;
  private speechPromise: Promise<ExpoSpeechApi> | undefined;
  private preferredVoiceGeneration = 0;
  private preferredVoicePromise: Promise<PreferredVoice | null> | undefined;

  constructor(
    private readonly loadSpeech: ExpoSpeechLoader = loadExpoSpeech,
    private readonly localVoiceProbe: LocalVoiceProbe | null =
      BrailleLocalVoiceModule,
    private readonly launchTtsSettings: TtsSettingsLauncher = () =>
      startActivityAsync(ActivityAction.TTS_SETTINGS),
  ) {}

  subscribePlayback(listener: TtsPlaybackListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  async getAvailability(): Promise<TtsAvailability> {
    if (!this.localVoiceProbe) {
      return LOCAL_VOICE_PROBE_FAILED;
    }

    try {
      const speech = await this.getSpeech();
      return (await this.getCurrentPreferredVoice(speech))
        ? LOCAL_VOICE_READY
        : LOCAL_VOICE_MISSING;
    } catch {
      return LOCAL_VOICE_PROBE_FAILED;
    }
  }

  speak(text: string, options: TtsSpeakOptions): Promise<PortResult> {
    return this.enqueue(() => this.speakOnce(text, options, false));
  }

  stop(): Promise<PortResult> {
    return this.enqueue(() => this.stopOnce());
  }

  repeat(): Promise<PortResult> {
    return this.enqueue(async () => {
      const previous = this.lastRequest;
      if (!previous) {
        return failed('Belum ada hasil suara yang dapat diulangi.');
      }
      return this.speakOnce(previous.text, previous.options, true);
    });
  }

  invalidateAvailabilityCache(): void {
    this.preferredVoiceGeneration += 1;
    this.preferredVoicePromise = undefined;
  }

  async openTtsSettings(): Promise<PortResult> {
    // Daftar voice dapat berubah setelah pengguna kembali dari pengaturan.
    this.invalidateAvailabilityCache();
    try {
      await this.launchTtsSettings();
      return { ok: true };
    } catch {
      return failed(
        'Pengaturan Text-to-Speech tidak dapat dibuka pada perangkat ini.',
      );
    }
  }

  private enqueue<T>(command: () => Promise<T>): Promise<T> {
    const next = this.commandQueue.then(command, command);
    this.commandQueue = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private async speakOnce(
    rawText: string,
    options: TtsSpeakOptions,
    force: boolean,
  ): Promise<PortResult> {
    const text = rawText.trim();
    const validationError = this.validateRequest(text, options);
    if (validationError) {
      return failed(validationError);
    }

    const key = requestKey(text, options);
    if (!force && this.activePlayback?.requestKey === key) {
      return { ok: true };
    }

    let speech: ExpoSpeechApi;
    let voice: PreferredVoice | null;
    try {
      speech = await this.getSpeech();
      voice = await this.getCurrentPreferredVoice(speech);
    } catch {
      return unavailable('Status suara Bahasa Indonesia lokal tidak dapat diperiksa.');
    }
    if (!voice) {
      return unavailable('Suara Bahasa Indonesia offline belum tersedia.');
    }

    if (this.activePlayback) {
      const stopped = await this.stopOnce();
      if (!stopped.ok) {
        return stopped;
      }
    }

    const token = ++this.playbackSequence;
    this.clearPlaybackWatchdog();
    this.activePlayback = { requestKey: key, text, token };

    try {
      speech.speak(text, {
        language: 'id',
        rate: options.rate,
        volume: options.volume,
        voice: voice.identifier,
        onDone: () => this.finishPlayback(token, 'done'),
        onError: () => this.failPlayback(token),
        onStopped: () => this.finishPlayback(token, 'stopped'),
      });
      this.lastRequest = { text, options: { ...options } };
      if (this.activePlayback?.token === token) {
        this.startPlaybackWatchdog(speech, token);
        this.emit({ status: 'speaking', text });
      }
      return { ok: true };
    } catch {
      if (this.activePlayback?.token === token) {
        this.activePlayback = undefined;
      }
      this.clearPlaybackWatchdog();
      this.emit({
        status: 'error',
        message: 'Suara gagal dibacakan.',
        text,
      });
      return failed('Suara gagal dibacakan.');
    }
  }

  private async stopOnce(): Promise<PortResult> {
    const active = this.activePlayback;
    this.clearPlaybackWatchdog();
    try {
      const speech = await this.getSpeech();
      await withTimeout(speech.stop(), TTS_STOP_TIMEOUT_MS);
    } catch {
      if (active && this.activePlayback?.token === active.token) {
        this.playbackSequence += 1;
        this.activePlayback = undefined;
        this.emit({
          status: 'error',
          message: 'Suara tidak dapat dihentikan.',
          text: active.text,
        });
      }
      return failed('Suara tidak dapat dihentikan.');
    }

    const callbackHasNotFinishedPlayback =
      active && this.activePlayback?.token === active.token;
    this.playbackSequence += 1;
    if (callbackHasNotFinishedPlayback) {
      this.activePlayback = undefined;
      this.emit({ status: 'idle', reason: 'stopped', text: active.text });
    }
    return { ok: true };
  }

  private finishPlayback(token: number, reason: 'done' | 'stopped'): void {
    const active = this.activePlayback;
    if (!active || active.token !== token) {
      return;
    }
    this.clearPlaybackWatchdog();
    this.activePlayback = undefined;
    this.emit({ status: 'idle', reason, text: active.text });
  }

  private failPlayback(token: number): void {
    const active = this.activePlayback;
    if (!active || active.token !== token) {
      return;
    }
    this.clearPlaybackWatchdog();
    this.activePlayback = undefined;
    this.emit({
      status: 'error',
      message: 'Suara gagal dibacakan.',
      text: active.text,
    });
  }

  private validateRequest(text: string, options: TtsSpeakOptions): string | null {
    if (!text) {
      return 'Teks yang akan dibacakan tidak boleh kosong.';
    }

    if (text.length > ANDROID_SPEECH_FALLBACK_LIMIT) {
      return 'Teks terlalu panjang untuk Text-to-Speech Android.';
    }
    if (
      !Number.isFinite(options.rate) ||
      options.rate < RATE_MIN ||
      options.rate > RATE_MAX
    ) {
      return 'Kecepatan suara tidak valid.';
    }
    if (
      !Number.isFinite(options.volume) ||
      options.volume < VOLUME_MIN ||
      options.volume > VOLUME_MAX
    ) {
      return 'Volume suara tidak valid.';
    }
    return null;
  }

  private async findPreferredVoice(
    speech: ExpoSpeechApi,
  ): Promise<PreferredVoice | null> {
    if (!this.localVoiceProbe) {
      throw new Error('Probe suara lokal tidak tersedia.');
    }

    const [nativeVoices, speechVoices] = await withTimeout(
      Promise.all([
        this.localVoiceProbe.getLocalVoicesAsync(),
        speech.getAvailableVoicesAsync(),
      ]),
      TTS_VOICE_PROBE_TIMEOUT_MS,
    );
    if (!Array.isArray(nativeVoices) || !Array.isArray(speechVoices)) {
      throw new Error('Daftar suara tidak valid.');
    }

    const localIndonesianIdentifiers = new Set(
      nativeVoices
        .filter(isLocalVoice)
        .filter((voice) => isIndonesian(voice.language))
        .map((voice) => voice.identifier),
    );

    return speechVoices
      .filter(isSpeechVoice)
      .filter(
        (voice) =>
          isIndonesian(voice.language) &&
          localIndonesianIdentifiers.has(voice.identifier),
      )
      .sort((left, right) => voiceRank(right) - voiceRank(left))[0] ?? null;
  }

  private getPreferredVoice(
    speech: ExpoSpeechApi,
  ): Promise<PreferredVoice | null> {
    if (!this.preferredVoicePromise) {
      const pending = this.findPreferredVoice(speech);
      this.preferredVoicePromise = pending;
      void pending.catch(() => {
        if (this.preferredVoicePromise === pending) {
          this.preferredVoicePromise = undefined;
        }
      });
    }
    return this.preferredVoicePromise;
  }

  private async getCurrentPreferredVoice(
    speech: ExpoSpeechApi,
  ): Promise<PreferredVoice | null> {
    for (;;) {
      const generation = this.preferredVoiceGeneration;
      try {
        const voice = await this.getPreferredVoice(speech);
        if (generation === this.preferredVoiceGeneration) {
          return voice;
        }
      } catch (error) {
        if (generation === this.preferredVoiceGeneration) {
          throw error;
        }
      }
    }
  }

  private getSpeech(): Promise<ExpoSpeechApi> {
    if (!this.speechPromise) {
      const pending = this.loadSpeech();
      this.speechPromise = pending;
      void pending.catch(() => {
        if (this.speechPromise === pending) {
          this.speechPromise = undefined;
        }
      });
    }
    return this.speechPromise;
  }

  private startPlaybackWatchdog(
    speech: ExpoSpeechApi,
    token: number,
  ): void {
    if (this.activePlayback?.token !== token) {
      return;
    }

    this.clearPlaybackWatchdog();
    this.playbackWatchdog = setTimeout(() => {
      const active = this.activePlayback;
      this.playbackWatchdog = undefined;
      if (!active || active.token !== token) {
        return;
      }

      this.playbackSequence += 1;
      this.activePlayback = undefined;
      this.emit({
        status: 'error',
        message: 'Suara gagal dibacakan.',
        text: active.text,
      });
      void speech.stop().catch(() => undefined);
    }, TTS_PLAYBACK_WATCHDOG_TIMEOUT_MS);
  }

  private clearPlaybackWatchdog(): void {
    if (this.playbackWatchdog !== undefined) {
      clearTimeout(this.playbackWatchdog);
      this.playbackWatchdog = undefined;
    }
  }

  private emit(event: TtsPlaybackEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch {
        // Satu consumer tidak boleh memutus lifecycle suara consumer lain.
      }
    }
  }
}

export const ttsPort: TtsPort = new ExpoSpeechTtsPort();
