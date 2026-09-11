import * as Haptics from 'expo-haptics';
import { useFocusEffect, useRouter } from 'expo-router';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { AppState, StyleSheet, Switch, Text, View } from 'react-native';

import {
  AccessibleButton,
  Screen,
  SectionCard,
  StatusAnnouncer,
} from '@/components';
import {
  INITIAL_WORD_DRAFT_STATE,
  MAX_WORD_LENGTH,
  getWordText,
  wordDraftReducer,
  type WordDraftAction,
  type WordDraftState,
} from '@/features/live/word-draft';
import { PROVISIONAL_MIN_CONFIDENCE } from '@/ml/prediction-stabilizer';
import {
  cameraPermissionPort,
  scannerPort,
  ttsPort,
  type CameraPermissionPort,
  type ScannerPort,
  type TtsPort,
} from '@/services';
import { useAppPreferences } from '@/storage';
import {
  colors,
  minimumTouchTarget,
  radii,
  spacing,
  typography,
} from '@/theme';
import { isValidLiveScanEvent } from '@/types';
import type {
  CameraPermissionSnapshot,
  LiveScanEvent,
  QualityReason,
  ScannerAvailability,
  TtsAvailability,
} from '@/types';

type LiveScreenViewProps = {
  scanner?: ScannerPort;
  cameraPermission?: CameraPermissionPort;
  tts?: TtsPort;
  wordDraft?: WordDraftState | undefined;
  onWordDraftChange?: ((nextDraft: WordDraftState) => void) | undefined;
  renderScannerPreview?: (props: ScannerPreviewProps) => ReactNode;
};

export type ScannerPreviewProps = {
  readonly active: boolean;
  readonly resetToken: number;
  readonly sessionId: string;
  readonly torchEnabled: boolean;
};

type StatusPresentation = {
  title: string;
  guidance?: string;
  tone: 'neutral' | 'warning' | 'success' | 'error';
};

const qualityPresentation: Record<QualityReason, StatusPresentation> = {
  blur: {
    title: 'Gambar buram',
    guidance: 'Tahan posisi sejenak.',
    tone: 'warning',
  },
  dark: {
    title: 'Cahaya kurang',
    guidance: 'Pindah ke tempat terang atau aktifkan senter.',
    tone: 'warning',
  },
  glare: {
    title: 'Pantulan terlalu kuat',
    guidance: 'Ubah sudut kamera atau cahaya.',
    tone: 'warning',
  },
  too_far: {
    title: 'Lebih dekat',
    guidance: 'Dekatkan kamera perlahan.',
    tone: 'warning',
  },
  tilted: {
    title: 'Kurangi kemiringan',
    guidance: 'Sejajarkan kamera.',
    tone: 'warning',
  },
  no_grid: {
    title: 'Belum menemukan Braille',
    guidance: 'Arahkan area target ke teks Braille.',
    tone: 'warning',
  },
  low_confidence: {
    title: 'Belum dapat dipastikan',
    guidance: 'Perbaiki posisi dan coba kembali.',
    tone: 'warning',
  },
};

const defaultPermission: CameraPermissionSnapshot = {
  status: 'unavailable',
  canAskAgain: false,
};

const defaultScannerAvailability: ScannerAvailability = {
  available: false,
  reason: 'native_scanner_not_installed',
  message: 'Pemindai belum diperiksa.',
};

const defaultTtsAvailability: TtsAvailability = {
  status: 'unavailable',
  localVoiceAvailable: false,
  message: 'Suara lokal belum diperiksa.',
};

const failedTtsAvailability: TtsAvailability = {
  status: 'unavailable',
  localVoiceAvailable: false,
  message: 'Status suara lokal tidak dapat diperiksa.',
};

function getValidatedStableText(event: LiveScanEvent | null) {
  if (event?.status !== 'stable') {
    return undefined;
  }

  const text = event.text?.trim();
  const confidence = event.confidence;

  if (
    !text ||
    !/^[A-Z]$/.test(text) ||
    confidence === undefined ||
    !Number.isFinite(confidence) ||
    confidence < PROVISIONAL_MIN_CONFIDENCE ||
    confidence > 1
  ) {
    return undefined;
  }

  return text;
}

function areLiveScanEventsEquivalentForUi(
  current: LiveScanEvent | null,
  next: LiveScanEvent,
): boolean {
  if (
    !current ||
    current.sessionId !== next.sessionId ||
    current.status !== next.status ||
    current.reason !== next.reason
  ) {
    return false;
  }

  if (next.status === 'stable') {
    return getValidatedStableText(current) === getValidatedStableText(next);
  }

  // Timestamp, confidence hasil sementara, dan metadata pemindai tidak ditampilkan oleh
  // layar. Mengabaikannya mencegah subtree kamera dirender ulang tiap frame.
  return true;
}

function presentEvent(
  event: LiveScanEvent | null,
  initializing: boolean,
  scanActive: boolean,
): StatusPresentation {
  if (!event) {
    if (initializing) {
      return {
        title: 'Menyiapkan pemindai',
        guidance: 'Memeriksa kamera, pemindai, dan suara lokal.',
        tone: 'neutral',
      };
    }

    if (scanActive) {
      return {
        title: 'Mencari satu sel Braille',
        guidance: 'Arahkan satu huruf Braille ke kotak target.',
        tone: 'neutral',
      };
    }

    return {
      title: 'Pemindai belum tersedia',
      guidance: 'Coba lagi atau periksa pengaturan aplikasi.',
      tone: 'error',
    };
  }

  switch (event.status) {
    case 'searching':
      return {
        title: 'Belum menemukan satu sel',
        guidance: 'Arahkan satu huruf Braille ke kotak target.',
        tone: 'neutral',
      };
    case 'quality_issue':
      return event.reason
        ? qualityPresentation[event.reason]
        : {
            title: 'Perbaiki kualitas',
            guidance: 'Sesuaikan posisi, cahaya, atau sudut kamera.',
            tone: 'warning',
          };
    case 'candidate':
      return { title: 'Memastikan huruf', tone: 'neutral' };
    case 'stable':
      return getValidatedStableText(event)
        ? {
            title: 'Huruf terdeteksi',
            guidance: 'Cocokkan huruf dengan sumber Braille sebelum digunakan.',
            tone: 'success',
          }
        : {
            title: 'Belum dapat dipastikan',
            guidance: 'Huruf belum dapat ditampilkan. Pindai kembali.',
            tone: 'warning',
          };
    case 'unsupported':
      return {
        title: 'Jenis Braille belum didukung',
        guidance: 'Lihat Bantuan untuk mengetahui jenis Braille yang didukung.',
        tone: 'warning',
      };
    case 'error':
      return {
        title: 'Pemindaian mengalami masalah',
        guidance: 'Coba lagi atau kembali ke Beranda.',
        tone: 'error',
      };
  }
}

export function LiveScreenView({
  cameraPermission = cameraPermissionPort,
  onWordDraftChange,
  renderScannerPreview,
  scanner = scannerPort,
  tts = ttsPort,
  wordDraft: controlledWordDraft,
}: LiveScreenViewProps) {
  const router = useRouter();
  const preferences = useAppPreferences();
  const startingWordDraft =
    controlledWordDraft ?? INITIAL_WORD_DRAFT_STATE;
  const startingWordText = getWordText(startingWordDraft);
  const startingWordSpeechKey = startingWordText
    ? `${startingWordDraft.revision}:${startingWordText}`
    : '';
  const startedRef = useRef(false);
  const startPendingRef = useRef(false);
  const acceptsEventsRef = useRef(false);
  const mountedRef = useRef(true);
  const startAttemptRef = useRef(0);
  const permissionRequestSequenceRef = useRef(0);
  const ttsRequestSequenceRef = useRef(0);
  const speechActionSequenceRef = useRef(0);
  const activeSessionIdRef = useRef('');
  const scannerResetTokenRef = useRef(0);
  const retryInFlightRef = useRef(false);
  const finishInFlightRef = useRef(false);
  const pauseInFlightRef = useRef(false);
  const torchInFlightRef = useRef(false);
  const scannerCommandQueueRef = useRef<Promise<void>>(Promise.resolve());
  const activeSpeechRef = useRef(false);
  const hapticKeysRef = useRef(new Set<string>());
  const currentWordTextRef = useRef(startingWordText);
  const currentWordSpeechKeyRef = useRef(startingWordSpeechKey);
  const previousWordSpeechKeyRef = useRef(startingWordSpeechKey);
  const lastStableTextRef = useRef('');
  const stableResultSequenceRef = useRef(0);
  const eventRef = useRef<LiveScanEvent | null>(null);
  const invalidEventDetectedRef = useRef(false);
  const wordDraftRef = useRef<WordDraftState>(startingWordDraft);
  const [permission, setPermission] =
    useState<CameraPermissionSnapshot>(defaultPermission);
  const [scannerAvailability, setScannerAvailability] =
    useState<ScannerAvailability>(defaultScannerAvailability);
  const [ttsAvailability, setTtsAvailability] =
    useState<TtsAvailability>(defaultTtsAvailability);
  const [event, setEvent] = useState<LiveScanEvent | null>(null);
  const [invalidEventDetected, setInvalidEventDetected] = useState(false);
  const [paused, setPaused] = useState(false);
  const [scanActive, setScanActive] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState('');
  const [scannerResetToken, setScannerResetToken] = useState(0);
  const [stableResultSequence, setStableResultSequence] = useState(0);
  const [localWordDraft, setLocalWordDraft] = useState<WordDraftState>(
    startingWordDraft,
  );
  const wordDraft = controlledWordDraft ?? localWordDraft;
  const [torchEnabled, setTorchEnabled] = useState(false);
  const [torchAvailable, setTorchAvailable] = useState(
    scanner.getCapabilities?.().torchAvailable ?? true,
  );
  const [speaking, setSpeaking] = useState(false);
  const [hasSpoken, setHasSpoken] = useState(false);
  const [actionMessage, setActionMessage] = useState('');
  const [initializing, setInitializing] = useState(true);
  const [retrying, setRetrying] = useState(false);

  const runScannerCommand = useCallback(
    function runScannerCommand<T>(command: () => Promise<T>): Promise<T> {
      const nextCommand = scannerCommandQueueRef.current.then(
        () => command(),
        () => command(),
      );
      scannerCommandQueueRef.current = nextCommand.then(
        () => undefined,
        () => undefined,
      );
      return nextCommand;
    },
    [],
  );

  const applyWordAction = useCallback(
    (action: WordDraftAction) => {
      const nextDraft = wordDraftReducer(wordDraftRef.current, action);
      if (nextDraft === wordDraftRef.current) {
        return null;
      }
      wordDraftRef.current = nextDraft;
      const nextWordText = getWordText(nextDraft);
      currentWordTextRef.current = nextWordText;
      currentWordSpeechKeyRef.current = nextWordText
        ? `${nextDraft.revision}:${nextWordText}`
        : '';
      setLocalWordDraft(nextDraft);
      onWordDraftChange?.(nextDraft);
      return nextDraft;
    },
    [onWordDraftChange],
  );

  const invalidateCandidate = useCallback((advanceEpoch = true) => {
    lastStableTextRef.current = '';
    eventRef.current = null;
    invalidEventDetectedRef.current = false;
    setEvent(null);
    setInvalidEventDetected(false);
    if (advanceEpoch) {
      const nextEpoch = scannerResetTokenRef.current + 1;
      scannerResetTokenRef.current = nextEpoch;
      setScannerResetToken(nextEpoch);
    }
  }, []);

  const leaveLive = useCallback(() => {
    if (router.canGoBack()) {
      router.back();
      return;
    }
    router.replace('/');
  }, [router]);

  useEffect(() => {
    if (!scanner.subscribeCapabilities) {
      return;
    }
    return scanner.subscribeCapabilities((capabilities) => {
      if (!mountedRef.current) {
        return;
      }
      setTorchAvailable(capabilities.torchAvailable);
      if (!capabilities.torchAvailable) {
        setTorchEnabled(false);
      }
    });
  }, [scanner]);

  useEffect(() => {
    let active = true;
    const subscription = AppState.addEventListener('change', (nextState) => {
      const permissionRequest = ++permissionRequestSequenceRef.current;
      const ttsRequest = ++ttsRequestSequenceRef.current;
      if (nextState !== 'active') {
        invalidateCandidate();
        if (activeSpeechRef.current) {
          activeSpeechRef.current = false;
          setSpeaking(false);
          void tts.stop().catch(() => undefined);
        }
        return;
      }

      void cameraPermission
        .getStatus()
        .then((nextPermission) => {
          if (
            active &&
            permissionRequest === permissionRequestSequenceRef.current
          ) {
            setPermission(nextPermission);
          }
        })
        .catch(() => undefined);
      tts.invalidateAvailabilityCache?.();
      void tts
        .getAvailability()
        .then((nextTts) => {
          if (active && ttsRequest === ttsRequestSequenceRef.current) {
            setTtsAvailability(nextTts);
          }
        })
        .catch(() => {
          if (active && ttsRequest === ttsRequestSequenceRef.current) {
            setTtsAvailability(failedTtsAvailability);
          }
        });
    });

    return () => {
      active = false;
      permissionRequestSequenceRef.current += 1;
      ttsRequestSequenceRef.current += 1;
      subscription.remove();
    };
  }, [cameraPermission, invalidateCandidate, tts]);

  useEffect(() => {
    let active = true;
    mountedRef.current = true;
    acceptsEventsRef.current = false;
    const startAttempt = ++startAttemptRef.current;

    const unsubscribe = scanner.subscribe((nextEvent) => {
      if (!active || !acceptsEventsRef.current) {
        return;
      }

      // Session dan epoch diperiksa sebelum payload lain agar event lama tidak
      // dapat mengubah UI, termasuk bila payload lama itu sendiri rusak.
      if (nextEvent.sessionId !== activeSessionIdRef.current) {
        return;
      }
      if (nextEvent.scanEpoch !== scannerResetTokenRef.current) {
        return;
      }

      if (!isValidLiveScanEvent(nextEvent)) {
        eventRef.current = null;
        setEvent(null);
        if (!invalidEventDetectedRef.current) {
          invalidEventDetectedRef.current = true;
          setInvalidEventDetected(true);
        }
        setSpeaking(false);
        setHasSpoken(false);
        setActionMessage('Data pemindai diabaikan karena tidak valid.');
        if (activeSpeechRef.current) {
          activeSpeechRef.current = false;
          void tts.stop().catch(() => undefined);
        }
        return;
      }

      if (invalidEventDetectedRef.current) {
        invalidEventDetectedRef.current = false;
        setInvalidEventDetected(false);
      }
      const nextStableText = getValidatedStableText(nextEvent);
      if (
        nextStableText &&
        lastStableTextRef.current !== nextStableText
      ) {
        stableResultSequenceRef.current += 1;
        lastStableTextRef.current = nextStableText;
        setStableResultSequence(stableResultSequenceRef.current);
      }
      if (nextEvent.status === 'error') {
        acceptsEventsRef.current = false;
        setScanActive(false);
        setPaused(false);
        setTorchEnabled(false);
        if (activeSpeechRef.current) {
          activeSpeechRef.current = false;
          setSpeaking(false);
          void tts.stop().catch(() => undefined);
        }
      }
      if (!areLiveScanEventsEquivalentForUi(eventRef.current, nextEvent)) {
        eventRef.current = nextEvent;
        setEvent(nextEvent);
      }
    });

    const initialize = async () => {
      setInitializing(true);

      const ttsRequest = ++ttsRequestSequenceRef.current;
      void tts
        .getAvailability()
        .then((nextTts) => {
          if (active && ttsRequest === ttsRequestSequenceRef.current) {
            setTtsAvailability(nextTts);
          }
        })
        .catch(() => {
          if (active && ttsRequest === ttsRequestSequenceRef.current) {
            setTtsAvailability(failedTtsAvailability);
          }
        });

      try {
        const permissionRequest = ++permissionRequestSequenceRef.current;
        const [nextPermission, nextScanner] = await Promise.all([
          cameraPermission.getStatus(),
          scanner.getAvailability(),
        ]);
        if (
          !active ||
          startAttemptRef.current !== startAttempt ||
          permissionRequest !== permissionRequestSequenceRef.current
        ) {
          return;
        }

        setPermission(nextPermission);
        setScannerAvailability(nextScanner);

        if (nextPermission.status === 'granted' && nextScanner.available) {
          startPendingRef.current = true;
          const result = await runScannerCommand(() => scanner.start());
          startPendingRef.current = false;

          if (!active || startAttemptRef.current !== startAttempt) {
            return;
          }

          const sessionId = result.ok ? result.sessionId.trim() : '';
          const started = result.ok && sessionId.length > 0;
          startedRef.current = started;
          activeSessionIdRef.current = sessionId;
          setActiveSessionId(sessionId);
          acceptsEventsRef.current = started;
          setScanActive(started);
          if (started) {
            lastStableTextRef.current = '';
            stableResultSequenceRef.current = 0;
            setStableResultSequence(0);
            hapticKeysRef.current.clear();
            eventRef.current = null;
            setEvent(null);
            invalidEventDetectedRef.current = false;
            setInvalidEventDetected(false);
            setPaused(false);
            setTorchEnabled(false);
            setSpeaking(false);
            setHasSpoken(false);
          } else {
            setActionMessage('Pemindaian tidak dapat dimulai.');
          }
        }
      } catch {
        if (active && startAttemptRef.current === startAttempt) {
          startedRef.current = false;
          startPendingRef.current = false;
          acceptsEventsRef.current = false;
          setScanActive(false);
          setActionMessage('Kesiapan pemindaian tidak dapat diperiksa.');
        }
      } finally {
        if (active && startAttemptRef.current === startAttempt) {
          setInitializing(false);
        }
      }
    };

    void initialize();

    return () => {
      active = false;
      mountedRef.current = false;
      startAttemptRef.current += 1;
      permissionRequestSequenceRef.current += 1;
      ttsRequestSequenceRef.current += 1;
      activeSessionIdRef.current = '';
      acceptsEventsRef.current = false;
      unsubscribe();
      if (startedRef.current || startPendingRef.current) {
        startedRef.current = false;
        startPendingRef.current = false;
        void runScannerCommand(() => scanner.stop()).catch(() => undefined);
      }
      if (activeSpeechRef.current) {
        activeSpeechRef.current = false;
        void tts.stop().catch(() => undefined);
      }
    };
  }, [cameraPermission, runScannerCommand, scanner, tts]);

  useFocusEffect(
    useCallback(() => {
      return () => {
        const shouldStopSpeech = activeSpeechRef.current;
        activeSpeechRef.current = false;
        if (mountedRef.current) {
          setSpeaking(false);
        }
        if (shouldStopSpeech) {
          void tts.stop().catch(() => undefined);
        }
      };
    }, [tts]),
  );

  const presentation = useMemo(() => {
    if (initializing) {
      return presentEvent(event, true, scanActive);
    }
    if (!scannerAvailability.available) {
      return {
        title: 'Pemindai satu sel belum tersedia',
        guidance:
          'Pemindai belum dapat digunakan. Coba buka kembali aplikasi.',
        tone: 'error' as const,
      };
    }
    if (permission.status !== 'granted') {
      return {
        title: 'Izin kamera diperlukan',
        guidance: 'Berikan izin kamera untuk memulai pemindaian.',
        tone: 'warning' as const,
      };
    }
    if (invalidEventDetected) {
      return {
        title: 'Belum dapat dipastikan',
        guidance: 'Huruf belum dapat ditampilkan. Pindai kembali.',
        tone: 'warning' as const,
      };
    }
    return presentEvent(event, false, scanActive);
  }, [
    event,
    initializing,
    invalidEventDetected,
    permission.status,
    scanActive,
    scannerAvailability,
  ]);
  const stableText = getValidatedStableText(event);
  const candidateEvidenceKey = stableText && event
    ? `${event.sessionId}:${event.scanEpoch}:${event.timestampMs}:${stableText}`
    : '';
  const stableKey = stableText
    ? `${event?.sessionId ?? 'unknown'}:${event?.scanEpoch ?? scannerResetToken}:${stableResultSequence}:${stableText}`
    : '';
  const wordText = getWordText(wordDraft);
  const wordSpeechKey = wordText
    ? `${wordDraft.revision}:${wordText}`
    : '';
  const wordIsFull = wordDraft.letters.length >= MAX_WORD_LENGTH;
  const canAddStableLetter =
    Boolean(stableText && candidateEvidenceKey) && !wordIsFull;
  const scannerReady = scannerAvailability.available;
  const canRequestCameraPermission =
    permission.status !== 'granted' &&
    (permission.status === 'undetermined' || permission.canAskAgain);
  const permissionRequiresSettings =
    permission.status === 'denied' || permission.status === 'blocked';
  const canRetryRecovery = scannerReady && !permissionRequiresSettings;
  const needsRecovery =
    !initializing &&
    (!scannerReady ||
      permission.status !== 'granted' ||
      !scanActive ||
      event?.status === 'error');
  const recoveryTitle =
    event?.status === 'error'
      ? 'Pemindaian bermasalah'
      : !scannerReady
        ? 'Pemindai belum tersedia'
        : permission.status !== 'granted'
          ? 'Kamera belum siap'
          : 'Pemindaian belum siap';
  const recoveryMessage =
    event?.status === 'error'
      ? 'Pemindaian mengalami masalah. Tidak ada teks yang dihasilkan.'
      : !scannerReady
        ? 'Pemindai satu sel belum dapat dijalankan pada perangkat ini.'
        : permission.status !== 'granted'
          ? permissionRequiresSettings
            ? 'Izin kamera ditolak. Berikan izin melalui pengaturan aplikasi Android.'
            : 'Izinkan akses kamera untuk mulai memindai.'
          : 'Pemindaian belum dapat dimulai.';
  const canSpeak =
    Boolean(wordText) &&
    ttsAvailability.status === 'available' &&
    ttsAvailability.localVoiceAvailable &&
    !preferences.muted &&
    preferences.speechVolume > 0;
  const voiceDisabledMessage = !wordText
    ? 'Bacakan tersedia setelah minimal satu huruf ditambahkan ke kata.'
    : preferences.muted
      ? 'Suara aplikasi sedang dibisukan.'
      : preferences.speechVolume <= 0
        ? 'Volume suara aplikasi diatur ke nol persen.'
        : ttsAvailability.status !== 'available' ||
            !ttsAvailability.localVoiceAvailable
          ? ttsAvailability.message.trim() ||
            'Suara Bahasa Indonesia lokal belum tersedia.'
          : '';
  const shouldOfferVoiceSettings = Boolean(wordText) && !canSpeak;

  useEffect(() => {
    if (!tts.subscribePlayback) {
      return;
    }

    return tts.subscribePlayback((playback) => {
      if (!mountedRef.current) {
        return;
      }
      if (playback.text && playback.text !== currentWordTextRef.current) {
        return;
      }
      if (playback.status === 'speaking') {
        if (
          currentWordTextRef.current &&
          playback.text === currentWordTextRef.current
        ) {
          activeSpeechRef.current = true;
          setSpeaking(true);
        }
        return;
      }

      activeSpeechRef.current = false;
      setSpeaking(false);
      if (playback.status === 'error') {
        setActionMessage(playback.message);
      } else if (playback.reason === 'done') {
        setActionMessage('Pembacaan selesai.');
      }
    });
  }, [tts]);

  useEffect(() => {
    const previousKey = previousWordSpeechKeyRef.current;
    previousWordSpeechKeyRef.current = wordSpeechKey;
    if (!previousKey || previousKey === wordSpeechKey) {
      return;
    }

    setHasSpoken(false);
    if (activeSpeechRef.current) {
      activeSpeechRef.current = false;
      setSpeaking(false);
      void tts.stop().catch(() => undefined);
    }
  }, [tts, wordSpeechKey]);

  useEffect(() => {
    if (
      !stableKey ||
      !preferences.hapticsEnabled ||
      hapticKeysRef.current.has(stableKey)
    ) {
      return;
    }

    hapticKeysRef.current.add(stableKey);
    void Haptics.notificationAsync(
      Haptics.NotificationFeedbackType.Success,
    ).catch(() => undefined);
  }, [preferences.hapticsEnabled, stableKey]);

  const retry = async () => {
    if (retryInFlightRef.current) {
      return;
    }

    retryInFlightRef.current = true;
    setRetrying(true);
    acceptsEventsRef.current = false;
    activeSessionIdRef.current = '';
    setActiveSessionId('');
    const startAttempt = ++startAttemptRef.current;
    setActionMessage('Memeriksa kembali pemindaian.');
    invalidateCandidate();

    try {
      if (activeSpeechRef.current) {
        activeSpeechRef.current = false;
        await tts.stop().catch(() => undefined);
      }

      if (startedRef.current || startPendingRef.current) {
        startedRef.current = false;
        startPendingRef.current = false;
        setScanActive(false);
        const stopResult = await runScannerCommand(() => scanner.stop());
        if (!stopResult.ok) {
          throw new Error(stopResult.message);
        }
      }

      if (!mountedRef.current || startAttemptRef.current !== startAttempt) {
        return;
      }

      setPaused(false);
      setTorchEnabled(false);
      setSpeaking(false);
      setHasSpoken(false);

      const permissionRequest = ++permissionRequestSequenceRef.current;
      const permissionCheck =
        permission.status === 'undetermined' || permission.canAskAgain
          ? cameraPermission.request()
          : cameraPermission.getStatus();
      const [nextPermission, nextScanner] = await Promise.all([
        permissionCheck,
        scanner.getAvailability(),
      ]);

      if (
        !mountedRef.current ||
        startAttemptRef.current !== startAttempt ||
        permissionRequest !== permissionRequestSequenceRef.current
      ) {
        return;
      }

      setPermission(nextPermission);
      setScannerAvailability(nextScanner);

      if (nextPermission.status === 'granted' && nextScanner.available) {
        startPendingRef.current = true;
        const result = await runScannerCommand(() => scanner.start());
        startPendingRef.current = false;

        if (!mountedRef.current || startAttemptRef.current !== startAttempt) {
          return;
        }

        const sessionId = result.ok ? result.sessionId.trim() : '';
        const started = result.ok && sessionId.length > 0;
        startedRef.current = started;
        activeSessionIdRef.current = sessionId;
        setActiveSessionId(sessionId);
        acceptsEventsRef.current = started;
        setScanActive(started);

        if (started) {
          hapticKeysRef.current.clear();
          eventRef.current = null;
          setEvent(null);
          invalidEventDetectedRef.current = false;
          setInvalidEventDetected(false);
          setPaused(false);
          setTorchEnabled(false);
          setSpeaking(false);
          setHasSpoken(false);
        }

        setActionMessage(
          started ? 'Pemindaian dimulai.' : 'Pemindaian tidak dapat dimulai.',
        );
        return;
      }

      setActionMessage(
        nextScanner.available
          ? 'Izin kamera belum diberikan.'
          : 'Pemindaian belum tersedia pada perangkat ini.',
      );
    } catch {
      if (mountedRef.current && startAttemptRef.current === startAttempt) {
        startedRef.current = false;
        startPendingRef.current = false;
        acceptsEventsRef.current = false;
        setScanActive(false);
        setActionMessage('Pemindaian tidak dapat dimulai ulang.');
      }
    } finally {
      retryInFlightRef.current = false;
      if (mountedRef.current && startAttemptRef.current === startAttempt) {
        setRetrying(false);
      }
    }
  };

  const openAppSettings = async () => {
    try {
      const result = await cameraPermission.openAppSettings();
      if (mountedRef.current) {
        setActionMessage(
          result.ok ? 'Pengaturan aplikasi dibuka.' : result.message,
        );
      }
    } catch {
      if (mountedRef.current) {
        setActionMessage('Pengaturan aplikasi tidak dapat dibuka.');
      }
    }
  };

  const togglePause = async () => {
    if (pauseInFlightRef.current) {
      return;
    }
    pauseInFlightRef.current = true;
    const controlAttempt = startAttemptRef.current;
    const shouldResume = paused;
    try {
      const result = await runScannerCommand(() =>
        shouldResume ? scanner.resume() : scanner.pause(),
      );
      if (
        !mountedRef.current ||
        controlAttempt !== startAttemptRef.current
      ) {
        return;
      }
      if (result.ok) {
        acceptsEventsRef.current = shouldResume;
        setPaused(!shouldResume);
        setActionMessage(
          shouldResume ? 'Pemindaian dilanjutkan.' : 'Pemindaian dijeda.',
        );
      } else {
        setActionMessage(
          shouldResume
            ? 'Pemindaian tidak dapat dilanjutkan.'
            : 'Pemindaian tidak dapat dijeda.',
        );
      }
    } catch {
      if (mountedRef.current) {
        setActionMessage(
          shouldResume
            ? 'Pemindaian tidak dapat dilanjutkan.'
            : 'Pemindaian tidak dapat dijeda.',
        );
      }
    } finally {
      pauseInFlightRef.current = false;
    }
  };

  const toggleTorch = async (enabled: boolean) => {
    if (torchInFlightRef.current || !torchAvailable) {
      return;
    }
    torchInFlightRef.current = true;
    const controlAttempt = startAttemptRef.current;
    try {
      const result = await runScannerCommand(() => scanner.setTorch(enabled));
      if (
        !mountedRef.current ||
        controlAttempt !== startAttemptRef.current
      ) {
        return;
      }
      if (result.ok) {
        setTorchEnabled(enabled);
        setActionMessage(enabled ? 'Senter dinyalakan.' : 'Senter dimatikan.');
      } else {
        setActionMessage('Senter tidak dapat diubah.');
      }
    } catch {
      if (mountedRef.current) {
        setActionMessage('Senter tidak dapat diubah.');
      }
    } finally {
      torchInFlightRef.current = false;
    }
  };

  const addStableLetter = () => {
    if (!stableText || !candidateEvidenceKey || !canAddStableLetter) {
      return;
    }
    const nextDraft = applyWordAction({
      type: 'commit_letter',
      evidenceKey: candidateEvidenceKey,
      letter: stableText,
    });
    if (!nextDraft) {
      return;
    }

    const nextWord = getWordText(nextDraft);
    invalidateCandidate();
    setActionMessage(
      `Huruf ${stableText} ditambahkan. Kata sekarang ${nextWord}. Pindai huruf berikutnya.`,
    );
  };

  const removeLastLetter = () => {
    const currentLetters = wordDraftRef.current.letters;
    const removedLetter = currentLetters[currentLetters.length - 1];
    const nextDraft = applyWordAction({ type: 'remove_last_letter' });
    if (!nextDraft || !removedLetter) {
      return;
    }
    const nextWord = getWordText(nextDraft);
    setActionMessage(
      nextWord
        ? `Huruf ${removedLetter} dihapus. Kata sekarang ${nextWord}.`
        : `Huruf ${removedLetter} dihapus. Kata sekarang kosong.`,
    );
  };

  const clearWord = () => {
    const nextDraft = applyWordAction({ type: 'clear_word' });
    if (!nextDraft) {
      return;
    }
    invalidateCandidate();
    setActionMessage('Kata dihapus. Mulai lagi dari huruf pertama.');
  };

  const rescanCurrentCell = () => {
    invalidateCandidate();
    setActionMessage('Hasil huruf dibersihkan. Pindai ulang sel saat ini.');
  };

  const speak = async () => {
    if (!wordText) {
      return;
    }
    const requestedWordKey = wordSpeechKey;
    const speechAction = ++speechActionSequenceRef.current;
    try {
      const result = await tts.speak(wordText, {
        rate: preferences.speechRate,
        volume: preferences.speechVolume,
      });
      if (
        !mountedRef.current ||
        currentWordSpeechKeyRef.current !== requestedWordKey ||
        speechAction !== speechActionSequenceRef.current
      ) {
        // Hanya request terbaru boleh menghentikan audio. Request yang lebih
        // lama tidak boleh mengantrekan stop setelah pembacaan baru dimulai.
        if (result.ok && speechAction === speechActionSequenceRef.current) {
          void tts.stop().catch(() => undefined);
        }
        return;
      }
      if (!tts.subscribePlayback) {
        activeSpeechRef.current = result.ok;
        setSpeaking(result.ok);
      } else if (!result.ok) {
        activeSpeechRef.current = false;
        setSpeaking(false);
      }
      setHasSpoken(result.ok);
      setActionMessage(
        result.ok ? 'Membacakan kata.' : 'Suara tidak tersedia.',
      );
    } catch {
      if (
        mountedRef.current &&
        speechAction === speechActionSequenceRef.current &&
        currentWordSpeechKeyRef.current === requestedWordKey
      ) {
        activeSpeechRef.current = false;
        setSpeaking(false);
        setActionMessage('Suara tidak tersedia.');
      }
    }
  };

  const stopSpeaking = async () => {
    const speechAction = ++speechActionSequenceRef.current;
    try {
      const result = await tts.stop();
      if (
        mountedRef.current &&
        speechAction === speechActionSequenceRef.current
      ) {
        if (result.ok) {
          activeSpeechRef.current = false;
          setSpeaking(false);
          setActionMessage('Suara dihentikan.');
        } else {
          setActionMessage('Suara tidak dapat dihentikan.');
        }
      }
    } catch {
      if (
        mountedRef.current &&
        speechAction === speechActionSequenceRef.current
      ) {
        setActionMessage('Suara tidak dapat dihentikan.');
      }
    }
  };

  const repeat = async () => {
    if (!wordText || !wordSpeechKey) {
      return;
    }
    const requestedWordKey = wordSpeechKey;
    const speechAction = ++speechActionSequenceRef.current;
    try {
      const result = await tts.speak(wordText, {
        rate: preferences.speechRate,
        volume: preferences.speechVolume,
      });
      if (
        !mountedRef.current ||
        currentWordSpeechKeyRef.current !== requestedWordKey ||
        speechAction !== speechActionSequenceRef.current
      ) {
        if (result.ok && speechAction === speechActionSequenceRef.current) {
          void tts.stop().catch(() => undefined);
        }
        return;
      }
      if (!tts.subscribePlayback) {
        activeSpeechRef.current = result.ok;
        setSpeaking(result.ok);
      } else if (!result.ok) {
        activeSpeechRef.current = false;
        setSpeaking(false);
      }
      setActionMessage(
        result.ok ? 'Mengulangi kata.' : 'Kata tidak dapat diulangi.',
      );
    } catch {
      if (
        mountedRef.current &&
        speechAction === speechActionSequenceRef.current &&
        currentWordSpeechKeyRef.current === requestedWordKey
      ) {
        activeSpeechRef.current = false;
        setSpeaking(false);
        setActionMessage('Kata tidak dapat diulangi.');
      }
    }
  };

  const finishScan = () => {
    if (finishInFlightRef.current) {
      return;
    }
    finishInFlightRef.current = true;
    const shouldStopSpeech = activeSpeechRef.current;
    const shouldStopScanner = startedRef.current || startPendingRef.current;

    startAttemptRef.current += 1;
    activeSessionIdRef.current = '';
    setActiveSessionId('');
    retryInFlightRef.current = false;
    acceptsEventsRef.current = false;
    startedRef.current = false;
    startPendingRef.current = false;
    activeSpeechRef.current = false;
    wordDraftRef.current = INITIAL_WORD_DRAFT_STATE;
    currentWordTextRef.current = '';
    currentWordSpeechKeyRef.current = '';
    setLocalWordDraft(INITIAL_WORD_DRAFT_STATE);
    onWordDraftChange?.(INITIAL_WORD_DRAFT_STATE);
    setRetrying(false);
    setScanActive(false);
    setSpeaking(false);

    const cleanupTasks: Promise<unknown>[] = [];
    if (shouldStopSpeech) {
      cleanupTasks.push(tts.stop().catch(() => undefined));
    }
    if (shouldStopScanner) {
      cleanupTasks.push(
        runScannerCommand(() => scanner.stop()).catch(() => undefined),
      );
    }
    void Promise.all(cleanupTasks);

    leaveLive();
  };

  return (
    <Screen testID="live-screen">
      <Text accessibilityRole="header" selectable style={styles.screenTitle}>
        Susun satu kata
      </Text>
      <StatusAnnouncer
        guidance={presentation.guidance ?? ''}
        stableAnnouncement={
          stableText
            ? `Huruf Braille ${stableText} terdeteksi. Periksa hasil sebelum digunakan.`
            : 'Huruf terdeteksi. Periksa hasil sebelum digunakan.'
        }
        stableKey={stableKey}
        status={event?.status === 'stable' && !stableText ? 'error' : event?.status ?? 'error'}
        title={presentation.title}
      />

      {scanActive
        ? renderScannerPreview?.({
              active:
                scanActive && !paused && permission.status === 'granted',
              resetToken: scannerResetToken,
              sessionId: activeSessionId,
              torchEnabled,
            }) ?? (
            <View
              accessibilityElementsHidden
              accessible={false}
              importantForAccessibility="no-hide-descendants"
              style={styles.preview}
            >
              <View style={styles.targetArea}>
                <Text style={styles.previewText}>Area target satu sel</Text>
              </View>
            </View>
          )
        : null}

      {!needsRecovery ? (
        <View
          accessibilityLabel={`${presentation.title}. ${presentation.guidance ?? ''}`}
          accessibilityRole="summary"
          accessible
          style={[styles.statusPanel, statusToneStyles[presentation.tone]]}
        >
          <Text
            accessible={false}
            importantForAccessibility="no-hide-descendants"
            style={styles.statusTitle}
          >
            {presentation.title}
          </Text>
          {presentation.guidance ? (
            <Text
              accessible={false}
              importantForAccessibility="no-hide-descendants"
              style={styles.body}
            >
              {presentation.guidance}
            </Text>
          ) : null}
        </View>
      ) : null}

      {scanActive ? (
        <SectionCard
          description="Susun satu kata, satu sel per huruf. Periksa huruf sebelum menambahkannya."
          style={styles.workspaceCard}
          title="Huruf dan kata"
        >
          <View style={styles.workspaceResults}>
            <View
              style={[
                styles.workspaceResultGroup,
                styles.candidateResultGroup,
              ]}
            >
              <Text selectable style={styles.workspaceLabel}>
                HURUF TERDETEKSI
              </Text>
              {stableText ? (
                <Text
                  accessibilityLabel={`Huruf Braille terdeteksi: ${stableText}`}
                  selectable
                  style={styles.candidateResult}
                >
                  {stableText}
                </Text>
              ) : (
                <Text selectable style={styles.candidateEmpty}>
                  Belum ada huruf stabil.
                </Text>
              )}
            </View>
            <View style={[styles.workspaceResultGroup, styles.wordResultGroup]}>
              <Text selectable style={styles.workspaceLabel}>
                KATA {wordDraft.letters.length} DARI {MAX_WORD_LENGTH}
              </Text>
              <Text
                accessibilityLabel={
                  wordText ? `Kata tersusun: ${wordText}` : 'Kata masih kosong'
                }
                selectable
                style={styles.wordResult}
              >
                {wordText || '-'}
              </Text>
            </View>
          </View>
          <View style={styles.primaryActions}>
            <AccessibleButton
              accessibilityHint={
                wordIsFull
                  ? `Kata sudah mencapai batas ${MAX_WORD_LENGTH} huruf`
                  : stableText
                    ? `Menambahkan huruf ${stableText} ke kata, lalu memindai sel berikutnya`
                    : 'Tersedia setelah satu huruf terdeteksi stabil'
              }
              disabled={!canAddStableLetter}
              label="Tambah huruf"
              onPress={addStableLetter}
              style={styles.controlButton}
              variant="action"
            />
            <AccessibleButton
              accessibilityHint={
                canSpeak
                  ? 'Membacakan kata tersusun menggunakan suara lokal'
                  : voiceDisabledMessage
              }
              disabled={!canSpeak}
              label="Bacakan"
              onPress={() => void speak()}
              style={styles.controlButton}
              variant="secondary"
            />
          </View>
          {wordIsFull ? (
            <Text accessibilityRole="alert" selectable style={styles.body}>
              Batas {MAX_WORD_LENGTH} huruf tercapai. Hapus satu huruf untuk
              melanjutkan.
            </Text>
          ) : null}
        </SectionCard>
      ) : null}

      {needsRecovery ? (
        <SectionCard
          description={
            !scannerReady
              ? 'Tidak ada gambar atau hasil pengganti yang dibuat.'
              : permissionRequiresSettings
                ? 'Buka pengaturan aplikasi Android untuk memberi izin kamera.'
                : permission.status !== 'granted'
              ? 'BrailleMe memerlukan akses kamera hanya ketika Anda memulai pemindaian.'
              : 'Coba lagi untuk memulai pemindaian.'
          }
          title={recoveryTitle}
        >
          <Text selectable style={styles.body}>
            {recoveryMessage}
          </Text>
          <View style={styles.actions}>
            {canRetryRecovery ? (
              <AccessibleButton
                accessibilityHint={
                  canRequestCameraPermission
                    ? 'Meminta izin kamera melalui dialog Android'
                    : 'Memeriksa kembali kamera dan memulai pemindaian'
                }
                disabled={retrying}
                label={
                  canRequestCameraPermission ? 'Izinkan kamera' : 'Coba lagi'
                }
                onPress={() => void retry()}
                variant="secondary"
              />
            ) : (
              <AccessibleButton
                accessibilityHint="Membuka penjelasan status dan batas aplikasi"
                label="Buka Bantuan"
                onPress={() => router.push('/help')}
                variant="secondary"
              />
            )}
            {permissionRequiresSettings ? (
              <AccessibleButton
                accessibilityHint="Membuka pengaturan aplikasi Android"
                label="Buka pengaturan aplikasi"
                onPress={() => void openAppSettings()}
                variant="secondary"
              />
            ) : null}
            <AccessibleButton
              label="Kembali"
              onPress={leaveLive}
              variant="secondary"
            />
          </View>
        </SectionCard>
      ) : null}

      {scanActive ? (
        <SectionCard title="Koreksi dan suara" variant="plain">
          <View style={styles.controlActions}>
            <AccessibleButton
              accessibilityHint="Membuang hasil huruf saat ini dan memulai pemindaian baru"
              label="Pindai ulang sel"
              onPress={rescanCurrentCell}
              style={styles.controlButton}
              variant="secondary"
            />
            <AccessibleButton
              accessibilityHint="Menghapus huruf terakhir dari kata tersusun"
              disabled={!wordText}
              label="Hapus huruf terakhir"
              onPress={removeLastLetter}
              style={styles.controlButton}
              variant="secondary"
            />
            <AccessibleButton
              accessibilityHint="Menghapus semua huruf dari kata tersusun"
              disabled={!wordText}
              label="Hapus kata"
              onPress={clearWord}
              style={styles.controlButton}
              variant="danger"
            />
            <AccessibleButton
              disabled={!speaking}
              label="Hentikan suara"
              onPress={() => void stopSpeaking()}
              style={styles.controlButton}
              variant="secondary"
            />
            <AccessibleButton
              disabled={!canSpeak || !hasSpoken}
              label="Ulangi"
              onPress={() => void repeat()}
              style={styles.controlButton}
              variant="secondary"
            />
            {wordText && voiceDisabledMessage ? (
              <Text
                selectable
                style={[styles.body, styles.controlMessage]}
              >
                {voiceDisabledMessage}
              </Text>
            ) : null}
            {shouldOfferVoiceSettings ? (
              <AccessibleButton
                accessibilityHint="Membuka pengaturan suara dan pemeriksaan suara lokal"
                label="Buka pengaturan suara"
                onPress={() => router.push('/settings')}
                style={styles.controlButton}
                variant="secondary"
              />
            ) : null}
          </View>
        </SectionCard>
      ) : null}

      {scanActive ? <SectionCard title="Kontrol pemindaian" variant="plain">
        <View style={styles.controlActions}>
          <AccessibleButton
            disabled={!scanActive}
            label={paused ? 'Lanjut' : 'Jeda'}
            onPress={() => void togglePause()}
            style={styles.controlButton}
            variant="secondary"
          />
          <View style={styles.switchRow}>
            <Text
              accessible={false}
              importantForAccessibility="no-hide-descendants"
              style={styles.body}
            >
              {torchAvailable ? 'Senter' : 'Senter tidak tersedia'}
            </Text>
            <Switch
              accessibilityLabel={
                torchAvailable ? 'Senter' : 'Senter tidak tersedia'
              }
              accessibilityRole="switch"
              accessibilityState={{
                checked: torchEnabled,
                disabled: !scanActive || !torchAvailable,
              }}
              disabled={!scanActive || !torchAvailable}
              hitSlop={8}
              onValueChange={(enabled) => void toggleTorch(enabled)}
              style={styles.switch}
              thumbColor={
                torchEnabled ? colors.primary : colors.surfaceContainerLowest
              }
              trackColor={{
                false: colors.outline,
                true: colors.primary,
              }}
              value={torchEnabled}
            />
          </View>
        </View>
      </SectionCard> : null}

      {actionMessage ? (
        <Text
          accessibilityLiveRegion={
            actionMessage === 'Membacakan kata.' ? 'none' : 'polite'
          }
          selectable
          style={styles.actionMessage}
        >
          {actionMessage}
        </Text>
      ) : null}

      {scanActive ? (
        <AccessibleButton
          label="Akhiri pemindaian"
          onPress={() => void finishScan()}
          variant="danger"
        />
      ) : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  screenTitle: {
    ...typography.displayLargeMobile,
    color: colors.text,
  },
  preview: {
    minHeight: 220,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.inverseSurface,
    borderCurve: 'continuous',
    borderRadius: radii.lg,
    padding: spacing.xl,
  },
  targetArea: {
    width: '100%',
    minHeight: 88,
    alignItems: 'center',
    justifyContent: 'center',
    borderColor: colors.inverseOnSurface,
    borderRadius: radii.default,
    borderStyle: 'dashed',
    borderWidth: 2,
    padding: spacing.md,
  },
  previewText: {
    ...typography.body,
    color: colors.inverseOnSurface,
    textAlign: 'center',
  },
  statusPanel: {
    gap: spacing.xs,
    borderLeftWidth: 4,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  statusTitle: {
    ...typography.heading,
    color: colors.text,
  },
  body: {
    ...typography.body,
    color: colors.text,
  },
  workspaceCard: {
    backgroundColor: colors.surfaceContainerLowest,
    borderColor: colors.primary,
  },
  workspaceResults: {
    gap: spacing.xs,
  },
  workspaceResultGroup: {
    minHeight: 104,
    flexGrow: 1,
    justifyContent: 'space-between',
    gap: spacing.xs,
    backgroundColor: colors.primaryContainer,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  candidateResultGroup: {
    width: '100%',
    minHeight: 80,
    flexBasis: 'auto',
    flexGrow: 0,
    flexDirection: 'row',
    alignItems: 'center',
  },
  wordResultGroup: {
    width: '100%',
    flexBasis: 'auto',
  },
  workspaceLabel: {
    ...typography.labelCaps,
    flexShrink: 1,
    color: colors.onPrimaryContainer,
  },
  candidateResult: {
    fontFamily: typography.title.fontFamily,
    fontSize: 48,
    fontWeight: '700',
    lineHeight: 56,
    color: colors.text,
  },
  candidateEmpty: {
    ...typography.body,
    flexShrink: 1,
    color: colors.text,
    textAlign: 'right',
  },
  wordResult: {
    ...typography.displayLargeMobile,
    color: colors.text,
    flexShrink: 1,
    letterSpacing: 2,
    minHeight: minimumTouchTarget,
  },
  primaryActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  actions: {
    gap: spacing.sm,
  },
  controlActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  controlButton: {
    minWidth: 132,
    flexGrow: 1,
    flexBasis: 148,
  },
  controlMessage: {
    width: '100%',
  },
  switchRow: {
    minHeight: minimumTouchTarget,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.lg,
  },
  switch: {
    minHeight: minimumTouchTarget,
    minWidth: 56,
  },
  actionMessage: {
    ...typography.body,
    color: colors.text,
    backgroundColor: colors.surfaceContainerLow,
    borderLeftColor: colors.primary,
    borderLeftWidth: 4,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
});

const statusToneStyles = StyleSheet.create({
  neutral: {
    backgroundColor: colors.surfaceContainerLow,
    borderLeftColor: colors.primary,
  },
  warning: {
    backgroundColor: colors.warningContainer,
    borderLeftColor: colors.secondary,
  },
  success: {
    backgroundColor: colors.successContainer,
    borderLeftColor: colors.success,
  },
  error: {
    backgroundColor: colors.errorContainer,
    borderLeftColor: colors.error,
  },
});
