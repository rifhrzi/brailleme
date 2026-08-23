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
import { PROVISIONAL_MIN_CONFIDENCE } from '@/ml/prediction-stabilizer';
import {
  cameraPermissionPort,
  scannerPort,
  ttsPort,
  type CameraPermissionPort,
  type ScannerPort,
  type TtsPort,
} from '@/services';
import { useAlphaPreferences } from '@/storage';
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

  // Timestamp, confidence kandidat, dan metadata model tidak ditampilkan oleh
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
        guidance: 'Memeriksa kamera, model, dan suara lokal.',
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
      return { title: 'Memeriksa kandidat', tone: 'neutral' };
    case 'stable':
      return getValidatedStableText(event)
        ? {
            title: 'Kandidat satu sel stabil',
            guidance: 'Verifikasi huruf terhadap sumber Braille sebelum digunakan.',
            tone: 'success',
          }
        : {
            title: 'Belum dapat dipastikan',
            guidance: 'Kandidat belum dapat ditampilkan. Pindai kembali.',
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
  renderScannerPreview,
  scanner = scannerPort,
  tts = ttsPort,
}: LiveScreenViewProps) {
  const router = useRouter();
  const preferences = useAlphaPreferences();
  const startedRef = useRef(false);
  const startPendingRef = useRef(false);
  const acceptsEventsRef = useRef(false);
  const mountedRef = useRef(true);
  const startAttemptRef = useRef(0);
  const permissionRequestSequenceRef = useRef(0);
  const ttsRequestSequenceRef = useRef(0);
  const activeSessionIdRef = useRef('');
  const retryInFlightRef = useRef(false);
  const finishInFlightRef = useRef(false);
  const pauseInFlightRef = useRef(false);
  const torchInFlightRef = useRef(false);
  const scannerCommandQueueRef = useRef<Promise<void>>(Promise.resolve());
  const activeSpeechRef = useRef(false);
  const hapticKeysRef = useRef(new Set<string>());
  const currentStableKeyRef = useRef('');
  const eventRef = useRef<LiveScanEvent | null>(null);
  const invalidEventDetectedRef = useRef(false);
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
  }, [cameraPermission, tts]);

  useEffect(() => {
    let active = true;
    mountedRef.current = true;
    acceptsEventsRef.current = false;
    const startAttempt = ++startAttemptRef.current;

    const unsubscribe = scanner.subscribe((nextEvent) => {
      if (!active || !acceptsEventsRef.current) {
        return;
      }

      if (!isValidLiveScanEvent(nextEvent)) {
        currentStableKeyRef.current = '';
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

      if (nextEvent.sessionId !== activeSessionIdRef.current) {
        return;
      }

      if (invalidEventDetectedRef.current) {
        invalidEventDetectedRef.current = false;
        setInvalidEventDetected(false);
      }
      const nextStableText = getValidatedStableText(nextEvent);
      const nextStableKey = nextStableText
        ? `${nextEvent.sessionId}:${nextStableText}`
        : '';
      if (currentStableKeyRef.current !== nextStableKey) {
        currentStableKeyRef.current = nextStableKey;
        if (activeSpeechRef.current) {
          activeSpeechRef.current = false;
          void tts.stop().catch(() => undefined);
        }
        setSpeaking(false);
        setHasSpoken(false);
      }
      if (nextEvent.status === 'error') {
        acceptsEventsRef.current = false;
        setScanActive(false);
        setPaused(false);
        setTorchEnabled(false);
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
            currentStableKeyRef.current = '';
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
          'Komponen kamera atau model belum dapat digunakan. Tidak ada hasil yang dibuat.',
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
        guidance: 'Kandidat belum dapat ditampilkan. Pindai kembali.',
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
  const speechStableKey = stableText
    ? `${event?.sessionId ?? 'unknown'}:${stableText}`
    : '';
  const stableKey = stableText
    ? `${event?.sessionId ?? 'unknown'}:${scannerResetToken}:${stableText}`
    : '';
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
    Boolean(stableText) &&
    ttsAvailability.status === 'available' &&
    ttsAvailability.localVoiceAvailable &&
    !preferences.muted &&
    preferences.speechVolume > 0;
  const voiceDisabledMessage = !stableText
    ? 'Bacakan tersedia setelah kandidat huruf stabil.'
    : preferences.muted
      ? 'Suara aplikasi sedang dibisukan.'
      : preferences.speechVolume <= 0
        ? 'Volume suara aplikasi diatur ke nol persen.'
        : ttsAvailability.status !== 'available' ||
            !ttsAvailability.localVoiceAvailable
          ? ttsAvailability.message.trim() ||
            'Suara Bahasa Indonesia lokal belum tersedia.'
          : '';
  const shouldOfferVoiceSettings = Boolean(stableText) && !canSpeak;

  useEffect(() => {
    if (!tts.subscribePlayback) {
      return;
    }

    return tts.subscribePlayback((playback) => {
      if (!mountedRef.current) {
        return;
      }
      if (playback.status === 'speaking') {
        if (stableText && playback.text === stableText) {
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
  }, [stableText, tts]);

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
          currentStableKeyRef.current = '';
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

  const speak = async () => {
    if (!stableText) {
      return;
    }
    const requestedStableKey = speechStableKey;
    try {
      const result = await tts.speak(stableText, {
        rate: preferences.speechRate,
        volume: preferences.speechVolume,
      });
      if (
        !mountedRef.current ||
        currentStableKeyRef.current !== requestedStableKey
      ) {
        if (result.ok) {
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
        result.ok ? 'Membacakan hasil.' : 'Suara tidak tersedia.',
      );
    } catch {
      if (mountedRef.current) {
        activeSpeechRef.current = false;
        setSpeaking(false);
        setActionMessage('Suara tidak tersedia.');
      }
    }
  };

  const stopSpeaking = async () => {
    try {
      const result = await tts.stop();
      if (mountedRef.current) {
        if (result.ok) {
          activeSpeechRef.current = false;
          setSpeaking(false);
          setActionMessage('Suara dihentikan.');
        } else {
          setActionMessage('Suara tidak dapat dihentikan.');
        }
      }
    } catch {
      if (mountedRef.current) {
        setActionMessage('Suara tidak dapat dihentikan.');
      }
    }
  };

  const repeat = async () => {
    if (!speechStableKey) {
      return;
    }
    const requestedStableKey = speechStableKey;
    try {
      const result = await tts.repeat();
      if (
        !mountedRef.current ||
        currentStableKeyRef.current !== requestedStableKey
      ) {
        if (result.ok) {
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
        result.ok ? 'Mengulangi hasil.' : 'Hasil tidak dapat diulangi.',
      );
    } catch {
      if (mountedRef.current) {
        activeSpeechRef.current = false;
        setSpeaking(false);
        setActionMessage('Hasil tidak dapat diulangi.');
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

  const clearResult = async () => {
    if (activeSpeechRef.current) {
      await tts.stop().catch(() => undefined);
      activeSpeechRef.current = false;
    }
    eventRef.current = null;
    setEvent(null);
    invalidEventDetectedRef.current = false;
    setInvalidEventDetected(false);
    currentStableKeyRef.current = '';
    hapticKeysRef.current.clear();
    setSpeaking(false);
    setHasSpoken(false);
    setScannerResetToken((current) => current + 1);
    setActionMessage('Hasil dibersihkan dari layar.');
  };

  return (
    <Screen testID="live-screen">
      <StatusAnnouncer
        guidance={presentation.guidance ?? ''}
        stableAnnouncement={
          stableText
            ? `Kandidat huruf Braille ${stableText} stabil. Verifikasi hasil sebelum digunakan.`
            : 'Kandidat satu sel stabil. Verifikasi hasil sebelum digunakan.'
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

      {scanActive ? (
        <Text accessibilityRole="alert" selectable style={styles.warningText}>
          Eksperimental: pindai tepat satu sel A sampai Z. Kandidat dapat salah
          dan belum mendukung kata, spasi, atau satu baris.
        </Text>
      ) : null}

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
        stableText ? (
          <SectionCard title="Kandidat terakhir">
            <Text accessibilityLabel={`Kandidat huruf Braille: ${stableText}`} selectable style={styles.result}>
              {stableText}
            </Text>
          </SectionCard>
        ) : (
          <SectionCard
            description="Huruf akan muncul setelah gambar cukup jelas dan hasil stabil."
            title="Kandidat terakhir"
          >
            <Text selectable style={styles.body}>
              Belum ada kandidat.
            </Text>
          </SectionCard>
        )
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

      {scanActive ? <SectionCard title="Kontrol suara">
        <View style={styles.actions}>
          <AccessibleButton
            accessibilityHint={
              canSpeak
                ? 'Membacakan hasil stabil menggunakan suara lokal'
                : voiceDisabledMessage
            }
            disabled={!canSpeak}
            label="Bacakan"
            onPress={() => void speak()}
            variant="action"
          />
          <AccessibleButton
            disabled={!speaking}
            label="Hentikan suara"
            onPress={() => void stopSpeaking()}
            variant="secondary"
          />
          <AccessibleButton
            disabled={!canSpeak || !hasSpoken}
            label="Ulangi"
            onPress={() => void repeat()}
            variant="secondary"
          />
          <AccessibleButton
            disabled={!event}
            label="Bersihkan"
            onPress={() => void clearResult()}
            variant="secondary"
          />
          {voiceDisabledMessage ? (
            <Text
              accessibilityLiveRegion="polite"
              selectable
              style={styles.body}
            >
              {voiceDisabledMessage}
            </Text>
          ) : null}
          {shouldOfferVoiceSettings ? (
            <AccessibleButton
              accessibilityHint="Membuka pengaturan suara dan pemeriksaan suara lokal"
              label="Buka pengaturan suara"
              onPress={() => router.push('/settings')}
              variant="secondary"
            />
          ) : null}
        </View>
      </SectionCard> : null}

      {scanActive ? <SectionCard title="Kontrol pemindaian">
        <View style={styles.actions}>
          <AccessibleButton
            disabled={!scanActive}
            label={paused ? 'Lanjut' : 'Jeda'}
            onPress={() => void togglePause()}
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
          accessibilityLiveRegion="polite"
          accessibilityRole="alert"
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
  preview: {
    minHeight: 220,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primaryContainer,
    borderCurve: 'continuous',
    borderRadius: radii.lg,
    padding: spacing.xl,
  },
  targetArea: {
    width: '100%',
    minHeight: 88,
    alignItems: 'center',
    justifyContent: 'center',
    borderColor: colors.outline,
    borderRadius: radii.default,
    borderStyle: 'dashed',
    borderWidth: 2,
    padding: spacing.md,
  },
  previewText: {
    ...typography.body,
    color: colors.text,
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
  warningText: {
    ...typography.body,
    color: colors.secondary,
    fontWeight: '700',
  },
  result: {
    ...typography.title,
    color: colors.text,
  },
  actions: {
    gap: spacing.sm,
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
