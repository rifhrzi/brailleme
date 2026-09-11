import Slider from '@react-native-community/slider';
import * as Haptics from 'expo-haptics';
import { useFocusEffect, useRouter } from 'expo-router';
import {
  useCallback,
  useEffect,
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
import type {
  WordPhotoCameraHandle,
  WordPhotoPreviewProps,
} from '@/features/live/word-photo-camera';
import {
  isBrailleLetter,
  type WordDraftState,
} from '@/features/live/word-draft';
import { WORD_DECODER_VERSION } from '@/ml/word-model-contract';
import { WORD_MODEL_EXPERIMENTAL } from '@/ml/word-model-assets';
import {
  cameraPermissionPort,
  ttsPort,
  type CameraPermissionPort,
  type TtsPort,
} from '@/services';
import { useAppPreferences } from '@/storage';
import { colors, minimumTouchTarget, spacing, typography } from '@/theme';
import type { CameraPermissionSnapshot, TtsAvailability } from '@/types';
import { isValidWordScanEvent, type WordScanEvent } from '@/types/word-scan';

type WordScanScreenProps = {
  readonly renderCamera: (props: WordPhotoPreviewProps) => ReactNode;
  readonly cameraPermission?: CameraPermissionPort;
  readonly tts?: TtsPort;
  readonly wordDraft?: WordDraftState | undefined;
  readonly onWordDraftChange?: ((value: WordDraftState) => void) | undefined;
};

const UNKNOWN_PERMISSION: CameraPermissionSnapshot = {
  status: 'unavailable',
  canAskAgain: false,
};
const UNKNOWN_VOICE: TtsAvailability = {
  status: 'unavailable',
  localVoiceAvailable: false,
  message: 'Suara lokal belum tersedia.',
};
let nextSession = 0;
const createSession = () => `word-${Date.now()}-${++nextSession}`;

function rejectionGuidance(reason: string | undefined) {
  switch (reason) {
    case 'dark':
      return 'Cahaya kurang. Tambahkan cahaya miring dari ujung atas kertas agar bayangan titik terlihat, lalu pindai ulang.';
    case 'glare':
      return 'Pantulan terlalu terang. Ubah arah cahaya lalu pindai ulang.';
    case 'blur':
      return 'Gambar kurang jelas. Pindai ulang, sentuh kata pada pratinjau untuk memfokuskan kamera, lalu tahan ponsel saat memotret.';
    case 'too_far':
      return 'Titik terlalu kecil. Dekatkan kamera dengan seluruh kata tetap berada dalam kotak.';
    case 'multiple_rows':
      return 'Letakkan satu kata dalam satu baris mendatar di dalam kotak.';
    case 'unsupported_pattern':
      return 'Ada pola yang belum didukung. Pindai satu kata Braille dasar A sampai Z.';
    case 'too_many_cells':
      return 'Pindai satu kata dengan maksimal 12 huruf.';
    case 'clipped':
    case 'gap':
      return 'Sebagian kata belum terlihat utuh. Sisakan ruang di kedua ujung kata lalu pindai ulang.';
    default:
      return 'Seluruh kata belum dapat dibaca dengan yakin. Perbaiki posisi, fokus, dan cahaya lalu pindai ulang.';
  }
}

export default function WordScanScreen({
  renderCamera: CameraPreview,
  cameraPermission = cameraPermissionPort,
  tts = ttsPort,
  wordDraft,
  onWordDraftChange,
}: WordScanScreenProps) {
  const router = useRouter();
  const preferences = useAppPreferences();
  const camera = useRef<WordPhotoCameraHandle>(null);
  const focused = useRef(false);
  const mounted = useRef(true);
  const requestId = useRef(0);
  const [sessionId, setSessionId] = useState(createSession);
  const sessionRef = useRef(sessionId);
  const speechSequence = useRef(0);
  const wordDraftRef = useRef({ wordDraft, onWordDraftChange });
  const eventRef = useRef<WordScanEvent | null>(null);
  const [active, setActive] = useState(false);
  const [checking, setChecking] = useState(true);
  const [permission, setPermission] = useState(UNKNOWN_PERMISSION);
  const [voice, setVoice] = useState(UNKNOWN_VOICE);
  const [ready, setReady] = useState(false);
  const [torchAvailable, setTorchAvailable] = useState(false);
  const [torchEnabled, setTorchEnabled] = useState(false);
  const [guideAspectRatio, setGuideAspectRatio] = useState(4);
  const [event, setEvent] = useState<WordScanEvent | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [speechMessage, setSpeechMessage] = useState('');

  useEffect(() => {
    wordDraftRef.current = { wordDraft, onWordDraftChange };
  }, [wordDraft, onWordDraftChange]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const replaceDraft = useCallback((text: string, evidence: string) => {
    const current = wordDraftRef.current;
    current.onWordDraftChange?.({
      letters: [...text].filter(isBrailleLetter),
      consumedEvidenceKeys: evidence ? [evidence] : [],
      revision: (current.wordDraft?.revision ?? 0) + 1,
    });
  }, []);

  const invalidate = useCallback(() => {
    const next = createSession();
    sessionRef.current = next;
    requestId.current = 0;
    eventRef.current = null;
    ++speechSequence.current;
    setSessionId(next);
    setEvent(null);
    setReady(false);
    setTorchEnabled(false);
    setSpeaking(false);
    setSpeechMessage('');
    replaceDraft('', '');
    void tts.stop().catch(() => undefined);
  }, [replaceDraft, setSpeaking, setTorchEnabled, tts]);

  const checkServices = useCallback(
    async (requestPermission = false) => {
      const session = sessionRef.current;
      setChecking(true);
      try {
        const snapshot = requestPermission
          ? await cameraPermission.request()
          : await cameraPermission.getStatus();
        if (
          !mounted.current ||
          !focused.current ||
          session !== sessionRef.current
        )
          return;
        setPermission(snapshot);
        tts.invalidateAvailabilityCache?.();
        const currentVoice = await tts
          .getAvailability()
          .catch(() => UNKNOWN_VOICE);
        if (
          !mounted.current ||
          !focused.current ||
          session !== sessionRef.current
        )
          return;
        setVoice(currentVoice);
      } catch {
        if (
          mounted.current &&
          focused.current &&
          session === sessionRef.current
        )
          setPermission(UNKNOWN_PERMISSION);
      } finally {
        if (
          mounted.current &&
          focused.current &&
          session === sessionRef.current
        )
          setChecking(false);
      }
    },
    [cameraPermission, tts],
  );

  useFocusEffect(
    useCallback(() => {
      focused.current = true;
      setActive(AppState.currentState === 'active');
      void checkServices();
      return () => {
        focused.current = false;
        setActive(false);
        invalidate();
      };
    }, [checkServices, invalidate]),
  );

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      if (state !== 'active') {
        setActive(false);
        invalidate();
      } else if (focused.current) {
        setActive(true);
        void checkServices();
      }
    });
    return () => subscription.remove();
  }, [checkServices, invalidate]);

  useEffect(
    () =>
      tts.subscribePlayback?.((playback) => {
        if (!mounted.current || !focused.current) return;
        if (
          playback.status === 'speaking' &&
          eventRef.current?.status === 'success' &&
          playback.text === eventRef.current.text
        )
          setSpeaking(true);
        else if (playback.status !== 'speaking') {
          setSpeaking(false);
          if (playback.status === 'error')
            setSpeechMessage(
              'Kata belum dapat dibacakan. Periksa pengaturan suara.',
            );
        }
      }),
    [tts],
  );

  useEffect(() => {
    if (preferences.muted || preferences.speechVolume === 0) {
      const sequence = ++speechSequence.current;
      void tts
        .stop()
        .catch(() => undefined)
        .then(() => {
          if (mounted.current && sequence === speechSequence.current)
            setSpeaking(false);
        });
    }
  }, [preferences.muted, preferences.speechVolume, tts]);

  const onEvent = useCallback(
    (next: WordScanEvent) => {
      if (
        !mounted.current ||
        !focused.current ||
        !isValidWordScanEvent(next) ||
        next.sessionId !== sessionRef.current
      )
        return;
      // captureId zero is reserved for camera startup errors before the shutter.
      if (
        next.captureId !== requestId.current &&
        !(
          next.captureId === 0 &&
          next.status === 'error' &&
          requestId.current === 0
        )
      )
        return;
      if (
        eventRef.current &&
        ['success', 'rejected', 'error'].includes(eventRef.current.status)
      )
        return;
      eventRef.current = next;
      setEvent(next);
      if (next.status === 'success' && next.text) {
        replaceDraft(next.text, `${next.sessionId}:${next.captureId}`);
        if (preferences.hapticsEnabled)
          void Haptics.notificationAsync(
            Haptics.NotificationFeedbackType.Success,
          ).catch(() => undefined);
      }
    },
    [preferences.hapticsEnabled, replaceDraft],
  );

  const onAvailability = useCallback(
    (isReady: boolean, hasTorch: boolean) => {
      if (sessionRef.current !== sessionId || !mounted.current) return;
      setReady(isReady);
      setTorchAvailable(hasTorch);
    },
    [sessionId],
  );
  const registerCamera = useCallback((handle: WordPhotoCameraHandle | null) => {
    camera.current = handle;
  }, []);
  const changeGuideWidth = (value: number) => {
    if (!active || requestId.current !== 0 || !Number.isFinite(value)) return;
    setGuideAspectRatio(Math.min(8, Math.max(1, value)));
  };

  const scan = () => {
    if (
      !active ||
      !ready ||
      permission.status !== 'granted' ||
      requestId.current !== 0
    )
      return;
    requestId.current = 1;
    ++speechSequence.current;
    setSpeechMessage('');
    replaceDraft('', '');
    const capturing: WordScanEvent = {
      schemaVersion: 3,
      sessionId,
      captureId: 1,
      timestampMs: performance.now(),
      status: 'capturing',
      cells: [],
      boxes: [],
      decoderVersion: WORD_DECODER_VERSION,
    };
    eventRef.current = capturing;
    setEvent(capturing);
    void tts.stop().catch(() => undefined);
    camera.current?.capture(1);
  };

  const speak = async () => {
    const result = eventRef.current;
    if (
      !active ||
      result?.status !== 'success' ||
      !result.text ||
      preferences.muted ||
      preferences.speechVolume === 0 ||
      voice.status !== 'available' ||
      !voice.localVoiceAvailable
    )
      return;
    const sequence = ++speechSequence.current;
    const session = sessionRef.current;
    try {
      const outcome = await tts.speak(result.text, {
        rate: preferences.speechRate,
        volume: preferences.speechVolume,
      });
      if (
        !mounted.current ||
        !focused.current ||
        session !== sessionRef.current ||
        sequence !== speechSequence.current
      )
        return;
      if (!tts.subscribePlayback) setSpeaking(outcome.ok);
      if (!outcome.ok)
        setSpeechMessage(
          'Kata belum dapat dibacakan. Periksa pengaturan suara.',
        );
    } catch {
      if (mounted.current && sequence === speechSequence.current)
        setSpeechMessage(
          'Kata belum dapat dibacakan. Periksa pengaturan suara.',
        );
    }
  };

  const busy = event?.status === 'capturing' || event?.status === 'processing';
  const success = event?.status === 'success';
  const failed = event?.status === 'rejected' || event?.status === 'error';
  const canSpeak =
    active &&
    success &&
    !preferences.muted &&
    preferences.speechVolume > 0 &&
    voice.status === 'available' &&
    voice.localVoiceAvailable;
  const title = busy
    ? event.status === 'capturing'
      ? 'Mengambil foto'
      : 'Membaca kata'
    : success
      ? 'Kata terbaca'
      : failed
        ? 'Pindai ulang kata'
        : 'Siapkan satu kata';
  const guidance = busy
    ? 'Tunggu hingga seluruh kata selesai diperiksa.'
    : success
      ? 'Pilih Bacakan untuk mendengar kata, atau Pindai ulang untuk mengambil foto baru.'
      : event?.status === 'error'
        ? event.captureId === 0
          ? 'Kamera belum dapat dibuka. Pilih Pindai ulang untuk membuka kamera kembali. Jika masih gagal, tutup aplikasi lain yang memakai kamera lalu coba lagi.'
          : 'Foto belum dapat diproses. Coba pindai ulang atau buka Bantuan.'
        : failed
          ? rejectionGuidance(event.reason)
          : 'Tempatkan satu kata utuh dalam kotak, maksimal 12 huruf A sampai Z. Posisikan kamera tepat di atas kata. Arahkan cahaya miring dari ujung atas kertas agar bayangan titik terlihat, lalu tekan Pindai kata.';

  return (
    <Screen testID="word-scan-screen">
      <Text accessibilityRole="header" selectable style={styles.title}>
        Pindai satu kata
      </Text>
      {WORD_MODEL_EXPERIMENTAL ? (
        <Text selectable style={styles.body}>
          Versi percobaan. Periksa kembali seluruh kata hasil pindai.
        </Text>
      ) : null}
      {checking ? (
        <Text accessibilityLiveRegion="polite" selectable style={styles.body}>
          Memeriksa izin kamera dan suara lokal.
        </Text>
      ) : permission.status !== 'granted' ? (
        <SectionCard title="Izin kamera diperlukan">
          <Text selectable style={styles.body}>
            Izinkan kamera untuk memotret satu kata Braille. Foto diproses di
            perangkat dan tidak disimpan.
          </Text>
          <AccessibleButton
            label={
              permission.canAskAgain || permission.status === 'undetermined'
                ? 'Izinkan kamera'
                : 'Buka pengaturan aplikasi'
            }
            onPress={() => {
              if (
                permission.canAskAgain ||
                permission.status === 'undetermined'
              )
                void checkServices(true);
              else
                void cameraPermission.openAppSettings().catch(() => undefined);
            }}
          />
        </SectionCard>
      ) : null}
      {active && !checking && permission.status === 'granted' ? (
        <CameraPreview
          active={active}
          sessionId={sessionId}
          torchEnabled={torchEnabled}
          guideAspectRatio={guideAspectRatio}
          cameraRef={registerCamera}
          onEvent={onEvent}
          onAvailability={onAvailability}
        />
      ) : null}
      {permission.status === 'granted' && !checking ? (
        <>
          {!event ? (
            <AccessibleButton
              accessibilityHint="Mengambil satu foto dan membaca seluruh kata dari kiri ke kanan"
              label="Pindai kata"
              disabled={!active || !ready}
              onPress={scan}
              variant="action"
            />
          ) : null}
          {!event ? (
            <View style={styles.guideControl}>
              <Text selectable style={styles.guideLabel}>
                Lebar bingkai
              </Text>
              <Text selectable style={styles.body}>
                Sesuaikan bingkai agar hanya memuat satu kata utuh.
              </Text>
              <Slider
                accessibilityLabel="Lebar bingkai"
                accessibilityHint="Persempit atau perlebar bingkai hingga satu kata utuh berada di dalamnya"
                accessibilityRole="adjustable"
                accessibilityState={{ disabled: !active || event !== null }}
                accessibilityValue={{
                  min: 1,
                  max: 8,
                  now: guideAspectRatio,
                  text: `${Math.round((guideAspectRatio / 8) * 100)} persen lebar maksimum`,
                }}
                accessibilityActions={[
                  { name: 'increment', label: 'Perlebar bingkai' },
                  { name: 'decrement', label: 'Persempit bingkai' },
                ]}
                onAccessibilityAction={({ nativeEvent }) => {
                  if (nativeEvent.actionName === 'increment')
                    changeGuideWidth(guideAspectRatio + 0.25);
                  else if (nativeEvent.actionName === 'decrement')
                    changeGuideWidth(guideAspectRatio - 0.25);
                }}
                disabled={!active || event !== null}
                minimumValue={1}
                maximumValue={8}
                step={0}
                value={guideAspectRatio}
                onValueChange={changeGuideWidth}
                minimumTrackTintColor={colors.primary}
                maximumTrackTintColor={colors.outline}
                thumbTintColor={colors.primary}
                style={styles.guideSlider}
              />
              <View style={styles.guideEnds}>
                <Text selectable style={styles.body}>
                  Pendek
                </Text>
                <Text selectable style={styles.body}>
                  Panjang
                </Text>
              </View>
            </View>
          ) : null}
          <SectionCard title={title}>
            <Text selectable style={styles.body}>
              {guidance}
            </Text>
            {active ? (
              <StatusAnnouncer
                status={
                  success
                    ? 'stable'
                    : failed
                      ? 'quality_issue'
                      : busy
                        ? 'candidate'
                        : 'searching'
                }
                title={title}
                guidance={guidance}
                stableKey={`${sessionId}:${event?.captureId ?? 0}`}
                stableAnnouncement="Kata terbaca. Pilih Bacakan untuk mendengar kata."
              />
            ) : null}
            {success ? (
              <Text
                accessibilityLabel={`Hasil kata: ${event.text}`}
                selectable
                style={styles.result}
              >
                {event.text}
              </Text>
            ) : null}
            {busy ? (
              <AccessibleButton
                accessibilityState={{ busy: true }}
                label="Memproses foto"
                disabled
              />
            ) : null}
            {success ? (
              <AccessibleButton
                label="Bacakan"
                disabled={!canSpeak || speaking}
                onPress={() => void speak()}
                variant="action"
              />
            ) : null}
            {success || failed ? (
              <AccessibleButton
                label="Pindai ulang"
                accessibilityHint="Menghapus foto dan hasil, lalu kembali ke kamera"
                onPress={invalidate}
                variant="secondary"
              />
            ) : null}
            {speaking ? (
              <AccessibleButton
                label="Hentikan suara"
                onPress={() => {
                  ++speechSequence.current;
                  void tts.stop().catch(() => undefined);
                  setSpeaking(false);
                }}
                variant="secondary"
              />
            ) : null}
            {success && !canSpeak ? (
              <>
                <Text selectable style={styles.body}>
                  {preferences.muted || preferences.speechVolume === 0
                    ? 'Suara dimatikan di pengaturan.'
                    : 'Suara lokal belum tersedia. Periksa pengaturan suara.'}
                </Text>
                <AccessibleButton
                  label="Buka pengaturan suara"
                  onPress={() => router.push('/settings')}
                  variant="secondary"
                />
              </>
            ) : null}
          </SectionCard>
          {!event ? (
            <View style={styles.switchRow}>
              <Text selectable style={[styles.body, styles.switchLabel]}>
                {torchAvailable ? 'Senter' : 'Senter tidak tersedia'}
              </Text>
              <Switch
                accessibilityLabel="Senter"
                accessibilityState={{
                  checked: torchEnabled,
                  disabled: !active || !torchAvailable || event !== null,
                }}
                disabled={!active || !torchAvailable || event !== null}
                onValueChange={setTorchEnabled}
                value={torchEnabled}
                style={styles.switch}
              />
            </View>
          ) : null}
        </>
      ) : null}
      {speechMessage ? (
        <Text accessibilityLiveRegion="polite" selectable style={styles.body}>
          {speechMessage}
        </Text>
      ) : null}
      <AccessibleButton
        label="Bantuan"
        onPress={() => router.push('/help')}
        variant="secondary"
      />
      <AccessibleButton
        label="Akhiri pemindaian"
        onPress={() => {
          invalidate();
          if (router.canGoBack()) router.back();
          else router.replace('/');
        }}
        variant="secondary"
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  title: { ...typography.displayLargeMobile, color: colors.text },
  body: { ...typography.body, color: colors.text },
  guideControl: { gap: spacing.xs },
  guideLabel: { ...typography.label, color: colors.text },
  guideSlider: {
    width: '100%',
    minHeight: minimumTouchTarget,
    minWidth: minimumTouchTarget,
  },
  guideEnds: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: spacing.md,
  },
  result: {
    ...typography.displayLargeMobile,
    color: colors.text,
    flexShrink: 1,
    letterSpacing: 2,
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: spacing.md,
    minHeight: minimumTouchTarget,
  },
  switch: { minWidth: 56, minHeight: minimumTouchTarget },
  switchLabel: { flexShrink: 1 },
});
