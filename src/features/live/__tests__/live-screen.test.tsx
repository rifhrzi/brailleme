import {
  act,
  fireEvent,
  render,
  screen,
  userEvent,
  waitFor,
} from '@testing-library/react-native';
import * as Haptics from 'expo-haptics';
import { AppState, type AppStateStatus } from 'react-native';

import { LiveScreenView } from '@/features/live/live-screen';
import {
  INITIAL_WORD_DRAFT_STATE,
  MAX_WORD_LENGTH,
  wordDraftReducer,
  type WordDraftState,
} from '@/features/live/word-draft';
import type {
  CameraPermissionPort,
  LiveScanListener,
  ScannerCapabilitiesListener,
  ScannerPort,
  TtsPlaybackListener,
  TtsPort,
} from '@/services';
import { resetAppPreferences, updateAppPreferences } from '@/storage';
import type { LiveScanEvent, ScanStatus } from '@/types';

// Announcement timing memiliki suite tersendiri; suite ini fokus pada presentasi
// Live dan kontrak port agar timer throttle tidak memperlambat setiap status.
jest.mock('@/components', () => {
  const actual = jest.requireActual<typeof import('@/components')>(
    '@/components',
  );

  return {
    ...actual,
    StatusAnnouncer: jest.fn(() => null),
  };
});

jest.mock('expo-haptics', () => ({
  NotificationFeedbackType: { Success: 'success' },
  notificationAsync: jest.fn(async () => undefined),
}));

const OK_RESULT = { ok: true, sessionId: 'test-session' } as const;

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });

  return { promise, resolve };
}

type ScannerHarness = {
  port: ScannerPort;
  emit(event: LiveScanEvent): void;
};

function createScannerHarness(): ScannerHarness {
  let listener: LiveScanListener | undefined;

  const port: ScannerPort = {
    getAvailability: jest.fn(async () => ({ available: true as const })),
    subscribe: jest.fn((nextListener) => {
      listener = nextListener;
      return jest.fn();
    }),
    start: jest.fn(async () => OK_RESULT),
    pause: jest.fn(async () => OK_RESULT),
    resume: jest.fn(async () => OK_RESULT),
    stop: jest.fn(async () => OK_RESULT),
    setTorch: jest.fn(async () => OK_RESULT),
  };

  return {
    port,
    emit(event) {
      if (!listener) {
        throw new Error('Listener scanner belum terpasang.');
      }
      listener(event);
    },
  };
}

function createGrantedPermission(): CameraPermissionPort {
  return {
    getStatus: jest.fn(async () => ({
      status: 'granted' as const,
      canAskAgain: true,
    })),
    request: jest.fn(async () => ({
      status: 'granted' as const,
      canAskAgain: true,
    })),
    openAppSettings: jest.fn(async () => OK_RESULT),
  };
}

function createAvailableTts(): TtsPort {
  return {
    getAvailability: jest.fn(async () => ({
      status: 'available' as const,
      localVoiceAvailable: true,
      message: 'Suara lokal tersedia.',
    })),
    invalidateAvailabilityCache: jest.fn(),
    speak: jest.fn(async () => OK_RESULT),
    stop: jest.fn(async () => OK_RESULT),
    repeat: jest.fn(async () => OK_RESULT),
    openTtsSettings: jest.fn(async () => OK_RESULT),
  };
}

function createEvent(
  status: ScanStatus,
  overrides: Partial<LiveScanEvent> = {},
): LiveScanEvent {
  return {
    schemaVersion: 2,
    sessionId: 'test-session',
    scanEpoch: 0,
    timestampMs: 1_786_128_000_000,
    status,
    boxes: [],
    decoderVersion: 'test-decoder-v1',
    ...(status === 'stable' ? { confidence: 0.95 } : {}),
    ...overrides,
  };
}

async function renderReadyLive({
  onWordDraftChange,
  wordDraft,
}: {
  onWordDraftChange?: ((nextDraft: WordDraftState) => void) | undefined;
  wordDraft?: WordDraftState | undefined;
} = {}) {
  const scanner = createScannerHarness();
  const cameraPermission = createGrantedPermission();
  const tts = createAvailableTts();

  const view = await render(
    <LiveScreenView
      cameraPermission={cameraPermission}
      onWordDraftChange={onWordDraftChange}
      scanner={scanner.port}
      tts={tts}
      wordDraft={wordDraft}
    />,
  );
  await waitFor(() => {
    expect(scanner.port.start).toHaveBeenCalledTimes(1);
  });

  return { cameraPermission, scanner, tts, unmount: view.unmount };
}

async function emitStableAndAdd(
  scanner: ScannerHarness,
  user: ReturnType<typeof userEvent.setup>,
  letter: string,
  scanEpoch = 0,
) {
  await act(() => {
    scanner.emit(
      createEvent('stable', {
        confidence: 0.95,
        scanEpoch,
        text: letter,
        timestampMs: 1_786_128_000_000 + scanEpoch,
      }),
    );
  });
  await user.press(screen.getByRole('button', { name: 'Tambah huruf' }));
}

describe('Pindai Live', () => {
  beforeEach(() => {
    resetAppPreferences();
  });

  it('menampilkan status persiapan lalu mencari setelah scanner aktif', async () => {
    const scanner = createScannerHarness();
    const cameraPermission = createGrantedPermission();
    const tts = createAvailableTts();
    const deferredStart = createDeferred<typeof OK_RESULT>();
    const startMock = scanner.port.start as jest.MockedFunction<
      ScannerPort['start']
    >;
    startMock.mockReturnValueOnce(deferredStart.promise);

    await render(
      <LiveScreenView
        cameraPermission={cameraPermission}
        scanner={scanner.port}
        tts={tts}
      />,
    );

    await waitFor(() => {
      expect(startMock).toHaveBeenCalledTimes(1);
    });
    expect(
      screen.getByRole('summary', { name: /Menyiapkan pemindai/i }),
    ).toBeOnTheScreen();
    expect(
      screen.queryByRole('button', { name: 'Coba lagi' }),
    ).toBeNull();

    await act(() => {
      deferredStart.resolve(OK_RESULT);
    });

    await waitFor(() => {
      expect(
        screen.getByRole('summary', { name: /Mencari satu sel Braille/i }),
      ).toBeOnTheScreen();
    });
    expect(
      screen.getByRole('header', { name: 'Susun satu kata' }),
    ).toBeOnTheScreen();
    expect(
      screen.getByText(/Susun satu kata, satu sel per huruf/i),
    ).toBeOnTheScreen();
  });

  it('tetap memulai scanner ketika pemeriksaan suara gagal', async () => {
    const scanner = createScannerHarness();
    const tts = createAvailableTts();
    const user = userEvent.setup();
    tts.getAvailability = jest.fn(async () => {
      throw new Error('TTS tidak merespons.');
    });

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        scanner={scanner.port}
        tts={tts}
      />,
    );

    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));
    expect(
      screen.getByRole('summary', { name: /Mencari satu sel Braille/i }),
    ).toBeOnTheScreen();

    await emitStableAndAdd(scanner, user, 'A');
    expect(
      screen.getByText('Status suara lokal tidak dapat diperiksa.'),
    ).toBeOnTheScreen();
    expect(
      screen.getByRole('button', { name: 'Buka pengaturan suara' }),
    ).toBeOnTheScreen();
  });

  it.each<{
    status: ScanStatus;
    overrides?: Partial<LiveScanEvent>;
    accessibleName: RegExp;
  }>([
    {
      status: 'searching',
      accessibleName: /Belum menemukan satu sel.*Arahkan satu huruf Braille/i,
    },
    {
      status: 'quality_issue',
      overrides: { reason: 'dark' },
      accessibleName: /Cahaya kurang.*Pindah ke tempat terang/i,
    },
    {
      status: 'candidate',
      overrides: { text: 'A' },
      accessibleName: /Memastikan huruf/i,
    },
    {
      status: 'stable',
      overrides: { text: 'A', confidence: 0.91 },
      accessibleName: /Huruf terdeteksi.*Cocokkan huruf/i,
    },
    {
      status: 'unsupported',
      accessibleName: /Jenis Braille belum didukung.*Lihat Bantuan/i,
    },
    {
      status: 'error',
      accessibleName: /Pemindaian mengalami masalah.*Coba lagi/i,
    },
  ])(
    'mempresentasikan status $status dari test port',
    async ({ accessibleName, overrides, status }) => {
      const { scanner } = await renderReadyLive();

      await act(() => {
        scanner.emit(createEvent(status, overrides));
      });

      if (status === 'error') {
        expect(
          screen.getByRole('header', { name: 'Pemindaian bermasalah' }),
        ).toBeOnTheScreen();
        expect(
          screen.getByText(/Pemindaian mengalami masalah.*Tidak ada teks/i),
        ).toBeOnTheScreen();
      } else {
        expect(
          screen.getByRole('summary', { name: accessibleName }),
        ).toBeOnTheScreen();
      }

      if (status === 'stable') {
        expect(
          screen.getByLabelText('Huruf Braille terdeteksi: A'),
        ).toBeOnTheScreen();
        expect(screen.queryByText(/Keyakinan 91 persen/i)).toBeNull();
        expect(
          screen.getByRole('button', { name: 'Tambah huruf' }),
        ).toBeEnabled();
        expect(
          screen.getByRole('button', { name: 'Bacakan' }),
        ).toBeDisabled();
      } else if (status !== 'error') {
        expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();
        expect(screen.queryByLabelText(/^Huruf Braille terdeteksi:/)).toBeNull();
        expect(
          screen.getByRole('button', { name: 'Bacakan' }),
        ).toBeDisabled();
      }

      if (status === 'error') {
        expect(screen.getByRole('button', { name: 'Coba lagi' })).toBeEnabled();
        expect(screen.queryByRole('button', { name: 'Jeda' })).toBeNull();
        expect(screen.queryByRole('switch', { name: 'Senter' })).toBeNull();
        expect(screen.queryByText('Belum ada huruf stabil.')).toBeNull();
      }
    },
    10_000,
  );

  it('tidak mengklaim hasil siap ketika event stable tidak membawa teks', async () => {
    const { scanner } = await renderReadyLive();

    await act(() => {
      scanner.emit(createEvent('stable', { text: '   ', confidence: 0.99 }));
    });

    expect(
      screen.getByRole('summary', {
        name: /Belum dapat dipastikan.*Huruf belum dapat ditampilkan/i,
      }),
    ).toBeOnTheScreen();
    expect(
      screen.queryByRole('summary', { name: /Huruf terdeteksi/i }),
    ).toBeNull();
    expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();
    expect(
      screen.getByRole('button', { name: 'Bacakan' }),
    ).toBeDisabled();
  });

  it('mengabaikan event terlambat dari sesi scanner sebelumnya', async () => {
    const { scanner, tts } = await renderReadyLive();

    await act(() => {
      scanner.emit(
        createEvent('stable', {
          sessionId: 'old-session',
          text: 'HASIL LAMA',
          confidence: 0.99,
        }),
      );
    });

    expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();
    expect(screen.queryByText('HASIL LAMA')).toBeNull();
    expect(screen.getByRole('button', { name: 'Bacakan' })).toBeDisabled();
    expect(tts.speak).not.toHaveBeenCalled();
  });

  it.each([
    ['confidence tidak ada', undefined],
    ['confidence NaN', Number.NaN],
    ['confidence di bawah ambang', 0.89],
    ['confidence di atas satu', 1.01],
  ])(
    'menolak stable dengan %s agar hasil tidak dapat dibacakan',
    async (_caseName, confidence) => {
      const { scanner, tts } = await renderReadyLive();
      const event =
        confidence === undefined
          ? ({
              ...createEvent('stable', { text: 'A' }),
              confidence: undefined,
            } as unknown as LiveScanEvent)
          : createEvent('stable', { text: 'A', confidence });

      await act(() => {
        scanner.emit(event);
      });

      expect(
        screen.getByRole('summary', {
          name: /Belum dapat dipastikan.*Huruf belum dapat ditampilkan/i,
        }),
      ).toBeOnTheScreen();
      expect(screen.queryByLabelText('Huruf Braille terdeteksi: A')).toBeNull();
      expect(
        screen.getByRole('button', { name: 'Bacakan' }),
      ).toBeDisabled();
      expect(tts.speak).not.toHaveBeenCalled();
    },
  );

  it('menggunakan port untuk kontrol scan, senter, dan suara', async () => {
    const user = userEvent.setup();
    const { scanner, tts } = await renderReadyLive();

    expect(screen.getByRole('button', { name: 'Jeda' })).toBeEnabled();
    expect(
      screen.getByRole('switch', { name: 'Senter' }),
    ).not.toBeChecked();
    expect(
      screen.getByRole('switch', { name: 'Senter' }),
    ).toBeEnabled();
    expect(
      screen.getByRole('button', { name: 'Bacakan' }),
    ).toBeDisabled();

    await emitStableAndAdd(scanner, user, 'A');
    await emitStableAndAdd(scanner, user, 'B', 1);
    await emitStableAndAdd(scanner, user, 'C', 2);
    await user.press(screen.getByRole('button', { name: 'Bacakan' }));

    expect(tts.speak).toHaveBeenCalledWith('ABC', {
      rate: 1,
      volume: 1,
    });
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeEnabled();

    await user.press(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    );
    expect(tts.stop).toHaveBeenCalledTimes(1);

    await user.press(screen.getByRole('button', { name: 'Jeda' }));
    expect(scanner.port.pause).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Lanjut' })).toBeOnTheScreen();

    await user.press(screen.getByRole('button', { name: 'Lanjut' }));
    expect(scanner.port.resume).toHaveBeenCalledTimes(1);

    await fireEvent(
      screen.getByRole('switch', { name: 'Senter' }),
      'valueChange',
      true,
    );
    expect(scanner.port.setTorch).toHaveBeenCalledWith(true);
    expect(
      screen.getByRole('switch', { name: 'Senter' }),
    ).toBeChecked();
    expect(
      screen.getByRole('switch', { name: 'Senter' }),
    ).toBeEnabled();

    await user.press(screen.getByRole('button', { name: 'Pindai ulang sel' }));
    expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();
    expect(
      screen.getByLabelText('Kata tersusun: ABC'),
    ).toBeOnTheScreen();
  });

  it('menambahkan kandidat hanya sekali dan menerima huruf berulang pada epoch berikutnya', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const renderScannerPreview = jest.fn(() => null);

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        renderScannerPreview={renderScannerPreview}
        scanner={scanner.port}
        tts={createAvailableTts()}
      />,
    );
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));

    await act(() => {
      scanner.emit(createEvent('stable', { text: 'A' }));
    });
    const addButton = screen.getByRole('button', { name: 'Tambah huruf' });
    await fireEvent.press(addButton);
    await fireEvent.press(addButton);

    await waitFor(() => {
      expect(screen.getByLabelText('Kata tersusun: A')).toBeOnTheScreen();
    });
    await waitFor(() => {
      expect(renderScannerPreview).toHaveBeenLastCalledWith(
        expect.objectContaining({ resetToken: 1 }),
      );
    });

    await act(() => {
      scanner.emit(
        createEvent('stable', {
          scanEpoch: 0,
          text: 'B',
          timestampMs: 1_786_128_000_100,
        }),
      );
    });
    expect(screen.queryByLabelText('Huruf Braille terdeteksi: B')).toBeNull();

    await act(() => {
      scanner.emit(
        createEvent('stable', {
          scanEpoch: 1,
          text: 'A',
          timestampMs: 1_786_128_000_200,
        }),
      );
    });
    await user.press(screen.getByRole('button', { name: 'Tambah huruf' }));

    expect(screen.getByLabelText('Kata tersusun: AA')).toBeOnTheScreen();
    expect(renderScannerPreview).toHaveBeenLastCalledWith(
      expect.objectContaining({ resetToken: 2 }),
    );
  });

  it('menjelaskan batas kata kepada TalkBack sebelum hint kandidat stabil', async () => {
    const scanner = createScannerHarness();
    let fullWordDraft = INITIAL_WORD_DRAFT_STATE;
    for (let index = 0; index < MAX_WORD_LENGTH; index += 1) {
      fullWordDraft = wordDraftReducer(fullWordDraft, {
        type: 'commit_letter',
        evidenceKey: `full-word:${index}`,
        letter: 'A',
      });
    }

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        onWordDraftChange={jest.fn()}
        scanner={scanner.port}
        tts={createAvailableTts()}
        wordDraft={fullWordDraft}
      />,
    );
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));
    await act(() => {
      scanner.emit(createEvent('stable', { text: 'B' }));
    });

    const addButton = screen.getByRole('button', { name: 'Tambah huruf' });
    expect(addButton).toBeDisabled();
    expect(addButton.props.accessibilityHint).toBe(
      `Kata sudah mencapai batas ${MAX_WORD_LENGTH} huruf`,
    );
  });

  it('mendukung hapus huruf, hapus kata, dan pindai ulang tanpa mengubah draft', async () => {
    const user = userEvent.setup();
    const onWordDraftChange = jest.fn();
    const { scanner } = await renderReadyLive({ onWordDraftChange });

    await emitStableAndAdd(scanner, user, 'A');
    await emitStableAndAdd(scanner, user, 'B', 1);
    await user.press(
      screen.getByRole('button', { name: 'Hapus huruf terakhir' }),
    );
    expect(screen.getByLabelText('Kata tersusun: A')).toBeOnTheScreen();

    await act(() => {
      scanner.emit(createEvent('stable', { scanEpoch: 2, text: 'C' }));
    });
    await user.press(screen.getByRole('button', { name: 'Pindai ulang sel' }));
    expect(screen.getByLabelText('Kata tersusun: A')).toBeOnTheScreen();
    expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();

    await act(() => {
      scanner.emit(createEvent('stable', { scanEpoch: 2, text: 'C' }));
    });
    expect(screen.queryByLabelText('Huruf Braille terdeteksi: C')).toBeNull();

    onWordDraftChange.mockClear();
    await user.press(screen.getByRole('button', { name: 'Hapus kata' }));
    expect(screen.getByLabelText('Kata masih kosong')).toBeOnTheScreen();
    expect(onWordDraftChange).toHaveBeenCalledWith(
      expect.objectContaining({ letters: [] }),
    );
    expect(
      screen.getByRole('button', { name: 'Hapus huruf terakhir' }),
    ).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Hapus kata' })).toBeDisabled();
  });

  it('mengumumkan dan menghaptikkan hasil yang sama lagi setelah dipindai ulang', async () => {
    const user = userEvent.setup();
    const { scanner } = await renderReadyLive();
    const components = jest.requireMock<{
      StatusAnnouncer: jest.Mock;
    }>('@/components');

    await act(() => {
      scanner.emit(createEvent('stable', { text: 'A' }));
    });
    await waitFor(() => {
      expect(Haptics.notificationAsync).toHaveBeenCalledTimes(1);
    });
    expect(components.StatusAnnouncer.mock.lastCall?.[0]).toEqual(
      expect.objectContaining({
        stableAnnouncement: expect.stringMatching(/huruf Braille A/i),
        stableKey: 'test-session:0:1:A',
      }),
    );

    await user.press(screen.getByRole('button', { name: 'Pindai ulang sel' }));
    await act(() => {
      scanner.emit(createEvent('stable', { scanEpoch: 1, text: 'A' }));
    });

    await waitFor(() => {
      expect(Haptics.notificationAsync).toHaveBeenCalledTimes(2);
    });
    expect(components.StatusAnnouncer.mock.lastCall?.[0]).toEqual(
      expect.objectContaining({
        stableAnnouncement: expect.stringMatching(/huruf Braille A/i),
        stableKey: 'test-session:1:2:A',
      }),
    );
  });

  it('mengumumkan dan menghaptikkan urutan A ke B ke A tanpa mengulang frame sama', async () => {
    const { scanner } = await renderReadyLive();
    const components = jest.requireMock<{
      StatusAnnouncer: jest.Mock;
    }>('@/components');

    await act(() => {
      scanner.emit(createEvent('stable', { text: 'A' }));
    });
    await waitFor(() => {
      expect(Haptics.notificationAsync).toHaveBeenCalledTimes(1);
    });

    await act(() => {
      scanner.emit(createEvent('stable', { text: 'A', timestampMs: 2 }));
    });
    expect(Haptics.notificationAsync).toHaveBeenCalledTimes(1);

    await act(() => {
      scanner.emit(createEvent('stable', { text: 'B', timestampMs: 3 }));
    });
    await waitFor(() => {
      expect(Haptics.notificationAsync).toHaveBeenCalledTimes(2);
    });
    expect(components.StatusAnnouncer.mock.lastCall?.[0]).toEqual(
      expect.objectContaining({ stableKey: 'test-session:0:2:B' }),
    );

    await act(() => {
      scanner.emit(createEvent('stable', { text: 'A', timestampMs: 4 }));
    });
    await waitFor(() => {
      expect(Haptics.notificationAsync).toHaveBeenCalledTimes(3);
    });
    expect(components.StatusAnnouncer.mock.lastCall?.[0]).toEqual(
      expect.objectContaining({ stableKey: 'test-session:0:3:A' }),
    );
  });

  it('menjelaskan ketika suara dibisukan dan menyediakan jalan ke Pengaturan', async () => {
    updateAppPreferences({ muted: true });
    const { scanner } = await renderReadyLive();
    const routerModule = jest.requireMock<{
      __router: { push: jest.Mock };
    }>('expo-router');
    const user = userEvent.setup();

    await emitStableAndAdd(scanner, user, 'A');

    expect(
      screen.getByText('Suara aplikasi sedang dibisukan.'),
    ).toBeOnTheScreen();
    await user.press(
      screen.getByRole('button', { name: 'Buka pengaturan suara' }),
    );
    expect(routerModule.__router.push).toHaveBeenCalledWith('/settings');
  });

  it('mengikuti lifecycle completion dan error TTS agar tombol hentikan tidak stale', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const tts = createAvailableTts();
    let playbackListener: TtsPlaybackListener | undefined;
    tts.subscribePlayback = jest.fn((listener) => {
      playbackListener = listener;
      return jest.fn();
    });
    tts.speak = jest.fn(async (text) => {
      playbackListener?.({ status: 'speaking', text });
      return OK_RESULT;
    });

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        scanner={scanner.port}
        tts={tts}
      />,
    );
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));
    await emitStableAndAdd(scanner, user, 'A');

    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeEnabled();

    await act(() => {
      playbackListener?.({ status: 'idle', reason: 'done', text: 'A' });
    });
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeDisabled();
    expect(screen.getByText('Pembacaan selesai.')).toBeOnTheScreen();

    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    await act(() => {
      playbackListener?.({
        status: 'error',
        message: 'Suara gagal dibacakan.',
        text: 'A',
      });
    });
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeDisabled();
    expect(screen.getByText('Suara gagal dibacakan.')).toBeOnTheScreen();
  });

  it('menghentikan repeat lama yang selesai setelah kata berubah', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const tts = createAvailableTts();
    const deferredRepeat = createDeferred<typeof OK_RESULT>();
    tts.speak = jest
      .fn()
      .mockResolvedValueOnce(OK_RESULT)
      .mockReturnValueOnce(deferredRepeat.promise);

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        scanner={scanner.port}
        tts={tts}
      />,
    );
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));
    await emitStableAndAdd(scanner, user, 'A');
    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    await user.press(screen.getByRole('button', { name: 'Ulangi' }));

    await emitStableAndAdd(scanner, user, 'B', 1);
    (tts.stop as jest.Mock).mockClear();

    await act(() => {
      deferredRepeat.resolve(OK_RESULT);
    });

    await waitFor(() => expect(tts.stop).toHaveBeenCalledTimes(1));
    expect(screen.queryByText('Mengulangi kata.')).toBeNull();
  });

  it('tidak membiarkan request lama menghentikan pembacaan kata terbaru', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const tts = createAvailableTts();
    const oldSpeech = createDeferred<typeof OK_RESULT>();
    const newSpeech = createDeferred<typeof OK_RESULT>();
    tts.speak = jest
      .fn()
      .mockReturnValueOnce(oldSpeech.promise)
      .mockReturnValueOnce(newSpeech.promise);

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        scanner={scanner.port}
        tts={tts}
      />,
    );
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));

    await emitStableAndAdd(scanner, user, 'A');
    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    await emitStableAndAdd(scanner, user, 'B', 1);
    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    (tts.stop as jest.Mock).mockClear();

    await act(() => {
      oldSpeech.resolve(OK_RESULT);
    });
    expect(tts.stop).not.toHaveBeenCalled();

    await act(() => {
      newSpeech.resolve(OK_RESULT);
    });
    await waitFor(() => {
      expect(screen.getByText('Membacakan kata.')).toBeOnTheScreen();
    });
    expect(tts.stop).not.toHaveBeenCalled();
    expect(tts.speak).toHaveBeenNthCalledWith(
      2,
      'AB',
      expect.objectContaining({ rate: 1, volume: 1 }),
    );
  });

  it('mengabaikan lifecycle playback kata lama setelah draft berubah', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const tts = createAvailableTts();
    let playbackListener: TtsPlaybackListener | undefined;
    tts.subscribePlayback = jest.fn((listener) => {
      playbackListener = listener;
      return jest.fn();
    });

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        scanner={scanner.port}
        tts={tts}
      />,
    );
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));
    await emitStableAndAdd(scanner, user, 'A');

    await act(() => {
      playbackListener?.({ status: 'speaking', text: 'A' });
    });
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeEnabled();

    await emitStableAndAdd(scanner, user, 'B', 1);
    await act(() => {
      playbackListener?.({
        status: 'error',
        message: 'Error kata lama.',
        text: 'A',
      });
      playbackListener?.({ status: 'speaking', text: 'AB' });
      playbackListener?.({ status: 'idle', reason: 'done', text: 'A' });
    });

    expect(screen.queryByText('Error kata lama.')).toBeNull();
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeEnabled();

    await act(() => {
      playbackListener?.({ status: 'idle', reason: 'done', text: 'AB' });
    });
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeDisabled();
  });

  it('mengirim sesi aktif, jeda, dan senter ke preview kamera', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const renderScannerPreview = jest.fn(() => null);

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        renderScannerPreview={renderScannerPreview}
        scanner={scanner.port}
        tts={createAvailableTts()}
      />,
    );

    await waitFor(() => {
      expect(renderScannerPreview).toHaveBeenLastCalledWith({
        active: true,
        resetToken: 0,
        sessionId: 'test-session',
        torchEnabled: false,
      });
    });

    await user.press(screen.getByRole('button', { name: 'Jeda' }));
    await waitFor(() => {
      expect(renderScannerPreview).toHaveBeenLastCalledWith({
        active: false,
        resetToken: 0,
        sessionId: 'test-session',
        torchEnabled: false,
      });
    });

    await user.press(screen.getByRole('button', { name: 'Lanjut' }));
    await fireEvent(
      screen.getByRole('switch', { name: 'Senter' }),
      'valueChange',
      true,
    );
    await waitFor(() => {
      expect(renderScannerPreview).toHaveBeenLastCalledWith({
        active: true,
        resetToken: 0,
        sessionId: 'test-session',
        torchEnabled: true,
      });
    });

    await act(() => {
      scanner.emit(createEvent('stable', { text: 'A', confidence: 0.97 }));
    });
    await user.press(screen.getByRole('button', { name: 'Pindai ulang sel' }));
    await waitFor(() => {
      expect(renderScannerPreview).toHaveBeenLastCalledWith({
        active: true,
        resetToken: 1,
        sessionId: 'test-session',
        torchEnabled: true,
      });
    });
  });

  it('menonaktifkan preview bila izin kamera dicabut saat kembali ke aplikasi', async () => {
    let appStateListener: ((state: AppStateStatus) => void) | undefined;
    const originalAddEventListener = AppState.addEventListener;
    AppState.addEventListener = jest.fn((_type, listener) => {
      appStateListener = listener;
      return { remove: jest.fn() };
    });
    const scanner = createScannerHarness();
    const cameraPermission = createGrantedPermission();
    const getStatus = cameraPermission.getStatus as jest.MockedFunction<
      CameraPermissionPort['getStatus']
    >;
    getStatus.mockResolvedValueOnce({
      status: 'granted',
      canAskAgain: false,
    });
    const renderScannerPreview = jest.fn(() => null);
    const tts = createAvailableTts();

    const view = await render(
      <LiveScreenView
        cameraPermission={cameraPermission}
        renderScannerPreview={renderScannerPreview}
        scanner={scanner.port}
        tts={tts}
      />,
    );
    await waitFor(() => {
      expect(renderScannerPreview).toHaveBeenLastCalledWith(
        expect.objectContaining({ active: true }),
      );
    });

    getStatus.mockResolvedValueOnce({
      status: 'denied',
      canAskAgain: false,
    });
    await act(async () => {
      appStateListener?.('active');
    });

    await waitFor(() => {
      expect(renderScannerPreview).toHaveBeenLastCalledWith(
        expect.objectContaining({ active: false }),
      );
      expect(
        screen.getByRole('header', { name: 'Kamera belum siap' }),
      ).toBeOnTheScreen();
      expect(tts.invalidateAvailabilityCache).toHaveBeenCalledTimes(1);
    });
    await view.unmount();
    AppState.addEventListener = originalAddEventListener;
  });

  it('mempertahankan refresh izin dan TTS terbaru ketika request lama selesai belakangan', async () => {
    type Permission = Awaited<
      ReturnType<CameraPermissionPort['getStatus']>
    >;
    type Availability = Awaited<ReturnType<TtsPort['getAvailability']>>;
    let appStateListener: ((state: AppStateStatus) => void) | undefined;
    const originalAddEventListener = AppState.addEventListener;
    AppState.addEventListener = jest.fn((_type, listener) => {
      appStateListener = listener;
      return { remove: jest.fn() };
    });
    const scanner = createScannerHarness();
    const cameraPermission = createGrantedPermission();
    const tts = createAvailableTts();
    const getStatus = cameraPermission.getStatus as jest.MockedFunction<
      CameraPermissionPort['getStatus']
    >;
    const getAvailability = tts.getAvailability as jest.MockedFunction<
      TtsPort['getAvailability']
    >;
    const firstPermissionRequest = createDeferred<Permission>();
    const latestPermissionRequest = createDeferred<Permission>();
    const firstTtsRequest = createDeferred<Availability>();
    const latestTtsRequest = createDeferred<Availability>();
    const renderScannerPreview = jest.fn(() => null);

    try {
      const view = await render(
        <LiveScreenView
          cameraPermission={cameraPermission}
          renderScannerPreview={renderScannerPreview}
          scanner={scanner.port}
          tts={tts}
        />,
      );
      await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));
      const user = userEvent.setup();
      await emitStableAndAdd(scanner, user, 'A');
      await waitFor(() => {
        expect(
          screen.getByRole('button', { name: 'Bacakan' }),
        ).toBeEnabled();
      });

      getStatus
        .mockReturnValueOnce(firstPermissionRequest.promise)
        .mockReturnValueOnce(latestPermissionRequest.promise);
      getAvailability
        .mockReturnValueOnce(firstTtsRequest.promise)
        .mockReturnValueOnce(latestTtsRequest.promise);

      await act(() => {
        appStateListener?.('active');
        appStateListener?.('active');
      });
      expect(getStatus).toHaveBeenCalledTimes(3);
      expect(getAvailability).toHaveBeenCalledTimes(3);

      await act(async () => {
        latestPermissionRequest.resolve({
          status: 'granted',
          canAskAgain: false,
        });
        latestTtsRequest.resolve({
          status: 'available',
          localVoiceAvailable: true,
          message: 'Suara lokal tersedia.',
        });
        await Promise.all([
          latestPermissionRequest.promise,
          latestTtsRequest.promise,
        ]);
      });

      await act(async () => {
        firstPermissionRequest.resolve({
          status: 'denied',
          canAskAgain: false,
        });
        firstTtsRequest.resolve({
          status: 'unavailable',
          localVoiceAvailable: false,
          message: 'Suara lokal tidak tersedia.',
        });
        await Promise.all([
          firstPermissionRequest.promise,
          firstTtsRequest.promise,
        ]);
      });

      await waitFor(() => {
        expect(renderScannerPreview).toHaveBeenLastCalledWith(
          expect.objectContaining({ active: true }),
        );
        expect(
          screen.queryByRole('header', { name: 'Kamera belum siap' }),
        ).toBeNull();
        expect(
          screen.getByRole('button', { name: 'Bacakan' }),
        ).toBeEnabled();
      });
      expect(tts.invalidateAvailabilityCache).toHaveBeenCalledTimes(2);
      await view.unmount();
    } finally {
      AppState.addEventListener = originalAddEventListener;
    }
  });

  it('mengabaikan hasil inisialisasi izin yang kalah oleh refresh AppState', async () => {
    type Permission = Awaited<
      ReturnType<CameraPermissionPort['getStatus']>
    >;
    let appStateListener: ((state: AppStateStatus) => void) | undefined;
    const originalAddEventListener = AppState.addEventListener;
    AppState.addEventListener = jest.fn((_type, listener) => {
      appStateListener = listener;
      return { remove: jest.fn() };
    });
    const scanner = createScannerHarness();
    const cameraPermission = createGrantedPermission();
    const tts = createAvailableTts();
    const getStatus = cameraPermission.getStatus as jest.MockedFunction<
      CameraPermissionPort['getStatus']
    >;
    const initialPermissionRequest = createDeferred<Permission>();
    getStatus
      .mockReturnValueOnce(initialPermissionRequest.promise)
      .mockResolvedValueOnce({ status: 'denied', canAskAgain: false });

    try {
      const view = await render(
        <LiveScreenView
          cameraPermission={cameraPermission}
          scanner={scanner.port}
          tts={tts}
        />,
      );
      await waitFor(() => expect(getStatus).toHaveBeenCalledTimes(1));

      await act(async () => {
        appStateListener?.('active');
      });
      await waitFor(() => expect(getStatus).toHaveBeenCalledTimes(2));

      await act(async () => {
        initialPermissionRequest.resolve({
          status: 'granted',
          canAskAgain: true,
        });
        await initialPermissionRequest.promise;
      });

      await waitFor(() => {
        expect(scanner.port.start).not.toHaveBeenCalled();
        expect(
          screen.getByRole('header', { name: 'Pemindai belum tersedia' }),
        ).toBeOnTheScreen();
        expect(
          screen.getByRole('button', { name: 'Buka pengaturan aplikasi' }),
        ).toBeOnTheScreen();
      });
      await view.unmount();
    } finally {
      AppState.addEventListener = originalAddEventListener;
    }
  });

  it('menghentikan TTS aktif ketika aplikasi masuk background', async () => {
    let appStateListener: ((state: AppStateStatus) => void) | undefined;
    const originalAddEventListener = AppState.addEventListener;
    AppState.addEventListener = jest.fn((_type, listener) => {
      appStateListener = listener;
      return { remove: jest.fn() };
    });
    const user = userEvent.setup();

    try {
      const { scanner, tts } = await renderReadyLive();
      await emitStableAndAdd(scanner, user, 'A');
      await user.press(screen.getByRole('button', { name: 'Bacakan' }));

      await act(() => {
        appStateListener?.('background');
      });

      expect(tts.stop).toHaveBeenCalledTimes(1);
      expect(
        screen.getByRole('button', { name: 'Hentikan suara' }),
      ).toBeDisabled();
      expect(screen.getByLabelText('Kata tersusun: A')).toBeOnTheScreen();
      expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();
    } finally {
      AppState.addEventListener = originalAddEventListener;
    }
  });

  it('menonaktifkan kontrol senter sampai kamera melaporkan kapabilitas', async () => {
    const scanner = createScannerHarness();
    let capabilityListener: ScannerCapabilitiesListener | undefined;
    scanner.port.getCapabilities = jest.fn(() => ({ torchAvailable: false }));
    scanner.port.subscribeCapabilities = jest.fn((listener) => {
      capabilityListener = listener;
      return jest.fn();
    });

    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        scanner={scanner.port}
        tts={createAvailableTts()}
      />,
    );
    await waitFor(() => {
      expect(
        screen.getByRole('switch', { name: 'Senter tidak tersedia' }),
      ).toBeDisabled();
    });
    expect(scanner.port.setTorch).not.toHaveBeenCalled();

    await act(() => {
      capabilityListener?.({ torchAvailable: true });
    });
    const torchSwitch = screen.getByRole('switch', { name: 'Senter' });
    expect(torchSwitch).toBeEnabled();
    await fireEvent(torchSwitch, 'valueChange', true);
    expect(scanner.port.setTorch).toHaveBeenCalledWith(true);

    await act(() => {
      capabilityListener?.({ torchAvailable: false });
    });
    expect(
      screen.getByRole('switch', { name: 'Senter tidak tersedia' }),
    ).toBeDisabled();
    expect(
      screen.getByRole('switch', { name: 'Senter tidak tersedia' }),
    ).not.toBeChecked();
  });

  it('menghentikan suara aktif saat kata dihapus', async () => {
    const user = userEvent.setup();
    const { scanner, tts } = await renderReadyLive();

    await emitStableAndAdd(scanner, user, 'A');
    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    await user.press(screen.getByRole('button', { name: 'Hapus kata' }));

    await waitFor(() => expect(tts.stop).toHaveBeenCalledTimes(1));
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeDisabled();
    expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();
    expect(screen.getByLabelText('Kata masih kosong')).toBeOnTheScreen();
  });

  it('menghentikan suara lama saat kata berubah dan layar ditutup', async () => {
    const user = userEvent.setup();
    const { scanner, tts, unmount } = await renderReadyLive();

    await emitStableAndAdd(scanner, user, 'A');
    await user.press(screen.getByRole('button', { name: 'Bacakan' }));

    await emitStableAndAdd(scanner, user, 'B', 1);
    await waitFor(() => expect(tts.stop).toHaveBeenCalledTimes(1));
    expect(
      screen.getByRole('button', { name: 'Hentikan suara' }),
    ).toBeDisabled();

    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    await waitFor(() => {
      expect(tts.speak).toHaveBeenCalledTimes(2);
    });
    await unmount();

    await waitFor(() => {
      expect(tts.stop).toHaveBeenCalledTimes(2);
    });
  });

  it('mengakhiri sesi satu kali tanpa menunggu native TTS yang lambat', async () => {
    const user = userEvent.setup();
    const onWordDraftChange = jest.fn();
    const { scanner, tts } = await renderReadyLive({ onWordDraftChange });
    const routerModule = jest.requireMock<{
      __router: { back: jest.Mock };
    }>('expo-router');
    tts.stop = jest.fn(() => new Promise<never>(() => undefined));

    await emitStableAndAdd(scanner, user, 'A');
    onWordDraftChange.mockClear();
    await user.press(screen.getByRole('button', { name: 'Bacakan' }));
    const finishButton = screen.getByRole('button', {
      name: 'Akhiri pemindaian',
    });
    await fireEvent.press(finishButton);
    await fireEvent.press(finishButton);

    expect(routerModule.__router.back).toHaveBeenCalledTimes(1);
    expect(tts.stop).toHaveBeenCalledTimes(1);
    expect(onWordDraftChange).toHaveBeenCalledWith(
      INITIAL_WORD_DRAFT_STATE,
    );
    await waitFor(() => expect(scanner.port.stop).toHaveBeenCalledTimes(1));
  });

  it('menghentikan scanner bila start selesai setelah layar ditutup', async () => {
    const scanner = createScannerHarness();
    const cameraPermission = createGrantedPermission();
    const tts = createAvailableTts();
    const deferredStart = createDeferred<typeof OK_RESULT>();
    const startMock = scanner.port.start as jest.MockedFunction<
      ScannerPort['start']
    >;
    startMock.mockReturnValueOnce(deferredStart.promise);

    const view = await render(
      <LiveScreenView
        cameraPermission={cameraPermission}
        scanner={scanner.port}
        tts={tts}
      />,
    );
    await waitFor(() => {
      expect(startMock).toHaveBeenCalledTimes(1);
    });

    await view.unmount();
    await act(() => {
      deferredStart.resolve(OK_RESULT);
    });

    await waitFor(() => {
      expect(scanner.port.stop).toHaveBeenCalledTimes(1);
    });
  });

  it('mencegah dua retry memulai scanner secara bersamaan', async () => {
    const scanner = createScannerHarness();
    const cameraPermission = createGrantedPermission();
    const tts = createAvailableTts();
    const deferredRetry = createDeferred<typeof OK_RESULT>();
    const startMock = scanner.port.start as jest.MockedFunction<
      ScannerPort['start']
    >;
    startMock
      .mockResolvedValueOnce({
        ok: false,
        code: 'failed',
        message: 'Pemindai gagal dimulai.',
      })
      .mockReturnValueOnce(deferredRetry.promise);

    await render(
      <LiveScreenView
        cameraPermission={cameraPermission}
        scanner={scanner.port}
        tts={tts}
      />,
    );
    const retryButton = await screen.findByRole('button', {
      name: 'Coba lagi',
    });

    await fireEvent.press(retryButton);
    await fireEvent.press(retryButton);

    await waitFor(() => {
      expect(startMock).toHaveBeenCalledTimes(2);
      expect(screen.getByRole('button', { name: 'Coba lagi' })).toBeDisabled();
    });

    await act(() => {
      deferredRetry.resolve(OK_RESULT);
    });
    await waitFor(() => {
      expect(
        screen.getByRole('summary', { name: /Mencari satu sel Braille/i }),
      ).toBeOnTheScreen();
    });
    expect(startMock).toHaveBeenCalledTimes(2);
  });

  it('membersihkan error, pause, senter, dan hasil lama setelah retry sukses', async () => {
    const user = userEvent.setup();
    const { scanner } = await renderReadyLive();

    await emitStableAndAdd(scanner, user, 'A');
    await fireEvent(
      screen.getByRole('switch', { name: 'Senter' }),
      'valueChange',
      true,
    );
    await act(() => {
      scanner.emit(createEvent('error', { scanEpoch: 1 }));
    });

    expect(screen.queryByRole('button', { name: 'Jeda' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Lanjut' })).toBeNull();
    expect(screen.queryByRole('switch', { name: 'Senter' })).toBeNull();
    expect(
      screen.getByRole('header', { name: 'Pemindaian bermasalah' }),
    ).toBeOnTheScreen();

    await user.press(screen.getByRole('button', { name: 'Coba lagi' }));

    await waitFor(() => {
      expect(scanner.port.start).toHaveBeenCalledTimes(2);
      expect(
        screen.getByRole('summary', { name: /Mencari satu sel Braille/i }),
      ).toBeOnTheScreen();
    });
    expect(screen.getByRole('button', { name: 'Jeda' })).toBeOnTheScreen();
    expect(screen.getByRole('switch', { name: 'Senter' })).not.toBeChecked();
    expect(screen.getByText('Belum ada huruf stabil.')).toBeOnTheScreen();
    expect(screen.getByLabelText('Kata tersusun: A')).toBeOnTheScreen();
    expect(
      screen.queryByRole('header', { name: 'Pemindaian bermasalah' }),
    ).toBeNull();
  });

  it('tidak merender ulang preview untuk event kandidat yang setara', async () => {
    const scanner = createScannerHarness();
    const renderScannerPreview = jest.fn(() => null);
    await render(
      <LiveScreenView
        cameraPermission={createGrantedPermission()}
        renderScannerPreview={renderScannerPreview}
        scanner={scanner.port}
        tts={createAvailableTts()}
      />,
    );
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));

    await act(() => {
      scanner.emit(
        createEvent('candidate', { text: 'A', confidence: 0.91 }),
      );
    });
    const renderCount = renderScannerPreview.mock.calls.length;

    await act(() => {
      scanner.emit(
        createEvent('candidate', {
          text: 'B',
          confidence: 0.99,
          timestampMs: 1_786_128_000_200,
        }),
      );
    });

    expect(renderScannerPreview).toHaveBeenCalledTimes(renderCount);
  });

  it('kembali ke Beranda ketika recovery dibuka lewat deep link', async () => {
    const user = userEvent.setup();
    const { scanner } = await renderReadyLive();
    const router = jest.requireMock<{
      __router: {
        back: jest.Mock;
        canGoBack: jest.Mock;
        replace: jest.Mock;
      };
    }>('expo-router').__router;
    router.canGoBack.mockReturnValue(false);
    await act(() => scanner.emit(createEvent('error')));

    await user.press(screen.getByRole('button', { name: 'Kembali' }));

    expect(router.back).not.toHaveBeenCalled();
    expect(router.replace).toHaveBeenCalledWith('/');
  });

  it('izin kamera yang ditolak permanen hanya menawarkan Pengaturan', async () => {
    const scanner = createScannerHarness();
    const cameraPermission: CameraPermissionPort = {
      getStatus: jest.fn(async () => ({
        status: 'denied' as const,
        canAskAgain: false,
      })),
      request: jest.fn(async () => ({
        status: 'denied' as const,
        canAskAgain: false,
      })),
      openAppSettings: jest.fn(async () => OK_RESULT),
    };
    await render(
      <LiveScreenView
        cameraPermission={cameraPermission}
        scanner={scanner.port}
        tts={createAvailableTts()}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByText(
          'Izin kamera ditolak. Berikan izin melalui pengaturan aplikasi Android.',
        ),
      ).toBeOnTheScreen();
    });

    expect(
      screen.getByRole('header', { name: 'Kamera belum siap' }),
    ).toBeOnTheScreen();
    expect(screen.queryByText('Belum ada huruf stabil.')).toBeNull();
    expect(screen.queryByLabelText(/^Huruf Braille terdeteksi:/)).toBeNull();
    expect(screen.queryByText(/Keyakinan \d+ persen/)).toBeNull();
    expect(screen.queryByRole('button', { name: 'Bacakan' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Jeda' })).toBeNull();
    expect(screen.queryByRole('switch', { name: 'Senter' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Coba lagi' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Izinkan kamera' })).toBeNull();
    expect(
      screen.getByRole('button', { name: 'Buka pengaturan aplikasi' }),
    ).toBeOnTheScreen();
    expect(screen.getByRole('button', { name: 'Kembali' })).toBeOnTheScreen();
    expect(cameraPermission.request).not.toHaveBeenCalled();
  });

  it('menawarkan izin kamera ketika dialog Android masih dapat diminta', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const cameraPermission: CameraPermissionPort = {
      getStatus: jest.fn(async () => ({
        status: 'undetermined' as const,
        canAskAgain: true,
      })),
      request: jest.fn(async () => ({
        status: 'granted' as const,
        canAskAgain: false,
      })),
      openAppSettings: jest.fn(async () => OK_RESULT),
    };
    await render(
      <LiveScreenView
        cameraPermission={cameraPermission}
        scanner={scanner.port}
        tts={createAvailableTts()}
      />,
    );

    const permissionButton = await screen.findByRole('button', {
      name: 'Izinkan kamera',
    });
    expect(
      screen.queryByRole('button', { name: 'Buka pengaturan aplikasi' }),
    ).toBeNull();
    await user.press(permissionButton);

    expect(cameraPermission.request).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(scanner.port.start).toHaveBeenCalledTimes(1));
  });

  it('menyediakan pemulihan ketika adapter tersedia tetapi gagal dimulai', async () => {
    const user = userEvent.setup();
    const scanner = createScannerHarness();
    const cameraPermission = createGrantedPermission();
    const tts = createAvailableTts();
    const startMock = scanner.port.start as jest.MockedFunction<
      ScannerPort['start']
    >;

    startMock
      .mockResolvedValueOnce({
        ok: false,
        code: 'failed',
        message: 'Pemindai gagal dimulai.',
      })
      .mockResolvedValueOnce(OK_RESULT);

    await render(
      <LiveScreenView
        cameraPermission={cameraPermission}
        scanner={scanner.port}
        tts={tts}
      />,
    );

    await waitFor(() => {
      expect(startMock).toHaveBeenCalledTimes(1);
      expect(
        screen.getByRole('header', { name: 'Pemindaian belum siap' }),
      ).toBeOnTheScreen();
    });
    expect(
      screen.getByText('Pemindaian tidak dapat dimulai.'),
    ).toBeOnTheScreen();

    await user.press(screen.getByRole('button', { name: 'Coba lagi' }));

    await waitFor(() => {
      expect(startMock).toHaveBeenCalledTimes(2);
      expect(
        screen.queryByRole('header', { name: 'Pemindaian belum siap' }),
      ).toBeNull();
    });
  });
});
