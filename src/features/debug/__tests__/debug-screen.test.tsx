import {
  render,
  screen,
  userEvent,
  waitFor,
} from '@testing-library/react-native';

import { DebugScreenView } from '@/features/debug/debug-screen';
import type {
  CameraPermissionPort,
  ReadinessPort,
  ScannerPort,
  TtsPort,
} from '@/services';
import { resetAlphaPreferences, updateAlphaPreferences } from '@/storage';

const OK_RESULT = { ok: true, sessionId: 'test-session' } as const;

function createPorts() {
  const readiness: ReadinessPort = {
    check: jest.fn(async () => ({
      status: 'not_ready' as const,
      model: {
        component: 'model' as const,
        status: 'unavailable' as const,
        message: 'Model test belum tersedia.',
      },
      camera: {
        component: 'camera' as const,
        status: 'needs_attention' as const,
        message: 'Izin kamera test diperlukan.',
      },
      tts: {
        component: 'tts' as const,
        status: 'ready' as const,
        message: 'TTS test siap.',
      },
      checkedAtMs: 1_786_128_000_000,
    })),
  };
  const scanner: ScannerPort = {
    getAvailability: jest.fn(async () => ({
      available: false as const,
      reason: 'native_scanner_not_installed' as const,
      message: 'Scanner test unavailable.',
    })),
    subscribe: jest.fn(() => jest.fn()),
    start: jest.fn(async () => OK_RESULT),
    pause: jest.fn(async () => OK_RESULT),
    resume: jest.fn(async () => OK_RESULT),
    stop: jest.fn(async () => OK_RESULT),
    setTorch: jest.fn(async () => OK_RESULT),
  };
  const cameraPermission: CameraPermissionPort = {
    getStatus: jest.fn(async () => ({
      status: 'denied' as const,
      canAskAgain: true,
    })),
    request: jest.fn(async () => ({
      status: 'denied' as const,
      canAskAgain: true,
    })),
    openAppSettings: jest.fn(async () => OK_RESULT),
  };
  const tts: TtsPort = {
    getAvailability: jest.fn(async () => ({
      status: 'available' as const,
      localVoiceAvailable: true,
      message: 'TTS test tersedia.',
    })),
    speak: jest.fn(async () => OK_RESULT),
    stop: jest.fn(async () => OK_RESULT),
    repeat: jest.fn(async () => OK_RESULT),
    openTtsSettings: jest.fn(async () => OK_RESULT),
  };

  return { cameraPermission, readiness, scanner, tts };
}

describe('Debug Menu', () => {
  beforeEach(() => {
    resetAlphaPreferences();
  });

  it('menolak akses langsung ketika Debug Menu nonaktif', async () => {
    const ports = createPorts();

    await render(<DebugScreenView {...ports} />);

    expect(screen.getByTestId('expo-router-redirect')).toHaveAccessibilityValue({
      text: '/settings',
    });
    expect(ports.readiness.check).not.toHaveBeenCalled();
    expect(ports.scanner.getAvailability).not.toHaveBeenCalled();
  });

  it('menampilkan informasi pengembangan hanya setelah diaktifkan', async () => {
    const user = userEvent.setup();
    const ports = createPorts();
    updateAlphaPreferences({ debugMenuEnabled: true });

    await render(<DebugScreenView {...ports} />);

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: 'Muat ulang diagnostik' }),
      ).toBeEnabled();
    });

    expect(
      screen.getByRole('text', {
        name: /Model pemindai.*Belum tersedia.*Model test belum tersedia/i,
      }),
    ).toBeOnTheScreen();
    expect(
      screen.getByRole('text', { name: /id-latin-6dot-alpha-v0/i }),
    ).toBeOnTheScreen();
    expect(
      screen.getByRole('text', {
        name: /ScannerPort.*unavailable: native_scanner_not_installed/i,
      }),
    ).toBeOnTheScreen();
    expect(
      screen.getByRole('text', {
        name: /CameraPermissionPort.*denied; canAskAgain=true/i,
      }),
    ).toBeOnTheScreen();
    expect(
      screen.getByRole('text', {
        name: /TtsPort.*available; localVoice=true/i,
      }),
    ).toBeOnTheScreen();
    expect(
      screen.getByText(/classifier satu sel A sampai Z/i),
    ).toBeOnTheScreen();

    await user.press(
      screen.getByRole('button', { name: 'Muat ulang diagnostik' }),
    );
    await waitFor(() => {
      expect(ports.readiness.check).toHaveBeenCalledTimes(2);
    });
  });
});
