import { readinessPort } from '@/services/runtime-readiness-port';

jest.mock('@/services/single-cell-scanner-port', () => ({
  scannerPort: {
    getAvailability: jest.fn(async () => ({ available: true as const })),
  },
}));
jest.mock('@/services/vision-camera-permission-port', () => ({
  cameraPermissionPort: {
    getStatus: jest.fn(async () => ({
      status: 'granted' as const,
      canAskAgain: false,
    })),
  },
}));
jest.mock('@/services/expo-speech-tts-port', () => ({
  ttsPort: {
    getAvailability: jest.fn(async () => ({
      status: 'available' as const,
      localVoiceAvailable: true,
      message: 'Suara lokal tersedia.',
    })),
  },
}));

describe('RuntimeReadinessPort', () => {
  it('tidak menandai model siap selama manifest release belum terverifikasi', async () => {
    await expect(readinessPort.check()).resolves.toMatchObject({
      status: 'not_ready',
      model: {
        component: 'model',
        status: 'needs_attention',
        message: expect.stringContaining('verifikasi akurasi belum lulus'),
      },
      camera: { component: 'camera', status: 'ready' },
      tts: { component: 'tts', status: 'ready' },
    });
  });
});
