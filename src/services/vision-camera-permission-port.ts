import type { PermissionStatus } from 'react-native-vision-camera';
import { Linking } from 'react-native';

import type { CameraPermissionPort } from '@/services/ports';
import type { CameraPermissionSnapshot, PortResult } from '@/types';

const UNAVAILABLE_PERMISSION: CameraPermissionSnapshot = Object.freeze({
  status: 'unavailable',
  canAskAgain: false,
});

export type VisionCameraPermissionApi = {
  readonly cameraPermissionStatus: PermissionStatus;
  requestCameraPermission(): Promise<boolean>;
};

type VisionCameraLoader = () => Promise<VisionCameraPermissionApi>;

async function loadVisionCamera(): Promise<VisionCameraPermissionApi> {
  // Muat modul native hanya ketika status izin benar-benar dibutuhkan. Ini
  // menjaga startup rute non-kamera dari pekerjaan native yang tidak perlu.
  const { VisionCamera } = await import('react-native-vision-camera');
  return VisionCamera;
}

export function mapVisionCameraPermissionStatus(
  status: PermissionStatus | string,
): CameraPermissionSnapshot {
  switch (status) {
    case 'authorized':
      return { status: 'granted', canAskAgain: false };
    case 'not-determined':
      return { status: 'undetermined', canAskAgain: true };
    case 'denied':
      return { status: 'denied', canAskAgain: false };
    case 'restricted':
      return { status: 'blocked', canAskAgain: false };
    default:
      // Lindungi boundary JS dari nilai baru/tidak valid yang mungkin datang
      // dari versi native yang tidak cocok dengan bundle aplikasi.
      return UNAVAILABLE_PERMISSION;
  }
}

export class VisionCameraPermissionPort implements CameraPermissionPort {
  private requestInFlight: Promise<CameraPermissionSnapshot> | null = null;

  constructor(private readonly loadCamera: VisionCameraLoader = loadVisionCamera) {}

  async getStatus(): Promise<CameraPermissionSnapshot> {
    try {
      const camera = await this.loadCamera();
      return mapVisionCameraPermissionStatus(camera.cameraPermissionStatus);
    } catch {
      return UNAVAILABLE_PERMISSION;
    }
  }

  request(): Promise<CameraPermissionSnapshot> {
    if (this.requestInFlight) {
      return this.requestInFlight;
    }

    const request = this.requestOnce();
    this.requestInFlight = request;
    void request.finally(() => {
      if (this.requestInFlight === request) {
        this.requestInFlight = null;
      }
    });

    return request;
  }

  private async requestOnce(): Promise<CameraPermissionSnapshot> {
    try {
      const camera = await this.loadCamera();
      const current = mapVisionCameraPermissionStatus(
        camera.cameraPermissionStatus,
      );

      // VisionCamera hanya mengizinkan dialog OS ketika statusnya
      // `not-determined`. Panggilan berulang tetap idempoten.
      if (!current.canAskAgain) {
        return current;
      }

      await camera.requestCameraPermission();
      return mapVisionCameraPermissionStatus(camera.cameraPermissionStatus);
    } catch {
      return UNAVAILABLE_PERMISSION;
    }
  }

  async openAppSettings(): Promise<PortResult> {
    try {
      await Linking.openSettings();
      return { ok: true };
    } catch {
      return {
        ok: false,
        code: 'failed',
        message: 'Pengaturan aplikasi tidak dapat dibuka pada perangkat ini.',
      };
    }
  }
}

export const cameraPermissionPort: CameraPermissionPort =
  new VisionCameraPermissionPort();
