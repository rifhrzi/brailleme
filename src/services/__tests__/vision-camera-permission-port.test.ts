import { Linking } from 'react-native';

import {
  mapVisionCameraPermissionStatus,
  VisionCameraPermissionPort,
  type VisionCameraPermissionApi,
} from '@/services/vision-camera-permission-port';

describe('VisionCameraPermissionPort', () => {
  it.each([
    ['authorized', { status: 'granted', canAskAgain: false }],
    ['not-determined', { status: 'undetermined', canAskAgain: true }],
    ['denied', { status: 'denied', canAskAgain: false }],
    ['restricted', { status: 'blocked', canAskAgain: false }],
  ] as const)('memetakan status native %s', (nativeStatus, expected) => {
    expect(mapVisionCameraPermissionStatus(nativeStatus)).toEqual(expected);
  });

  it('gagal aman untuk status native yang tidak dikenal', () => {
    expect(mapVisionCameraPermissionStatus('future-status')).toEqual({
      status: 'unavailable',
      canAskAgain: false,
    });
  });

  it('membaca status izin native saat ini', async () => {
    const camera: VisionCameraPermissionApi = {
      cameraPermissionStatus: 'authorized',
      requestCameraPermission: jest.fn(async () => true),
    };
    const port = new VisionCameraPermissionPort(async () => camera);

    await expect(port.getStatus()).resolves.toEqual({
      status: 'granted',
      canAskAgain: false,
    });
  });

  it('meminta izin dan membaca ulang status setelah dialog OS selesai', async () => {
    let status: VisionCameraPermissionApi['cameraPermissionStatus'] =
      'not-determined';
    const requestCameraPermission = jest.fn(async () => {
      status = 'authorized';
      return true;
    });
    const camera: VisionCameraPermissionApi = {
      get cameraPermissionStatus() {
        return status;
      },
      requestCameraPermission,
    };
    const port = new VisionCameraPermissionPort(async () => camera);

    await expect(port.request()).resolves.toEqual({
      status: 'granted',
      canAskAgain: false,
    });
    expect(requestCameraPermission).toHaveBeenCalledTimes(1);
  });

  it('mempertahankan retry ketika Android masih melaporkan not-determined', async () => {
    const requestCameraPermission = jest.fn(async () => false);
    const camera: VisionCameraPermissionApi = {
      cameraPermissionStatus: 'not-determined',
      requestCameraPermission,
    };
    const port = new VisionCameraPermissionPort(async () => camera);

    await expect(port.request()).resolves.toEqual({
      status: 'undetermined',
      canAskAgain: true,
    });
    await expect(port.request()).resolves.toEqual({
      status: 'undetermined',
      canAskAgain: true,
    });
    expect(requestCameraPermission).toHaveBeenCalledTimes(2);
  });

  it('tidak membuka dialog lagi ketika izin tidak dapat diminta', async () => {
    const requestCameraPermission = jest.fn(async () => false);
    const camera: VisionCameraPermissionApi = {
      cameraPermissionStatus: 'denied',
      requestCameraPermission,
    };
    const port = new VisionCameraPermissionPort(async () => camera);

    await expect(port.request()).resolves.toEqual({
      status: 'denied',
      canAskAgain: false,
    });
    expect(requestCameraPermission).not.toHaveBeenCalled();
  });

  it('menggabungkan request bersamaan agar dialog OS hanya dibuka sekali', async () => {
    let status: VisionCameraPermissionApi['cameraPermissionStatus'] =
      'not-determined';
    let finishRequest: (() => void) | undefined;
    const nativeRequest = new Promise<void>((resolve) => {
      finishRequest = resolve;
    });
    const requestCameraPermission = jest.fn(async () => {
      await nativeRequest;
      status = 'authorized';
      return true;
    });
    const camera: VisionCameraPermissionApi = {
      get cameraPermissionStatus() {
        return status;
      },
      requestCameraPermission,
    };
    const port = new VisionCameraPermissionPort(async () => camera);

    const first = port.request();
    const second = port.request();
    expect(first).toBe(second);
    expect(requestCameraPermission).toHaveBeenCalledTimes(0);

    await Promise.resolve();
    expect(requestCameraPermission).toHaveBeenCalledTimes(1);
    finishRequest?.();

    await expect(Promise.all([first, second])).resolves.toEqual([
      { status: 'granted', canAskAgain: false },
      { status: 'granted', canAskAgain: false },
    ]);
    expect(requestCameraPermission).toHaveBeenCalledTimes(1);
  });

  it('melepas lock request setelah kegagalan native', async () => {
    let attempt = 0;
    let status: VisionCameraPermissionApi['cameraPermissionStatus'] =
      'not-determined';
    const requestCameraPermission = jest.fn(async () => {
      attempt += 1;
      if (attempt === 1) {
        throw new Error('dialog failed');
      }
      status = 'authorized';
      return true;
    });
    const camera: VisionCameraPermissionApi = {
      get cameraPermissionStatus() {
        return status;
      },
      requestCameraPermission,
    };
    const port = new VisionCameraPermissionPort(async () => camera);

    await expect(port.request()).resolves.toEqual({
      status: 'unavailable',
      canAskAgain: false,
    });
    await expect(port.request()).resolves.toEqual({
      status: 'granted',
      canAskAgain: false,
    });
    expect(requestCameraPermission).toHaveBeenCalledTimes(2);
  });

  it('gagal aman ketika modul native tidak dapat dimuat', async () => {
    const port = new VisionCameraPermissionPort(async () => {
      throw new Error('native module unavailable');
    });

    await expect(port.getStatus()).resolves.toEqual({
      status: 'unavailable',
      canAskAgain: false,
    });
    await expect(port.request()).resolves.toEqual({
      status: 'unavailable',
      canAskAgain: false,
    });
  });

  it('membuka pengaturan aplikasi', async () => {
    const openSettings = jest
      .spyOn(Linking, 'openSettings')
      .mockResolvedValueOnce(undefined);
    const port = new VisionCameraPermissionPort();

    await expect(port.openAppSettings()).resolves.toEqual({ ok: true });
    expect(openSettings).toHaveBeenCalledTimes(1);
  });

  it('mengembalikan kegagalan saat pengaturan aplikasi tidak dapat dibuka', async () => {
    jest
      .spyOn(Linking, 'openSettings')
      .mockRejectedValueOnce(new Error('settings unavailable'));
    const port = new VisionCameraPermissionPort();

    await expect(port.openAppSettings()).resolves.toEqual({
      ok: false,
      code: 'failed',
      message: 'Pengaturan aplikasi tidak dapat dibuka pada perangkat ini.',
    });
  });
});
