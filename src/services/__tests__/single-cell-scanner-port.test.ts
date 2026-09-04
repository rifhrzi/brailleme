import { SingleCellScannerPort } from '@/services/single-cell-scanner-port';
import type { LiveScanEvent, ScannerStartResult } from '@/types';

type ScannerPayload = Omit<LiveScanEvent, 'schemaVersion' | 'sessionId'>;

function createPayload(
  overrides: Partial<ScannerPayload> = {},
): ScannerPayload {
  return {
    scanEpoch: 0,
    timestampMs: 1_786_214_400_000,
    status: 'searching',
    boxes: [],
    decoderVersion: 'single-cell-v1',
    ...overrides,
  };
}

function requireSession(result: ScannerStartResult): string {
  if (!result.ok) {
    throw new Error(`Scanner gagal dimulai: ${result.message}`);
  }

  expect(result.sessionId.trim()).not.toHaveLength(0);
  return result.sessionId;
}

describe('SingleCellScannerPort', () => {
  it('tersedia dan hanya menerbitkan event ketika sesi sedang berjalan', async () => {
    const port = new SingleCellScannerPort();
    const listener = jest.fn<void, [LiveScanEvent]>();
    port.subscribe(listener);

    await expect(port.getAvailability()).resolves.toEqual({ available: true });
    expect(port.publish('not-active', createPayload())).toBe(false);
    expect(listener).not.toHaveBeenCalled();

    const sessionId = requireSession(await port.start());
    const payload = createPayload({
      timestampMs: 1_786_214_400_100,
      status: 'candidate',
      text: 'A',
      confidence: 0.94,
      boxes: [{ x: 0.25, y: 0.25, width: 0.5, height: 0.5 }],
      modelVersion: 'test-model-version',
    });

    expect(port.publish(sessionId, payload)).toBe(true);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenLastCalledWith({
      schemaVersion: 2,
      sessionId,
      ...payload,
    });

    await expect(port.stop()).resolves.toEqual({ ok: true });
    expect(port.publish(sessionId, createPayload())).toBe(false);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('menahan event saat jeda lalu menerbitkan kembali pada sesi yang sama', async () => {
    const port = new SingleCellScannerPort();
    const listener = jest.fn<void, [LiveScanEvent]>();
    port.subscribe(listener);
    const sessionId = requireSession(await port.start());

    await expect(port.pause()).resolves.toEqual({ ok: true });
    expect(
      port.publish(
        sessionId,
        createPayload({ status: 'candidate', text: 'B' }),
      ),
    ).toBe(false);
    expect(listener).not.toHaveBeenCalled();

    await expect(port.resume()).resolves.toEqual({ ok: true });
    expect(
      port.publish(
        sessionId,
        createPayload({ status: 'stable', text: 'B', confidence: 0.96 }),
      ),
    ).toBe(true);
    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({
        schemaVersion: 2,
        sessionId,
        status: 'stable',
        text: 'B',
      }),
    );
  });

  it('mempertahankan sesi pada start idempoten dan membuat sesi baru setelah stop', async () => {
    const port = new SingleCellScannerPort();
    const listener = jest.fn<void, [LiveScanEvent]>();
    port.subscribe(listener);

    const firstSession = requireSession(await port.start());
    const repeatedSession = requireSession(await port.start());
    expect(repeatedSession).toBe(firstSession);

    expect(
      port.publish(
        firstSession,
        createPayload({ status: 'stable', text: 'C', confidence: 0.97 }),
      ),
    ).toBe(true);
    await expect(port.stop()).resolves.toEqual({ ok: true });

    const secondSession = requireSession(await port.start());
    expect(secondSession).not.toBe(firstSession);
    expect(
      port.publish(
        firstSession,
        createPayload({ status: 'stable', text: 'HASIL LAMA', confidence: 0.99 }),
      ),
    ).toBe(false);
    expect(
      port.publish(
        secondSession,
        createPayload({ status: 'stable', text: 'D', confidence: 0.98 }),
      ),
    ).toBe(true);

    expect(listener.mock.calls.map(([event]) => event.sessionId)).toEqual([
      firstSession,
      secondSession,
    ]);
  });

  it('mengisolasi langganan dan unsubscribe bersifat aman untuk dipanggil ulang', async () => {
    const port = new SingleCellScannerPort();
    const firstListener = jest.fn<void, [LiveScanEvent]>();
    const secondListener = jest.fn<void, [LiveScanEvent]>();
    const unsubscribeFirst = port.subscribe(firstListener);
    const unsubscribeSecond = port.subscribe(secondListener);
    const sessionId = requireSession(await port.start());

    expect(port.publish(sessionId, createPayload())).toBe(true);
    expect(firstListener).toHaveBeenCalledTimes(1);
    expect(secondListener).toHaveBeenCalledTimes(1);

    unsubscribeFirst();
    unsubscribeFirst();
    expect(
      port.publish(
        sessionId,
        createPayload({ timestampMs: 1_786_214_400_200 }),
      ),
    ).toBe(true);
    expect(firstListener).toHaveBeenCalledTimes(1);
    expect(secondListener).toHaveBeenCalledTimes(2);

    unsubscribeSecond();
    expect(
      port.publish(
        sessionId,
        createPayload({ timestampMs: 1_786_214_400_300 }),
      ),
    ).toBe(true);
    expect(secondListener).toHaveBeenCalledTimes(2);
  });

  it('tetap memberi tahu listener sehat ketika listener lain melempar error', async () => {
    const port = new SingleCellScannerPort();
    const healthyListener = jest.fn<void, [LiveScanEvent]>();
    port.subscribe(() => {
      throw new Error('Consumer rusak');
    });
    port.subscribe(healthyListener);
    const sessionId = requireSession(await port.start());

    expect(() => port.publish(sessionId, createPayload())).not.toThrow();
    expect(healthyListener).toHaveBeenCalledTimes(1);
  });

  it('menolak payload runtime yang tidak memenuhi kontrak event', async () => {
    const port = new SingleCellScannerPort();
    const listener = jest.fn<void, [LiveScanEvent]>();
    port.subscribe(listener);
    const sessionId = requireSession(await port.start());

    const invalidPayloads = [
      createPayload({ timestampMs: Number.NaN }),
      createPayload({ scanEpoch: -1 }),
      createPayload({ confidence: 1.01 }),
      createPayload({ status: 'stable', text: 'AB', confidence: 0.99 }),
      createPayload({
        boxes: [{ x: 0.75, y: 0.25, width: 0.5, height: 0.5 }],
      }),
    ];

    for (const payload of invalidPayloads) {
      expect(port.publish(sessionId, payload)).toBe(false);
    }
    expect(listener).not.toHaveBeenCalled();
  });

  it('menolak kontrol yang memerlukan sesi dan menjaga stop tetap idempoten', async () => {
    const port = new SingleCellScannerPort();

    await expect(port.pause()).resolves.toMatchObject({
      ok: false,
      code: 'failed',
    });
    await expect(port.resume()).resolves.toMatchObject({
      ok: false,
      code: 'failed',
    });
    await expect(port.stop()).resolves.toEqual({ ok: true });
    await expect(port.setTorch(true)).resolves.toMatchObject({
      ok: false,
      code: 'failed',
    });

    requireSession(await port.start());
    await expect(port.setTorch(true)).resolves.toMatchObject({
      ok: false,
      code: 'failed',
    });
    port.setTorchAvailable(true);
    expect(port.getCapabilities()).toEqual({ torchAvailable: true });
    await expect(port.setTorch(true)).resolves.toEqual({ ok: true });
    await expect(port.setTorch(false)).resolves.toEqual({ ok: true });

    await expect(port.stop()).resolves.toEqual({ ok: true });
    expect(port.getCapabilities()).toEqual({ torchAvailable: false });
  });

  it('menerbitkan perubahan kapabilitas torch tanpa notifikasi duplikat', () => {
    const port = new SingleCellScannerPort();
    const listener = jest.fn();
    const unsubscribe = port.subscribeCapabilities(listener);

    expect(port.getCapabilities()).toEqual({ torchAvailable: false });
    port.setTorchAvailable(true);
    port.setTorchAvailable(true);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenLastCalledWith({ torchAvailable: true });

    unsubscribe();
    port.setTorchAvailable(false);
    expect(listener).toHaveBeenCalledTimes(1);
  });
});
