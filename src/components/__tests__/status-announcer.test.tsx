import { act, render } from '@testing-library/react-native';
import { AccessibilityInfo } from 'react-native';

import { StatusAnnouncer } from '../status-announcer';

describe('StatusAnnouncer', () => {
  const announceSpy = jest.spyOn(
    AccessibilityInfo,
    'announceForAccessibility',
  );

  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-01-01T00:00:00.000Z'));
    announceSpy.mockClear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  afterAll(() => {
    announceSpy.mockRestore();
  });

  it('keeps candidate state silent and cancels pending guidance', async () => {
    const { rerender } = await render(
      <StatusAnnouncer status="quality_issue" guidance="Tambah cahaya" />,
    );

    expect(announceSpy).toHaveBeenCalledWith('Tambah cahaya');

    await rerender(
      <StatusAnnouncer status="quality_issue" guidance="Dekatkan kamera" />,
    );
    await rerender(
      <StatusAnnouncer status="candidate" guidance="Teks kandidat" />,
    );

    await act(() => {
      jest.advanceTimersByTime(3_000);
    });

    expect(announceSpy).toHaveBeenCalledTimes(1);
  });

  it('announces only changed guidance at intervals of at least three seconds', async () => {
    const { rerender } = await render(
      <StatusAnnouncer
        status="quality_issue"
        guidance="  Tambah   cahaya  "
        throttleMs={10}
      />,
    );

    expect(announceSpy).toHaveBeenNthCalledWith(1, 'Tambah cahaya');

    await rerender(
      <StatusAnnouncer
        status="quality_issue"
        guidance="Dekatkan kamera"
        throttleMs={10}
      />,
    );

    await act(() => {
      jest.advanceTimersByTime(2_999);
    });
    expect(announceSpy).toHaveBeenCalledTimes(1);

    await act(() => {
      jest.advanceTimersByTime(1);
    });
    expect(announceSpy).toHaveBeenNthCalledWith(2, 'Dekatkan kamera');

    await rerender(
      <StatusAnnouncer
        status="quality_issue"
        guidance="Dekatkan kamera"
        throttleMs={10}
      />,
    );
    await act(() => {
      jest.advanceTimersByTime(3_000);
    });

    expect(announceSpy).toHaveBeenCalledTimes(2);
  });

  it('announces the status title together with actionable guidance', async () => {
    await render(
      <StatusAnnouncer
        status="error"
        title="Pemindaian mengalami masalah"
        guidance="Coba lagi"
      />,
    );

    expect(announceSpy).toHaveBeenCalledWith(
      'Pemindaian mengalami masalah. Coba lagi',
    );
  });

  it('announces a stable key once without reading result text', async () => {
    const { rerender } = await render(
      <StatusAnnouncer
        status="stable"
        stableKey="result-1"
        guidance="teks hasil rahasia"
      />,
    );

    expect(announceSpy).toHaveBeenCalledTimes(1);
    expect(announceSpy).toHaveBeenLastCalledWith(
      'Huruf terdeteksi. Periksa hasil di layar.',
    );

    await rerender(<StatusAnnouncer status="candidate" />);
    await rerender(
      <StatusAnnouncer
        status="stable"
        stableKey="result-1"
        guidance="teks hasil berubah"
      />,
    );

    expect(announceSpy).toHaveBeenCalledTimes(1);

    await rerender(<StatusAnnouncer status="stable" stableKey="result-2" />);

    expect(announceSpy).toHaveBeenCalledTimes(2);
    expect(announceSpy).toHaveBeenLastCalledWith(
      'Huruf terdeteksi. Periksa hasil di layar.',
    );
  });

  it('uses a safe custom announcement without reading the result text', async () => {
    await render(
      <StatusAnnouncer
        status="stable"
        stableAnnouncement="Huruf satu sel terdeteksi. Periksa hasil."
        stableKey="private-result"
        title="Huruf A"
      />,
    );

    expect(announceSpy).toHaveBeenCalledWith(
      'Huruf satu sel terdeteksi. Periksa hasil.',
    );
  });
});
