import { useCallback, useEffect, useRef } from 'react';
import { AccessibilityInfo, View } from 'react-native';

export type AnnouncerStatus =
  | 'searching'
  | 'quality_issue'
  | 'candidate'
  | 'stable'
  | 'unsupported'
  | 'error';

export type StatusAnnouncerProps = {
  status: AnnouncerStatus;
  title?: string | undefined;
  guidance?: string | undefined;
  stableAnnouncement?: string | undefined;
  stableKey?: string | undefined;
  throttleMs?: number | undefined;
};

const MINIMUM_GUIDANCE_INTERVAL_MS = 1_000;
const STABLE_WITHOUT_KEY = '__stable_without_key__';

function normalizeGuidance(guidance: string | undefined) {
  return guidance?.trim().replace(/\s+/g, ' ') ?? '';
}

function buildAnnouncement(
  title: string | undefined,
  guidance: string | undefined,
) {
  const normalizedTitle = normalizeGuidance(title);
  const normalizedGuidance = normalizeGuidance(guidance);

  if (!normalizedTitle) {
    return normalizedGuidance;
  }
  if (!normalizedGuidance || normalizedGuidance === normalizedTitle) {
    return normalizedTitle;
  }
  return `${normalizedTitle}. ${normalizedGuidance}`;
}

export function StatusAnnouncer({
  guidance,
  stableAnnouncement = 'Kandidat stabil. Periksa hasil di layar.',
  stableKey,
  status,
  throttleMs = MINIMUM_GUIDANCE_INTERVAL_MS,
  title,
}: StatusAnnouncerProps) {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingGuidanceRef = useRef<string | null>(null);
  const lastGuidanceRef = useRef<string | null>(null);
  const lastGuidanceAtRef = useRef(Number.NEGATIVE_INFINITY);
  const announcedStableKeysRef = useRef(new Set<string>());

  const cancelPendingGuidance = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    pendingGuidanceRef.current = null;
  }, []);

  useEffect(() => {
    if (status === 'candidate') {
      cancelPendingGuidance();
      lastGuidanceRef.current = null;
      return;
    }

    if (status === 'stable') {
      cancelPendingGuidance();
      lastGuidanceRef.current = null;
      const normalizedStableKey = stableKey?.trim() || STABLE_WITHOUT_KEY;

      if (!announcedStableKeysRef.current.has(normalizedStableKey)) {
        announcedStableKeysRef.current.add(normalizedStableKey);
        AccessibilityInfo.announceForAccessibility(
          normalizeGuidance(stableAnnouncement) ||
            'Kandidat stabil. Periksa hasil di layar.',
        );
      }

      return;
    }

    const normalizedGuidance = buildAnnouncement(title, guidance);

    if (!normalizedGuidance) {
      cancelPendingGuidance();
      lastGuidanceRef.current = null;
      return;
    }

    if (
      normalizedGuidance === lastGuidanceRef.current ||
      normalizedGuidance === pendingGuidanceRef.current
    ) {
      return;
    }

    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    pendingGuidanceRef.current = normalizedGuidance;

    const safeThrottleMs = Number.isFinite(throttleMs)
      ? Math.max(MINIMUM_GUIDANCE_INTERVAL_MS, throttleMs)
      : MINIMUM_GUIDANCE_INTERVAL_MS;
    const elapsedMs = performance.now() - lastGuidanceAtRef.current;
    const waitMs = Math.max(0, safeThrottleMs - elapsedMs);

    const announcePendingGuidance = () => {
      const pendingGuidance = pendingGuidanceRef.current;

      timerRef.current = null;
      pendingGuidanceRef.current = null;

      if (!pendingGuidance || pendingGuidance === lastGuidanceRef.current) {
        return;
      }

      AccessibilityInfo.announceForAccessibility(pendingGuidance);
      lastGuidanceRef.current = pendingGuidance;
      lastGuidanceAtRef.current = performance.now();
    };

    if (waitMs === 0) {
      announcePendingGuidance();
      return;
    }

    timerRef.current = setTimeout(announcePendingGuidance, waitMs);
  }, [
    cancelPendingGuidance,
    guidance,
    stableAnnouncement,
    stableKey,
    status,
    throttleMs,
    title,
  ]);

  useEffect(
    () => () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
      }
    },
    [],
  );

  return (
    <View
      accessibilityElementsHidden
      accessible={false}
      importantForAccessibility="no-hide-descendants"
    />
  );
}
