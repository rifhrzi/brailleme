import "expo-sqlite/localStorage/install";

import { useSyncExternalStore } from "react";

import {
  APP_BRAILLE_PROFILE,
  APP_PREFERENCES_SCHEMA_VERSION,
  APP_SPEECH_MODE,
  type AppPreferences,
  type AppPreferencesPatch,
} from "@/types";
import {
  APP_PREFERENCES_DEFAULTS,
  areAppPreferencesEqual,
  normalizeAppPreferences,
} from "@/storage/app-preferences-schema";

export const APP_PREFERENCES_STORAGE_KEY = "brailleme.preferences.v1" as const;
const LEGACY_PREFERENCES_STORAGE_KEY = "brailleme.alpha-preferences.v1" as const;

type Listener = () => void;

interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem?(key: string): void;
}

const listeners = new Set<Listener>();
const memoryFallback = new Map<string, string>();
let persistentStorageFailed = false;
let currentSnapshot: AppPreferences | undefined;

function getInstalledStorage(): KeyValueStorage | undefined {
  if (persistentStorageFailed) {
    return undefined;
  }

  try {
    const candidate = (
      globalThis as typeof globalThis & { localStorage?: KeyValueStorage }
    ).localStorage;

    if (
      candidate &&
      typeof candidate.getItem === "function" &&
      typeof candidate.setItem === "function"
    ) {
      return candidate;
    }
  } catch {
    persistentStorageFailed = true;
  }

  return undefined;
}

function readRawPreferences(key: string): string | null {
  const installedStorage = getInstalledStorage();

  if (installedStorage) {
    try {
      return installedStorage.getItem(key);
    } catch {
      persistentStorageFailed = true;
    }
  }

  return memoryFallback.get(key) ?? null;
}

function writeRawPreferences(key: string, value: string): boolean {
  // Selalu simpan bayangan memori agar SSR/Jest tetap reaktif tanpa Storage API.
  memoryFallback.set(key, value);

  const installedStorage = getInstalledStorage();
  if (!installedStorage) {
    return true;
  }

  try {
    installedStorage.setItem(key, value);
    return true;
  } catch {
    persistentStorageFailed = true;
    return false;
  }
}

function removeRawPreferences(key: string): void {
  memoryFallback.delete(key);

  const installedStorage = getInstalledStorage();
  if (!installedStorage || typeof installedStorage.removeItem !== "function") {
    return;
  }

  try {
    installedStorage.removeItem(key);
  } catch {
    persistentStorageFailed = true;
  }
}

function parseRawPreferences(raw: string): {
  readonly malformed: boolean;
  readonly value: unknown;
} {
  try {
    return { malformed: false, value: JSON.parse(raw) as unknown };
  } catch {
    return { malformed: true, value: undefined };
  }
}

function loadSnapshot(): AppPreferences {
  const currentRaw = readRawPreferences(APP_PREFERENCES_STORAGE_KEY);
  const legacyRaw = readRawPreferences(LEGACY_PREFERENCES_STORAGE_KEY);
  const raw = currentRaw ?? legacyRaw;

  if (raw === null) {
    return APP_PREFERENCES_DEFAULTS;
  }

  let parsedResult = parseRawPreferences(raw);

  // Pulihkan dari key lama hanya jika JSON key baru benar-benar rusak.
  // Schema masa depan yang valid tetap ditangani fail-closed oleh normalizer.
  if (
    currentRaw !== null &&
    parsedResult.malformed &&
    legacyRaw !== null
  ) {
    parsedResult = parseRawPreferences(legacyRaw);
  }

  const normalized = normalizeAppPreferences(parsedResult.value);
  const normalizedJson = JSON.stringify(normalized);

  // Menyelesaikan migrasi/repair sekali tanpa mempersistensikan data lain.
  const storedAtCurrentKey =
    currentRaw !== null && currentRaw === normalizedJson;
  const migrationPersisted =
    storedAtCurrentKey ||
    writeRawPreferences(APP_PREFERENCES_STORAGE_KEY, normalizedJson);

  if (legacyRaw !== null && migrationPersisted) {
    removeRawPreferences(LEGACY_PREFERENCES_STORAGE_KEY);
  }

  return normalized;
}

function getSnapshot(): AppPreferences {
  currentSnapshot ??= loadSnapshot();
  return currentSnapshot;
}

function emitChange(): void {
  for (const listener of listeners) {
    listener();
  }
}

function storeSnapshot(next: AppPreferences): void {
  const previous = getSnapshot();
  currentSnapshot = next;
  writeRawPreferences(APP_PREFERENCES_STORAGE_KEY, JSON.stringify(next));

  if (!areAppPreferencesEqual(previous, next)) {
    emitChange();
  }
}

export function getAppPreferences(): AppPreferences {
  return getSnapshot();
}

export function subscribeAppPreferences(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function updateAppPreferences(
  update:
    | AppPreferencesPatch
    | ((current: AppPreferences) => AppPreferencesPatch),
): void {
  const current = getSnapshot();
  const patch = typeof update === "function" ? update(current) : update;

  const next = normalizeAppPreferences({
    ...current,
    ...patch,
    schemaVersion: APP_PREFERENCES_SCHEMA_VERSION,
    profile: APP_BRAILLE_PROFILE,
    speechMode: APP_SPEECH_MODE,
  });

  if (areAppPreferencesEqual(current, next)) {
    return;
  }

  storeSnapshot(next);
}

export function resetAppPreferences(): void {
  storeSnapshot(APP_PREFERENCES_DEFAULTS);
}

/** Snapshot reaktif; aksi update/reset tetap fungsi terpisah agar referensinya stabil. */
export function useAppPreferences(): AppPreferences {
  return useSyncExternalStore(
    subscribeAppPreferences,
    getSnapshot,
    () => APP_PREFERENCES_DEFAULTS,
  );
}
