import "expo-sqlite/localStorage/install";

import { useSyncExternalStore } from "react";

import {
  ALPHA_BRAILLE_PROFILE,
  ALPHA_PREFERENCES_SCHEMA_VERSION,
  ALPHA_SPEECH_MODE,
  type AlphaPreferences,
  type AlphaPreferencesPatch,
} from "@/types";
import {
  ALPHA_PREFERENCES_DEFAULTS,
  areAlphaPreferencesEqual,
  normalizeAlphaPreferences,
} from "@/storage/alpha-preferences-schema";

export const ALPHA_PREFERENCES_STORAGE_KEY =
  "brailleme.alpha-preferences.v1" as const;

type Listener = () => void;

interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const listeners = new Set<Listener>();
const memoryFallback = new Map<string, string>();
let persistentStorageFailed = false;
let currentSnapshot: AlphaPreferences | undefined;

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

function readRawPreferences(): string | null {
  const installedStorage = getInstalledStorage();

  if (installedStorage) {
    try {
      return installedStorage.getItem(ALPHA_PREFERENCES_STORAGE_KEY);
    } catch {
      persistentStorageFailed = true;
    }
  }

  return memoryFallback.get(ALPHA_PREFERENCES_STORAGE_KEY) ?? null;
}

function writeRawPreferences(value: string): void {
  // Selalu simpan bayangan memori agar SSR/Jest tetap reaktif tanpa Storage API.
  memoryFallback.set(ALPHA_PREFERENCES_STORAGE_KEY, value);

  const installedStorage = getInstalledStorage();
  if (!installedStorage) {
    return;
  }

  try {
    installedStorage.setItem(ALPHA_PREFERENCES_STORAGE_KEY, value);
  } catch {
    persistentStorageFailed = true;
  }
}

function loadSnapshot(): AlphaPreferences {
  const raw = readRawPreferences();
  if (raw === null) {
    return ALPHA_PREFERENCES_DEFAULTS;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    parsed = undefined;
  }

  const normalized = normalizeAlphaPreferences(parsed);
  const normalizedJson = JSON.stringify(normalized);

  // Menyelesaikan migrasi/repair sekali tanpa mempersistensikan data lain.
  if (raw !== normalizedJson) {
    writeRawPreferences(normalizedJson);
  }

  return normalized;
}

function getSnapshot(): AlphaPreferences {
  currentSnapshot ??= loadSnapshot();
  return currentSnapshot;
}

function emitChange(): void {
  for (const listener of listeners) {
    listener();
  }
}

function storeSnapshot(next: AlphaPreferences): void {
  const previous = getSnapshot();
  currentSnapshot = next;
  writeRawPreferences(JSON.stringify(next));

  if (!areAlphaPreferencesEqual(previous, next)) {
    emitChange();
  }

}

export function getAlphaPreferences(): AlphaPreferences {
  return getSnapshot();
}

export function subscribeAlphaPreferences(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function updateAlphaPreferences(
  update:
    | AlphaPreferencesPatch
    | ((current: AlphaPreferences) => AlphaPreferencesPatch),
): void {
  const current = getSnapshot();
  const patch = typeof update === "function" ? update(current) : update;

  const next = normalizeAlphaPreferences({
    ...current,
    ...patch,
    schemaVersion: ALPHA_PREFERENCES_SCHEMA_VERSION,
    profile: ALPHA_BRAILLE_PROFILE,
    speechMode: ALPHA_SPEECH_MODE,
  });

  if (areAlphaPreferencesEqual(current, next)) {
    return;
  }

  storeSnapshot(next);
}

export function resetAlphaPreferences(): void {
  storeSnapshot(ALPHA_PREFERENCES_DEFAULTS);
}

/** Snapshot reaktif; aksi update/reset tetap fungsi terpisah agar referensinya stabil. */
export function useAlphaPreferences(): AlphaPreferences {
  return useSyncExternalStore(
    subscribeAlphaPreferences,
    getSnapshot,
    () => ALPHA_PREFERENCES_DEFAULTS,
  );
}
