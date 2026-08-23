import type { ReadinessPort } from "@/services/ports";
import { BRAILLE_MODEL_RELEASE_VERIFIED } from '@/ml/braille-model-contract';
import { scannerPort } from "@/services/single-cell-scanner-port";
import { ttsPort } from "@/services/expo-speech-tts-port";
import { cameraPermissionPort } from "@/services/vision-camera-permission-port";
import type { ReadinessItem, ReadinessSnapshot } from "@/types";

const MODEL_MESSAGE =
  "Classifier satu sel belum tersedia pada build ini.";
function getCameraReadiness(
  status: Awaited<ReturnType<typeof cameraPermissionPort.getStatus>>['status'],
): ReadinessItem {
  switch (status) {
    case 'granted':
      return {
        component: 'camera',
        status: 'ready',
        message: 'Izin kamera tersedia.',
      };
    case 'undetermined':
      return {
        component: 'camera',
        status: 'needs_attention',
        message: 'Izin kamera belum diberikan.',
      };
    case 'denied':
      return {
        component: 'camera',
        status: 'needs_attention',
        message: 'Izin kamera ditolak. Aktifkan melalui Pengaturan aplikasi.',
      };
    case 'blocked':
      return {
        component: 'camera',
        status: 'needs_attention',
        message: 'Akses kamera dibatasi oleh pengaturan perangkat.',
      };
    case 'unavailable':
      return {
        component: 'camera',
        status: 'unavailable',
        message: 'Status izin kamera tidak dapat diperiksa pada perangkat ini.',
      };
  }
}

class RuntimeReadinessPort implements ReadinessPort {
  async check(): Promise<ReadinessSnapshot> {
    const [scanner, camera, tts] = await Promise.all([
      scannerPort.getAvailability(),
      cameraPermissionPort.getStatus(),
      ttsPort.getAvailability(),
    ]);

    const modelItem: ReadinessItem = scanner.available
      ? BRAILLE_MODEL_RELEASE_VERIFIED
        ? {
            component: 'model',
            status: 'ready',
            message: 'Classifier satu sel terpasang dan terverifikasi.',
          }
        : {
            component: 'model',
            status: 'needs_attention',
            message:
              'Classifier satu sel terpasang, tetapi verifikasi akurasi belum lulus.',
          }
      : {
          component: "model",
          status: "unavailable",
          message: MODEL_MESSAGE,
        };

    const cameraItem = getCameraReadiness(camera.status);

    const ttsItem: ReadinessItem =
      tts.status === "available" && tts.localVoiceAvailable
        ? {
            component: "tts",
            status: "ready",
            message: "Suara lokal tersedia.",
          }
        : {
            component: "tts",
            status: "unavailable",
            message: tts.message,
          };

    const status =
      modelItem.status === "ready" &&
      cameraItem.status === "ready" &&
      ttsItem.status === "ready"
        ? "ready"
        : "not_ready";

    return {
      status,
      model: modelItem,
      camera: cameraItem,
      tts: ttsItem,
      checkedAtMs: Date.now(),
    };
  }
}

export const readinessPort: ReadinessPort = new RuntimeReadinessPort();
