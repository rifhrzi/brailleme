import { requireNativeModule } from 'expo';
import { NitroModules } from 'react-native-nitro-modules';
import type { Photo } from 'react-native-vision-camera';

import type { WordPhotoGeometry } from './WordPhotoGeometry.nitro';

let geometry: WordPhotoGeometry | undefined;

/** Initialize on the JS runtime, then pass the stateless HybridObject to worklets. */
export function createWordPixelProcessor(): WordPhotoGeometry {
  if (!geometry) {
    requireNativeModule<{ initialize(): void }>('BrailleWordGeometry').initialize();
    geometry = NitroModules.createHybridObject<WordPhotoGeometry>('WordPhotoGeometry');
  }
  return geometry;
}

export function getWordPhotoGeometry(photo: Photo) {
  return createWordPixelProcessor().getGeometry(photo);
}
