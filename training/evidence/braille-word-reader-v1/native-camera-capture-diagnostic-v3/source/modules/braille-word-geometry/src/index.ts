import { requireNativeModule } from 'expo';
import { NitroModules } from 'react-native-nitro-modules';
import type { Photo } from 'react-native-vision-camera';

import type { WordPhotoGeometry } from './WordPhotoGeometry.nitro';

let geometry: WordPhotoGeometry | undefined;

export function getWordPhotoGeometry(photo: Photo) {
  if (!geometry) {
    requireNativeModule<{ initialize(): void }>('BrailleWordGeometry').initialize();
    geometry = NitroModules.createHybridObject<WordPhotoGeometry>('WordPhotoGeometry');
  }
  return geometry.getGeometry(photo);
}
