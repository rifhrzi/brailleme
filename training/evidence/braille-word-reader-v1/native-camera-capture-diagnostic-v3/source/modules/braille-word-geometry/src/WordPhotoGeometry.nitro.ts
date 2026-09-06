import type { HybridObject } from 'react-native-nitro-modules';
import type { Photo } from 'react-native-vision-camera';

export interface PhotoGeometry {
  width: number;
  height: number;
  sensorToBuffer: number[];
  rotationDegrees: number;
  isMirrored: boolean;
  cropRect: number[];
}

/** Copies capture metadata while the original Photo remains alive. No pixels are retained. */
export interface WordPhotoGeometry extends HybridObject<{ android: 'kotlin' }> {
  getGeometry(photo: Photo): PhotoGeometry;
}
