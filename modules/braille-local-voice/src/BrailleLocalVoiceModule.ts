import { NativeModule, requireOptionalNativeModule } from 'expo';

import type { BrailleLocalVoice } from './BrailleLocalVoice.types';

export declare class BrailleLocalVoiceNativeModule extends NativeModule<{}> {
  getLocalVoicesAsync(): Promise<BrailleLocalVoice[]>;
}

export default requireOptionalNativeModule<BrailleLocalVoiceNativeModule>(
  'BrailleLocalVoice',
);
