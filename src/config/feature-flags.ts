/**
 * Auto-speak hanya merupakan flag internal pengembangan dan tidak diekspos ke
 * pengaturan pengguna. Nilai default dan runtime frontend alpha selalu mati.
 */
export const INTERNAL_AUTO_SPEAK_ENABLED = false as const;

export const INTERNAL_FEATURE_FLAGS = Object.freeze({
  autoSpeakEnabled: INTERNAL_AUTO_SPEAK_ENABLED,
});
