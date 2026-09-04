export function isBrailleModelRuntimeAllowed(
  isProductionBuild: boolean,
  releaseVerified: boolean,
): boolean {
  return !isProductionBuild || releaseVerified;
}

export function assertBrailleModelRuntimeAllowed(
  isProductionBuild: boolean,
  releaseVerified: boolean,
): void {
  if (!isBrailleModelRuntimeAllowed(isProductionBuild, releaseVerified)) {
    throw new Error(
      'Build produksi menolak classifier yang belum terverifikasi.',
    );
  }
}
