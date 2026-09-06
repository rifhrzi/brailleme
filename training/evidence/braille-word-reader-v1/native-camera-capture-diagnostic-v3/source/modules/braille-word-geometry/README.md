# Word photo geometry

Android-only local Expo module with one Nitro helper. Expo autolinking discovers this directory automatically. It uses the existing VisionCamera `Photo`; it never retains the photo, reads image pixels, or changes a dependency.

`getGeometry(photo)` copies the CameraX image's sensor-to-buffer matrix, raw dimensions, clockwise rotation, mirror flag and crop rectangle while the photo is alive. `src/ml/word-camera-geometry.ts` maps the four guide corners from the preview's actual sensor coordinate system into the upright image returned by `Photo.toImageAsync`. It rejects missing or singular matrices, incompatible dimensions, invalid crop rectangles, clipped guides and nonrectangular mappings. The frozen guide is drawn from the same resulting photo ROI as inference.

VisionCamera `CameraRef.convertViewPointToCameraPoint` receives React Native view coordinates in dp, applies the native density internally, and returns opaque sensor coordinates. They must not be treated as normalized coordinates or multiplied by density again. Photo orientation and mirroring are applied once in the coordinate mapping, in the same order as the native bitmap conversion. The bitmap itself is already upright and is not rotated again.

The supported implementation is pinned to VisionCamera 5.2.2, CameraX 1.7.0-alpha02, Nitro Modules 0.36.5 and Nitro Image 0.15.1. CameraX corrects the photo matrix for a HAL-rotated JPEG before handing it to VisionCamera. Preview stabilization is explicitly disabled during word capture. Review these assumptions when upgrading the camera stack.

Generate bindings from this directory:

```powershell
npx.cmd --yes nitrogen@0.36.5
```

The 14 generated files are checked in; generation is not required during app installation or APK builds. The module was scaffolded with `create-expo-module --local`; its unused Apple/web templates were removed. There is no additional npm dependency.

Verification before APK build:

```powershell
npm.cmd test -- --runTestsByPath src/ml/__tests__/word-camera-geometry.test.ts src/features/live/__tests__/word-photo-camera.test.tsx
npx.cmd expo-modules-autolinking resolve --platform android --json
```

The unit tests cover both directions of a 4:3/16:9 mismatch, sensor origin/scale, every quarter rotation with mirroring, dp semantics, stale captures, unavailable preview and invalid geometry. These tests and source inspection do not replace native verification: compile all APK ABIs and use the camera imagefile fixture to compare stationary guide-corner landmarks before/after actual `capturePhoto` and `toImageAsync`. Record negotiated dimensions and orientation. Repeat the corner/edge check on the physical phone; a passing emulator result is separate evidence.

An internal build with `appEnvironment: preview` emits one `BRAILLE_WORD_CAPTURE_GEOMETRY` record to Android Logcat after a successful geometry mapping. It contains capture number, preview size in dp, negotiated preview resolution when available, guide sensor corners, raw photo dimensions/matrix/crop/orientation, upright dimensions and the mapped ROI. Read it with `adb logcat -s ReactNativeJS:I` while testing the same APK. Production builds emit no such record. No pixels, recognized text, file paths, native image objects or persistent settings are logged. Metadata alone does not establish landmark agreement; compare independent landmarks in actual preview and frozen-photo screenshots as well.

Upstream references: [CameraX crop and field of view](https://developer.android.com/media/camera/camerax/configuration#crop-rect), [CameraX output transforms](https://developer.android.com/media/camera/camerax/transform-output). The version-matched source audit is retained under `.braille-training-work/word-scan-v1-20260905/native-crop-audit/`.
