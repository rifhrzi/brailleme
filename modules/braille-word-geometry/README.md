# Word photo geometry

Android-only local Expo module with one stateless Nitro helper. Expo autolinking discovers this directory automatically. Geometry uses the existing VisionCamera `Photo`; the pixel method receives a caller-owned raw buffer. Neither method retains its source photo or buffer.

`createWordPixelProcessor()` initializes the HybridObject on the JS runtime; pass it into a background worklet. `resizePixels(buffer, width, height, pixelFormat, androidBitmap)` returns an owning float32 `ArrayBuffer` with 480×192×3 RGB values plus mean luminance and contrast. The pure Java core matches the TypeScript/Pillow 22-bit bilinear antialias weights and rounds each pass to uint8 before float conversion. Generic BGRA keeps normal BGRA order; `androidBitmap=true` corrects only the pinned Nitro Image BGRA metadata defect. Calls allocate independent output storage and never reuse mutable buffers between worklets.

The generated bindings use Nitro Modules 0.36.5 `ArrayBuffer.getBuffer(false)` only during the synchronous native call, and `ArrayBuffer.wrap` for the direct output buffer. Run `scripts/word-native-parity/pixel-resize-parity.cjs` with explicit javac, java, Python, new work directory, and new report path to compare the production Java core with actual TypeScript and independent Pillow. Its report includes RGB/channel-order, up/down/identity resize, invalid dimension/buffer, and ownership cases. Host JVM timings are not Android camera or Nitro bridge timings; verify those in the rebuilt APK.

For an Android ART check without installing an APK, compile `WordPixelResize.java` and `scripts/word-native-parity/WordPixelResizeCli.java` using JDK 17, convert every resulting class with Android SDK D8, then run `scripts/word-native-parity/pixel-resize-android.cjs` with explicit adb, `emulator-5554`, `classes.dex`, the host case directory, a new output directory, and a new report path. It runs the production core through `dalvikvm`, pulls every output, and compares float32 bytes and luminance statistics with the current TypeScript and retained Pillow references. This verifies the core on ART; the Nitro wrapper, camera callback, model inference and physical phone still require their own checks. `prepare-camera-fixture.py` creates a padded camera feed with every original source pixel retained and records its transformed word ROI.

`getGeometry(photo)` copies the CameraX image's sensor-to-buffer matrix, raw dimensions, clockwise rotation, mirror flag and crop rectangle while the photo is alive. `src/ml/word-camera-geometry.ts` maps the four guide corners from the preview's actual sensor coordinate system into the upright image returned by `Photo.toImageAsync`. It rejects missing or singular matrices, incompatible dimensions, invalid crop rectangles, clipped guides and nonrectangular mappings. The frozen guide is drawn from the same resulting photo ROI as inference.

VisionCamera `CameraRef.convertViewPointToCameraPoint` receives React Native view coordinates in dp, applies the native density internally, and returns opaque sensor coordinates. They must not be treated as normalized coordinates or multiplied by density again. Photo orientation and mirroring are applied once in the coordinate mapping, in the same order as the native bitmap conversion. The bitmap itself is already upright and is not rotated again.

The supported implementation is pinned to VisionCamera 5.2.2, CameraX 1.7.0-alpha02, Nitro Modules 0.36.5 and Nitro Image 0.15.1. CameraX corrects the photo matrix for a HAL-rotated JPEG before handing it to VisionCamera. Preview stabilization is explicitly disabled during word capture. Review these assumptions when upgrading the camera stack.

Generate bindings from this directory:

```powershell
npx.cmd --yes nitrogen@0.36.5
```

The 17 generated files are checked in; generation is not required during app installation or APK builds. The module was scaffolded with `create-expo-module --local`; its unused Apple/web templates were removed. There is no additional npm dependency.

Verification before APK build:

```powershell
npm.cmd test -- --runTestsByPath src/ml/__tests__/word-camera-geometry.test.ts src/features/live/__tests__/word-photo-camera.test.tsx
npx.cmd expo-modules-autolinking resolve --platform android --json
```

The unit tests cover both directions of a 4:3/16:9 mismatch, sensor origin/scale, every quarter rotation with mirroring, dp semantics, stale captures, unavailable preview and invalid geometry. These tests and source inspection do not replace native verification: compile all APK ABIs and use the camera imagefile fixture to compare stationary guide-corner landmarks before/after actual `capturePhoto` and `toImageAsync`. Record negotiated dimensions and orientation. Repeat the corner/edge check on the physical phone; a passing emulator result is separate evidence.

An internal build with `appEnvironment: preview` emits one `BRAILLE_WORD_CAPTURE_GEOMETRY` record to Android Logcat after a successful geometry mapping. It contains capture number, preview size in dp, negotiated preview resolution when available, guide sensor corners, raw photo dimensions/matrix/crop/orientation, upright dimensions and the mapped ROI. Read it with `adb logcat -s ReactNativeJS:I` while testing the same APK. Production builds emit no such record. No pixels, recognized text, file paths, native image objects or persistent settings are logged. Metadata alone does not establish landmark agreement; compare independent landmarks in actual preview and frozen-photo screenshots as well.

Upstream references: [CameraX crop and field of view](https://developer.android.com/media/camera/camerax/configuration#crop-rect), [CameraX output transforms](https://developer.android.com/media/camera/camerax/transform-output). The version-matched source audit is retained under `.braille-training-work/word-scan-v1-20260905/native-crop-audit/`.
