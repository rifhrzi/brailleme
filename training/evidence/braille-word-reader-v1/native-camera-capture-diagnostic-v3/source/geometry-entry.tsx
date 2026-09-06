import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Button, PixelRatio, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { registerRootComponent } from 'expo';
import { Camera, useCameraDevice, useCameraPermission, usePhotoOutput, type CameraDevice, type CameraRef, type Constraint } from 'react-native-vision-camera';
import { NativeNitroImage, type Image } from 'react-native-nitro-image';
import { getWordPhotoGeometry } from './modules/braille-word-geometry/src';
import { mapSensorGuideToWordPhoto, snapshotWordGuideSensorPoints } from './src/ml/word-camera-geometry';

// Isolated synthetic camera diagnostic only. No model import or recognition path.
const GUIDE = { x: .3, y: .3, width: .4, height: .4 };
const FULL = { x: 0, y: 0, width: 1, height: 1 };
const CASES = [
  { name: 'photo4by3-view1by1', target: { width: 1920, height: 1440 }, aspect: 1 },
  { name: 'photo16by9-view1by1', target: { width: 1920, height: 1080 }, aspect: 1 },
  { name: 'photo4by3-view2by1', target: { width: 1920, height: 1440 }, aspect: 2 },
] as const;
const CONSTRAINTS: Constraint[] = [{ previewStabilizationMode: 'off' }];
const log = (kind: string, data: unknown) => console.info('WORD_GEOMETRY_DIAGNOSTIC ' + JSON.stringify({ kind, data }));

function CameraCase({ index, device }: { index: number; device: CameraDevice }) {
  const spec = CASES[index]!;
  const output = usePhotoOutput({ targetResolution: spec.target, quality: 1, qualityPrioritization: 'quality' });
  const outputs = useMemo(() => [output], [output]);
  const ref = useRef<CameraRef>(null);
  const window = useWindowDimensions();
  const width = Math.min(window.width - 24, (window.height - 180) * spec.aspect);
  const height = width / spec.aspect;
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const lock = useRef(false);
  const counter = useRef(0);
  const owned = useRef<Image[]>([]);
  const [frozen, setFrozen] = useState<Image | null>(null);
  const [roi, setRoi] = useState<typeof GUIDE | null>(null);
  const [status, setStatus] = useState('Waiting for preview');
  const release = () => { setFrozen(null); setRoi(null); owned.current.forEach(image => image.dispose()); owned.current = []; };
  useEffect(() => () => { owned.current.forEach(image => image.dispose()); owned.current = []; }, []);
  async function capture() {
    if (lock.current || !ready || !ref.current) return;
    lock.current = true; setBusy(true); setStatus('Capturing'); release();
    let photo: Awaited<ReturnType<typeof output.capturePhoto>> | undefined;
    const id = `${spec.name}-${Date.now()}-${++counter.current}`;
    try {
      const guidePoints = snapshotWordGuideSensorPoints(ref.current, GUIDE, width, height);
      const fullPoints = snapshotWordGuideSensorPoints(ref.current, FULL, width, height);
      const resolution = ref.current.preview?.previewOutput?.currentResolution;
      const preview = await ref.current.takeSnapshot();
      owned.current.push(preview);
      const previewPath = await preview.saveToTemporaryFileAsync('png');
      photo = await output.capturePhoto({ flashMode: 'off', enableDistortionCorrection: false, enableVirtualDeviceFusion: false }, {});
      const geometry = getWordPhotoGeometry(photo);
      // Encoded ImageProxy JPEG pixels with VisionCamera-written EXIF: compare
      // independently against the upright Nitro bitmap, without extra crop.
      const jpegPath = await photo.saveToTemporaryFileAsync();
      const source = await photo.toImageAsync();
      owned.current.push(source);
      photo.dispose(); photo = undefined;
      const photoPath = await source.saveToTemporaryFileAsync('png');
      // Save evidence even if the production helper correctly rejects the mapping.
      const record = { id, case: spec.name, target: spec.target, orientationSource: 'interface', window: { width: window.width, height: window.height }, density: PixelRatio.get(), preview: { widthDp: width, heightDp: height, width: preview.width, height: preview.height, resolution, path: previewPath }, guide: GUIDE, guidePoints, fullPoints, geometry, jpegPath, photo: { width: source.width, height: source.height, path: photoPath } };
      log('capture', record);
      const result = mapSensorGuideToWordPhoto(guidePoints, geometry, source.width, source.height);
      setRoi(result); setFrozen(source); setStatus('Captured ' + id);
      log('mapped', { id, roi: result });
    } catch (error) { setStatus(String(error)); log('error', { id, error: String(error) }); }
    finally { photo?.dispose(); lock.current = false; setBusy(false); }
  }
  const scale = frozen ? Math.max(width / frozen.width, height / frozen.height) : 1;
  const overlay = frozen && roi ? {
    left: roi.x * frozen.width * scale - (frozen.width * scale - width) / 2,
    top: roi.y * frozen.height * scale - (frozen.height * scale - height) / 2,
    width: roi.width * frozen.width * scale,
    height: roi.height * frozen.height * scale,
  } : { left: GUIDE.x * width, top: GUIDE.y * height, width: GUIDE.width * width, height: GUIDE.height * height };
  return <View style={styles.case}>
    <Text style={styles.text}>{spec.name} | {Math.round(width)}×{Math.round(height)} dp</Text>
    <View style={{ width, height, overflow: 'hidden', backgroundColor: 'black' }}>
      <Camera ref={ref} device={device} isActive outputs={outputs} constraints={CONSTRAINTS} orientationSource="interface" implementationMode="compatible" resizeMode="cover" style={StyleSheet.absoluteFill} onPreviewStarted={() => { setReady(true); setStatus('Ready'); log('ready', { case: spec.name, width, height }); }} onPreviewStopped={() => setReady(false)} onError={error => { setStatus(String(error)); log('camera-error', String(error)); }} />
      {frozen ? <NativeNitroImage image={frozen} resizeMode="cover" style={StyleSheet.absoluteFill} /> : null}
      <View pointerEvents="none" style={[styles.guide, overlay]} />
    </View>
    <View style={styles.row}><Button title="Capture" disabled={!ready || busy} onPress={capture} /><Button title="Preview" disabled={busy} onPress={() => { release(); setStatus('Ready'); }} /></View>
    <Text style={styles.status}>{status}</Text>
  </View>;
}
function App() {
  const permission = useCameraPermission();
  const device = useCameraDevice('back');
  const [index, setIndex] = useState(0);
  useEffect(() => { if (permission.canRequestPermission) void permission.requestPermission(); }, [permission.canRequestPermission, permission.requestPermission]);
  return <View style={styles.root}><Text style={styles.title}>GEOMETRY DIAGNOSTIC — NO RECOGNITION</Text><View style={styles.row}>{CASES.map((spec, i) => <Button key={spec.name} title={'Case ' + (i + 1)} onPress={() => setIndex(i)} />)}</View>{permission.hasPermission && device ? <CameraCase key={index} index={index} device={device} /> : <Text style={styles.text}>Waiting for camera device</Text>}</View>;
}
const styles = StyleSheet.create({ root: { flex: 1, paddingTop: 34, backgroundColor: '#111', alignItems: 'center' }, title: { color: '#fff', fontSize: 12, marginBottom: 6 }, text: { color: '#fff', fontSize: 12 }, case: { alignItems: 'center' }, row: { flexDirection: 'row', gap: 18, paddingVertical: 6 }, guide: { position: 'absolute', borderColor: '#00ff00', borderWidth: 2 }, status: { color: '#fff', fontSize: 10, maxWidth: 330 } });
registerRootComponent(App);
