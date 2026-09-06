"""Read actual native records and independently locate colors in preview/photo PNGs."""
from pathlib import Path
import hashlib
import json
import argparse
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='camera-capture')
parser.add_argument('--offline', action='store_true')
args = parser.parse_args()
OUT = HERE / args.directory
ADB = Path('C:/Users/rifqi/AppData/Local/Android/Sdk/platform-tools/adb.exe')
PACKAGE = 'com.brailleme.geometrydiagnostic'
manifest = json.loads((OUT / 'feed-manifest.json').read_text())

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def adb(*args): return subprocess.check_output([str(ADB), '-s', 'emulator-5554', *args])
def centroid(image, color):
    # Compare pixels with the fixed input palette; geometry is never fitted to landmarks.
    data = np.asarray(image.convert('RGB'), dtype=np.float32)
    palette = np.array([item['rgb'] for item in manifest['markers']], dtype=np.float32)
    distances = np.stack([np.sum((data - p) ** 2, axis=2) for p in palette], axis=2)
    index = [m['id'] for m in manifest['markers']].index(color)
    mask = (np.argmin(distances, axis=2) == index) & (distances[..., index] < 85**2)
    ys, xs = np.where(mask)
    if len(xs) < 20: raise ValueError(f'Missing marker {color}: {len(xs)} pixels')
    # Coordinates describe pixel centers in edge coordinates, matching native view transforms.
    return np.array([float(xs.mean()) + .5, float(ys.mean()) + .5]), int(len(xs))

def sensor_to_upright(sensor, geometry):
    v = np.asarray(geometry['sensorToBuffer'], dtype=np.float64).reshape(3,3) @ np.array([*sensor, 1.])
    x, y = v[:2] / v[2]
    width, height = geometry['width'], geometry['height']
    if geometry['isMirrored']: x = width - x
    return np.array({0: [x,y], 90: [height-y,x], 180:[width-x,height-y],270:[y,width-x]}[geometry['rotationDegrees']])

raw = (OUT / 'logcat.txt').read_text(encoding='utf-8') if args.offline else adb('logcat', '-d', '-v', 'raw').decode('utf-8', errors='replace')
(OUT / 'logcat.txt').write_text(raw, encoding='utf-8')
events = []
for line in raw.splitlines():
    token = 'WORD_GEOMETRY_DIAGNOSTIC '
    if token in line:
        events.append(json.loads(line.split(token, 1)[1]))
(OUT / 'events.json').write_text(json.dumps(events, indent=2)+'\n')
mapped = {e['data']['id']: e['data']['roi'] for e in events if e['kind'] == 'mapped'}
records = []
for event in events:
    if event['kind'] != 'capture': continue
    record = event['data']; identifier = record['id']; local = OUT / identifier
    local.mkdir(exist_ok=True)
    for name in ['preview', 'photo']:
        path = local / (name + '.png')
        if not path.exists():
            path.write_bytes(adb('exec-out', 'run-as', PACKAGE, 'cat', record[name]['path']))
    preview = Image.open(local / 'preview.png').convert('RGB')
    photo = Image.open(local / 'photo.png').convert('RGB')
    jpeg_check = None
    if record.get('jpegPath'):
        path = local / 'imageproxy.jpg'
        if not path.exists(): path.write_bytes(adb('exec-out', 'run-as', PACKAGE, 'cat', record['jpegPath']))
        encoded = Image.open(path)
        upright = ImageOps.exif_transpose(encoded).convert('RGB')
        same_size = upright.size == photo.size
        difference = np.abs(np.asarray(upright, dtype=np.int16) - np.asarray(photo, dtype=np.int16)) if same_size else None
        jpeg_check = {'encoded_size': list(encoded.size), 'exif_orientation': encoded.getexif().get(274),
                      'upright_size': list(upright.size), 'same_dimensions_as_nitro': same_size,
                      'mean_absolute_rgb_difference': float(difference.mean()) if difference is not None else None,
                      'max_absolute_rgb_difference': int(difference.max()) if difference is not None else None,
                      'claim': 'JPEG plane bytes saved by Photo.saveToTemporaryFileAsync (EXIF metadata attached, no pixel crop/re-encode in installed source); independently decoded and EXIF-oriented with Pillow.'}
    full = np.array([[p['x'], p['y']] for p in record['fullPoints']], dtype=np.float64)
    # Bilinear interpolation of native four-corner sensor coordinates: no landmark fitting.
    def project(point):
        u, v = point / np.array(preview.size)
        sensor = full[0]*(1-u)*(1-v) + full[1]*u*(1-v) + full[2]*u*v + full[3]*(1-u)*v
        return sensor_to_upright(sensor, record['geometry'])
    checks = []
    comparison = photo.copy(); draw = ImageDraw.Draw(comparison)
    failures = []
    for marker in manifest['markers']:
        try:
            p, countp = centroid(preview, marker['id'])
            actual, countf = centroid(photo, marker['id'])
            predicted = project(p)
            error = float(np.linalg.norm(predicted-actual))
            checks.append({'id':marker['id'], 'preview_centroid':p.tolist(), 'photo_centroid':actual.tolist(),
                           'predicted_photo_centroid':predicted.tolist(), 'error_photo_pixels':error,
                           'preview_color_pixels':countp,'photo_color_pixels':countf,
                           'passed':error <= manifest['tolerance_photo_pixels']})
            x,y=predicted; draw.line((x-18,y,x+18,y),fill=(0,255,0),width=3); draw.line((x,y-18,x,y+18),fill=(0,255,0),width=3)
        except ValueError as e: failures.append(str(e))
    failure = '; '.join(failures) if failures else None
    roi = mapped.get(identifier)
    roi_check = None
    if roi:
        box=[roi['x']*photo.width,roi['y']*photo.height,(roi['x']+roi['width'])*photo.width,(roi['y']+roi['height'])*photo.height]
        guide = record['guide']
        projected = np.array([project(np.array([u*preview.width,v*preview.height])) for u,v in [
            (guide['x'],guide['y']), (guide['x']+guide['width'],guide['y']),
            (guide['x']+guide['width'],guide['y']+guide['height']), (guide['x'],guide['y']+guide['height'])]])
        expected = np.array([projected[:,0].min(),projected[:,1].min(),projected[:,0].max(),projected[:,1].max()])
        roi_check = {'expected_pixels_xyxy':expected.tolist(),'actual_pixels_xyxy':box,
                     'max_absolute_corner_error_pixels':float(np.abs(expected-np.array(box)).max())}
        for check in checks:
            px,py=np.array(check['preview_centroid'])/np.array(preview.size)
            fx,fy=check['photo_centroid']
            check['inside_preview_guide']=bool(guide['x']<=px<=guide['x']+guide['width'] and guide['y']<=py<=guide['y']+guide['height'])
            check['inside_photo_roi']=bool(box[0]<=fx<=box[2] and box[1]<=fy<=box[3])
            check['guide_membership_matches']=check['inside_preview_guide']==check['inside_photo_roi']
        draw.rectangle(box,outline=(0,255,0),width=3)
        photo.crop(tuple([int(np.floor(box[0])),int(np.floor(box[1])),int(np.ceil(box[2])),int(np.ceil(box[3]))])).save(local/'mapped-roi.png')
    comparison.save(local/'photo-review.png')
    report={'capture':record,'roi':roi,'roi_independent_corner_check':roi_check,'checks':checks,'failure':failure,'jpeg_vs_nitro':jpeg_check,
            'passed':failure is None and roi is not None and len(checks)==4 and all(c['passed'] for c in checks),
            'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in local.iterdir() if p.suffix in ('.png','.jpg')}}
    (local/'report.json').write_text(json.dumps(report,indent=2)+'\n'); records.append(report)
report={'schema_version':1,'scope':'isolated_camera_geometry_not_main_apk_or_recognition_evidence',
        'analyzer_sha256':sha(Path(__file__)),'feed_manifest_sha256':sha(OUT/'feed-manifest.json'),
        'capture_count':len(records),'passed_count':sum(r['passed'] for r in records),'records':records,
        'limits':['Synthetic emulator imagefile feed only; no physical-camera or main-app smoke claim.',
                  'Orientation source is interface for deterministic adb portrait/landscape changes.',
                  'Reported negotiated stream dimensions determine aspect coverage, not requested target dimensions.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['capture_count','passed_count']}))
for r in records: print(r['capture']['id'], r['passed'], r['failure'], [(c['id'],round(c['error_photo_pixels'],3)) for c in r['checks']])
