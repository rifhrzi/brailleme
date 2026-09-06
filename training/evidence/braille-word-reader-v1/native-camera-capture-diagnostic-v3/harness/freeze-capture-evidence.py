"""Freeze isolated camera evidence without modifying application or model sources."""
from pathlib import Path
import datetime, hashlib, json, re, shutil, zipfile
import numpy as np
from PIL import Image

STAGE=Path(__file__).resolve().parents[1]
ROOT=Path('C:/Users/rifqi/OneDrive/Documents/Github/brailleme')
E=STAGE/'evidence'
OUT=ROOT/'.braille-training-work/word-scan-v1-20260905/native-camera-capture-diagnostic-v3'
OUT.mkdir(parents=True,exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def copy(path,relative):
    dest=OUT/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    return {'path':str(relative).replace('\\','/'),'sha256':sha(dest),'bytes':dest.stat().st_size}
sources=json.loads((E/'capture-source-copy.json').read_text())['sources']
source_rows=[]
for row in sources:
    source=STAGE/row['path']
    assert sha(source)==row['sha256']==sha(ROOT/row['path']), row['path']
    source_rows.append(copy(source,Path('source')/row['path']))
for relative in ['geometry-entry.tsx','package.json','package-lock.json','tsconfig.json','metro.config.js',
                 'app.config.ts','android/app/build.gradle','android/gradle.properties',
                 'android/app/src/main/AndroidManifest.xml','android/app/src/main/java/com/brailleme/app/MainApplication.kt',
                 'node_modules/react-native-vision-camera/android/src/main/java/com/margelo/nitro/camera/hybrids/instances/HybridPhoto.kt',
                 'node_modules/react-native-vision-camera/android/src/main/java/com/margelo/nitro/camera/extensions/ByteBuffer+readableBytes.kt']:
    source_rows.append(copy(STAGE/relative,Path('source')/relative))
for name in ['prepare-capture-feed.py','prepare-capture-feed-v3.py','analyze-captures.py','capture-control.py',
             'run-capture-cases.py','freeze-capture-evidence.py','limit-native-jobs.gradle']:
    source_rows.append(copy(E/name,Path('harness')/name))
evidence_rows=[]
for directory in ['camera-capture','camera-capture-v3','camera-source-audit']:
    for path in (E/directory).rglob('*'):
        if path.is_file():evidence_rows.append(copy(path,Path(directory)/path.relative_to(E/directory)))
for name in ['capture-app-build.log','capture-app-build-v2.log','capture-app-build-v3.log',
             'emulator-capture-command.json','emulator-capture-v3-command.json',
             'emulator-capture-stdout.log','emulator-capture-v2-stdout.log','emulator-capture-v3-stdout.log',
             'emulator-capture-v3-stderr.log','capture-v3-apksigner.txt','capture-v3-zipalign.txt',
             'capture-v3-badging.txt','capture-cases-v3.log','capture-cases-v3-retry.log',
             'initial-harness-logcat.txt','initial-camera-service.txt','final-v3-camera-service.txt']:
    evidence_rows.append(copy(E/name,Path('logs')/name))
bundle_path=STAGE/'android/app/build/generated/assets/react/release/index.android.bundle'
bundle_row=copy(bundle_path,Path('bundle/index.android.bundle'))
packager=STAGE/'android/app/build/intermediates/sourcemaps/react/release/index.android.bundle.packager.map'
source_map=json.loads(packager.read_text(encoding='utf-8'))
bundle_sources=[]
for name,content in zip(source_map['sources'],source_map['sourcesContent']):
    relative=name.lstrip('/')
    if relative in ['geometry-entry.tsx','src/ml/word-camera-geometry.ts','src/types/word-scan.ts','src/ml/word-model-contract.ts','modules/braille-word-geometry/src/index.ts']:
        digest=hashlib.sha256(content.encode('utf-8')).hexdigest()
        assert digest==sha(STAGE/relative)
        bundle_sources.append({'path':relative,'sha256':digest})
assert len(bundle_sources)==5
copy(packager,Path('bundle/index.android.bundle.packager.map'))
apk=STAGE/'android/app/build/outputs/apk/release/app-release.apk'
assert sha(apk)=='e2061133ad023e9a034fd7630ce1c4b515113a41fc7854a7012659e6db29e569', 'Only the actually verified/captured APK may be archived here.'
signing_text=(E/'capture-v3-apksigner.txt').read_text(encoding='utf-16') if (E/'capture-v3-apksigner.txt').read_bytes().startswith(b'\xff\xfe') else (E/'capture-v3-apksigner.txt').read_text(encoding='utf-8-sig')
assert signing_text.startswith('Verifies')
certificate_sha256=re.search(r'certificate SHA-256 digest: ([0-9a-f]{64})',signing_text).group(1)
with zipfile.ZipFile(apk) as z:
    libs={n:{'sha256':hashlib.sha256(z.read(n)).hexdigest(),'bytes':z.getinfo(n).file_size} for n in z.namelist() if n.startswith('lib/') and n.endswith('.so')}
    models=[n for n in z.namelist() if n.endswith(('.tflite','.keras','.h5'))]
    assert not models
    assert all(n.startswith('lib/x86_64/') for n in libs)
    assert 'lib/x86_64/libBrailleWordGeometry.so' in libs
    assert hashlib.sha256(z.read('assets/index.android.bundle')).hexdigest()==bundle_row['sha256']
capture=json.loads((E/'camera-capture-v3/report.json').read_text())
assert capture['capture_count']==8 and capture['passed_count']==6
assert all(r['jpeg_vs_nitro']['max_absolute_rgb_difference']==0 for r in capture['records'])
roi_corner_error=max(r['roi_independent_corner_check']['max_absolute_corner_error_pixels'] for r in capture['records'])
assert roi_corner_error<.001
assert all(c['guide_membership_matches'] for r in capture['records'] if r['passed'] for c in r['checks'])
crop_checks=[]
for first,second in [(0,1),(2,3)]:
    a,b=[capture['records'][i]['capture']['id'] for i in (first,second)]
    full=np.asarray(Image.open(E/'camera-capture-v3'/a/'imageproxy.jpg').convert('RGB'))
    short=np.asarray(Image.open(E/'camera-capture-v3'/b/'imageproxy.jpg').convert('RGB'))
    checks=[]
    for offset in [0,120,240]:
        difference=np.abs(full[offset:offset+short.shape[0]].astype(np.int16)-short.astype(np.int16))
        checks.append({'raw_jpeg_crop_y':offset,'mae':float(difference.mean()),'max_abs':int(difference.max()),'identical_channel_fraction':float((difference==0).mean())})
    crop_checks.append({'full_capture':a,'short_capture':b,'checks':checks})
report={
    'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'mixed_diagnostic_not_main_apk_or_recognition_acceptance',
    'scope':'Actual VisionCamera5 preview snapshot, Photo ImageProxy geometry, raw JPEG and upright Nitro image on a synthetic Android imagefile feed. No model imported or bundled. No main application edits.',
    'apk':{'path':str(apk),'sha256':sha(apk),'bytes':apk.stat().st_size,'package':'com.brailleme.geometrydiagnostic','debug_signed':True,'certificate_sha256':certificate_sha256,'release_variant_debuggable_for_run_as':True,'bundle':bundle_row,'model_assets':models,'native_libraries':libs,'apksigner_verified':True,'zipalign_16kb_verified':True},
    'package_lock_sha256':sha(STAGE/'package-lock.json'),'current_shared_sources_exactly_equal':True,
    'source_rows':source_rows,'bundle_original_source_bindings':bundle_sources,'evidence_rows':evidence_rows,
    'build':{'gradle_workers':1,'heap_mib':1024,'metaspace_mib':384,'native_jobs':1,'offline':True,'abi':'x86_64','initial_seconds':860,'guard_rebuild_seconds':98,'jpeg_rebuild_seconds':69},
    'emulator':{'version':'37.1.11.0','build_id':'15917651','avd':'BrailleMe_API_36','api':36,'read_only':True,'headless':True,'gpu':'swiftshader','initial_requested_memory_mib':1536,'initial_actual_memory_mib':2560,'repeat_requested_memory_mib':2560,'repeat_actual_memory_mib':2560,'repeat_root_resource_window_approved':True,'shutdown_verified':True,'orientation_source':'interface'},
    'results':{'captures':8,'four_markers_visible_each_capture':True,'four_by_three_cases_passed':6,'four_by_three_cases_total':6,'four_by_three_max_landmark_error_pixels':max(c['error_photo_pixels'] for r in capture['records'] if r['passed'] for c in r['checks']), 'sixteen_by_nine_cases_failed':2,'sixteen_by_nine_error_range_pixels':[min(c['error_photo_pixels'] for r in capture['records'] if not r['passed'] for c in r['checks']),max(c['error_photo_pixels'] for r in capture['records'] if not r['passed'] for c in r['checks'])],'fixed_tolerance_photo_pixels':4,'jpeg_vs_nitro_identical_rgb_captures':8,'actual_rotations_degrees':sorted({r['capture']['geometry']['rotationDegrees'] for r in capture['records']}),'actual_preview_dimensions':[1280,960],'photo_dimensions':[[1280,960],[1280,720]],'raw_jpeg_crop_comparisons':crop_checks},
    'finding':'The 16:9 mismatch is already present between encoded JPEG pixels and the sensor-to-buffer metadata. Nitro toImage reproduces the JPEG pixels exactly after EXIF orientation; no extra crop is introduced. The installed CameraX JPEG in-memory path wraps metadata without cropping pixels in the no-effect path. A pinned public AOSP emulator converter uses origin crop_x/crop_y=0 when fitting differing output aspect ratios, consistent with the measured one-sided JPEG crop instead of the metadata center crop.',
    'independent_roi_checks':{'max_corner_error_pixels':roi_corner_error,'four_by_three_landmark_guide_membership_matches':24,'four_by_three_landmark_guide_membership_total':24},
    'source_inference_limit':'AOSP emu-main-dev revision9172e21fe3376fba0585c69dea67060e16c2b376 is retained by URL/hash as supporting source; it is not cryptographically mapped to installed emulator build15917651. Exact cause inside that emulator binary is inferred from source plus observed pixels, not established by binary instrumentation.',
    'limits':['No main APK smoke or recognition/model accuracy evidence. No physical camera evidence. No production readiness claim.', 'First pass retained with two missing markers; not counted as four-marker success.', '16:9 remains failed; no heuristic 120px correction, source patch, native dependency fork, or changed acceptance threshold.', 'Android emulator imagefile output can crop feed content; feed placement is synthetic and cannot establish main camera word recognition.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(str(OUT/'report.json'));print(sha(OUT/'report.json'));print(json.dumps(report['results']))
