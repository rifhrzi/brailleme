"""Inspect the exact internal APK after Android build; never implies runtime QA."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile
from PIL import Image

REQUIRED_NATIVE_LIBRARIES = ['BrailleWordGeometry', 'VisionCamera', 'NitroImage', 'NitroModules', 'NitroTflite',
                             'worklets', 'tensorflowlite_jni', 'fbjni', 'jsi', 'reactnative', 'c++_shared', 'hermesvm',
                             'expo-modules-core', 'expo-sqlite']


def load_fixture_pixels(path, expected_hash):
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise ValueError('Local startup fixture differs from the model manifest SHA-256')
    with Image.open(io.BytesIO(encoded)) as original:
        return original.size, original.convert('RGBA').tobytes()

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--apk', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--keystore', type=Path, required=True, help='Exact debug keystore used by this build; certificate identity must match')
args = parser.parse_args()
root = Path.cwd()
manifest_path = root / 'assets/models/braille-word-reader-v1/manifest.json'
manifest = json.loads(manifest_path.read_bytes())
args.output.mkdir(parents=True, exist_ok=True)
build_tools = Path(os.environ['LOCALAPPDATA']) / 'Android/Sdk/build-tools/36.0.0'
environment = dict(os.environ, JAVA_HOME=r'C:\Users\rifqi\.jdks\brailleme-temurin-17\jdk-17.0.20+8')
environment['PATH'] = str(Path(environment['JAVA_HOME']) / 'bin') + os.pathsep + environment['PATH']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(name, command):
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    output = completed.stdout + completed.stderr
    (args.output / f'{name}.txt').write_text(output, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError(f'{name} failed: {completed.returncode}; see retained log')
    return output


signature = inspect('apksigner', [str(build_tools / 'apksigner.bat'), 'verify', '--verbose', '--print-certs', str(args.apk)])
inspect('zipalign', [str(build_tools / 'zipalign.exe'), '-c', '-P', '16', '-v', '4', str(args.apk)])
badging = inspect('aapt-badging', [str(build_tools / 'aapt.exe'), 'dump', 'badging', str(args.apk)])
permissions = inspect('aapt-permissions', [str(build_tools / 'aapt.exe'), 'dump', 'permissions', str(args.apk)])
assert "name='com.brailleme.app'" in badging
assert "versionCode='2'" in badging and "versionName='0.2.0'" in badging
assert "sdkVersion:'29'" in badging and "targetSdkVersion:'36'" in badging
assert 'android.permission.CAMERA' in permissions
for forbidden in ('INTERNET', 'ACCESS_NETWORK_STATE', 'READ_EXTERNAL_STORAGE', 'WRITE_EXTERNAL_STORAGE', 'RECORD_AUDIO'):
    assert f'android.permission.{forbidden}' not in permissions, forbidden
assert 'CN=Android Debug' in signature, 'Expected the declared internal debug certificate'
keystore_certificate = inspect('build-keystore-certificate', [str(Path(environment['JAVA_HOME']) / 'bin/keytool.exe'),
    '-list', '-v', '-keystore', str(args.keystore), '-alias', 'androiddebugkey', '-storepass', 'android'])
expected_fingerprint = re.search(r'SHA256:\s*([a-fA-F0-9:]+)', keystore_certificate)
assert expected_fingerprint is not None, 'Build keystore SHA-256 certificate fingerprint is missing'

with zipfile.ZipFile(args.apk) as archive:
    names = archive.namelist()
    abis = sorted({name.split('/')[1] for name in names if name.startswith('lib/') and name.endswith('.so')})
    assert abis == ['arm64-v8a', 'armeabi-v7a', 'x86', 'x86_64'], abis
    elf_architectures = {'armeabi-v7a': (1, 40), 'arm64-v8a': (2, 183), 'x86': (1, 3), 'x86_64': (2, 62)}
    native_libraries = []
    for abi in abis:
        for library in REQUIRED_NATIVE_LIBRARIES:
            entry = f'lib/{abi}/lib{library}.so'
            assert entry in names, f'Missing required native library: {entry}'
            binary = archive.read(entry)
            assert binary[:4] == b'\x7fELF' and binary[5] == 1, f'Invalid little-endian ELF: {entry}'
            elf_class, machine = binary[4], int.from_bytes(binary[18:20], 'little')
            assert (elf_class, machine) == elf_architectures[abi], f'ELF architecture differs from ABI path: {entry}'
            native_libraries.append({'file': entry, 'sha256': sha(binary), 'sizeBytes': len(binary), 'elfClass': elf_class, 'elfMachine': machine})
    models = [{'file': name, 'sha256': sha(archive.read(name)), 'sizeBytes': archive.getinfo(name).file_size} for name in names if name.endswith('.tflite')]
    assert sum(model['sha256'] == manifest['artifact']['sha256'] for model in models) == 1
    bundle = 'assets/index.android.bundle'
    assert bundle in names
    bundle_hash = sha(archive.read(bundle))
    fixture_hash = manifest['nativeSelfTest']['sha256']
    fixtures = [name for name in names if name.endswith('.png') and sha(archive.read(name)) == fixture_hash]
    fixture_pixel_matches = []
    # AAPT may rewrite PNG encoding losslessly. Preserve both byte hashes and
    # prove every decoded RGBA pixel, rather than claiming unchanged PNG bytes.
    expected_size, expected_rgba = load_fixture_pixels(manifest_path.parent / manifest['nativeSelfTest']['file'], fixture_hash)
    for name in names:
        if not name.endswith('.png'):
            continue
        encoded = archive.read(name)
        with Image.open(io.BytesIO(encoded)) as bundled:
            if bundled.size == expected_size and bundled.convert('RGBA').tobytes() == expected_rgba:
                fixture_pixel_matches.append({'file': name, 'sha256': sha(encoded), 'identicalPngBytes': sha(encoded) == fixture_hash})
    assert fixture_pixel_matches, 'Bundled startup fixture must preserve all decoded source pixels'

fingerprint = re.search(r'Signer #1 certificate SHA-256 digest: ([a-fA-F0-9]+)', signature)
assert fingerprint is not None
assert fingerprint.group(1).lower() == expected_fingerprint.group(1).replace(':', '').lower(), 'APK certificate differs from the build keystore'
report = {
    'scope': 'Debug-signed universal internal APK artifact inspection. This report does not claim runtime, physical-camera, or production readiness.',
    'apk': str(args.apk.resolve()), 'apkSha256': sha(args.apk.read_bytes()), 'sizeBytes': args.apk.stat().st_size,
    'package': 'com.brailleme.app', 'versionName': '0.2.0', 'versionCode': 2,
    'minimumSdk': 29, 'targetSdk': 36, 'abis': abis,
    'signing': {'verified': True, 'scope': 'android_debug_internal_only', 'certificateSha256': fingerprint.group(1).lower(),
                'matchesBuildKeystore': True, 'buildKeystoreSha256': sha(args.keystore.read_bytes())},
    'zipAlignmentPassed': True, 'uncompressedNativeZipPageAlignmentKiB': 16, 'offlinePermissionChecksPassed': True,
    'wordModelSha256': manifest['artifact']['sha256'], 'modelManifestSha256': sha(manifest_path.read_bytes()),
    'bundledModels': models, 'bundleSha256': bundle_hash,
    'requiredNativeLibraries': native_libraries,
    'startupFixtureSha256': fixture_hash, 'startupFixtureApkEntries': fixtures,
    'startupFixturePixelMatches': fixture_pixel_matches,
    'physicalCameraValidated': False,
}
(args.output / 'apk-inspection.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
