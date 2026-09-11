"""Read-only APK integrity checks for the 0.3.0 internal preview; never installs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import zipfile
import zlib


EXPECTED_ABIS = ['arm64-v8a', 'x86_64']
FORBIDDEN_PERMISSIONS = {
    'android.permission.INTERNET', 'android.permission.READ_EXTERNAL_STORAGE',
    'android.permission.WRITE_EXTERNAL_STORAGE', 'android.permission.MANAGE_EXTERNAL_STORAGE',
    'android.permission.WRITE_MEDIA_STORAGE', 'android.permission.MOUNT_UNMOUNT_FILESYSTEMS',
    'android.permission.ACCESS_MEDIA_LOCATION',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_badging(text):
    def attribute(line, key):
        found = re.search(r"(?:^|\s)" + re.escape(key) + r"='([^']*)'", line)
        return found[1] if found else None
    package = next((line for line in text.splitlines() if line.startswith('package: ')), '')
    sdk = re.search(r"^sdkVersion:'(\d+)'$", text, re.MULTILINE)
    native = next((line for line in text.splitlines() if line.startswith('native-code: ')), '')
    return {'package': attribute(package, 'name'), 'version_name': attribute(package, 'versionName'),
            'version_code': attribute(package, 'versionCode'), 'min_sdk': int(sdk[1]) if sdk else None,
            'native_code': sorted(re.findall(r"'([^']+)'", native))}


def parse_permissions(text):
    # Include sdk-23 forms and maxSdkVersion-restricted declarations.
    return sorted(set(re.findall(r"^uses-permission(?:-[^: ]+)?:\s+name='([^']+)'", text, re.MULTILINE)))


def png_dimensions(header):
    if (len(header) < 33 or header[:8] != b'\x89PNG\r\n\x1a\n'
            or struct.unpack('>I', header[8:12])[0] != 13 or header[12:16] != b'IHDR'
            or zlib.crc32(header[12:29]) != struct.unpack('>I', header[29:33])[0]):
        raise ValueError('Startup fixture has no valid PNG IHDR.')
    width, height = struct.unpack('>II', header[16:24])
    if width == 0 or height == 0:
        raise ValueError('Startup fixture PNG dimensions must be positive.')
    return {'width': width, 'height': height}


def inspect_elf(stream, size, abi):
    """Read ELF headers only; no execution and no inference from ZIP alignment."""
    header = stream.read(64)
    if len(header) != 64 or header[:7] != b'\x7fELF\x02\x01\x01':
        raise ValueError('Expected ELF64 little-endian version 1.')
    kind, machine, version, _entry, offset, _shoff, _flags, header_size, entry_size, count, *_rest = struct.unpack('<HHIQQQIHHHHHH', header[16:])
    if kind != 3 or version != 1 or header_size != 64:
        raise ValueError('Expected ELF64 shared object header.')
    if not count or entry_size < 56 or offset < 64 or offset + count * entry_size > size:
        raise ValueError('Invalid or truncated ELF program-header table.')
    loads = []
    for index in range(count):
        stream.seek(offset + index * entry_size)
        values = struct.unpack('<IIQQQQQQ', stream.read(56))
        segment_type, flags, file_offset, virtual_address, _physical_address, file_size, memory_size, alignment = values
        if segment_type != 1:
            continue
        if file_offset + file_size > size or memory_size < file_size or (alignment > 1 and alignment & (alignment - 1)):
            raise ValueError('Invalid ELF PT_LOAD size or alignment.')
        loads.append({'offset': file_offset, 'virtual_address': virtual_address,
                      'file_bytes': file_size, 'memory_bytes': memory_size, 'flags': flags,
                      'alignment': alignment, 'alignment_at_least_16kb': alignment >= 16384})
    if not loads:
        raise ValueError('ELF shared object has no PT_LOAD segments.')
    return {'elf_class': 64, 'byte_order': 'little', 'e_machine': machine,
            'expected_e_machine': {'arm64-v8a': 183, 'x86_64': 62}.get(abi),
            'abi_matches': machine == {'arm64-v8a': 183, 'x86_64': 62}.get(abi),
            'load_segments': loads, 'all_load_alignments_at_least_16kb': all(row['alignment_at_least_16kb'] for row in loads)}


def inspect_archive(apk, expected_hash, expected_fixture_hash):
    with zipfile.ZipFile(apk) as archive:
        names = archive.namelist()
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        abis = sorted({name.split('/')[1] for name in names if re.fullmatch(r'lib/[^/]+/[^/]+\.so', name)})
        models, pngs, native_libraries = [], [], []
        for info in archive.infolist():
            if info.filename.lower().endswith(('.tflite', '.png')):
                with archive.open(info) as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                (models if info.filename.lower().endswith('.tflite') else pngs).append({'entry': info.filename, 'bytes': info.file_size, 'sha256': digest})
            if re.fullmatch(r'lib/[^/]+/[^/]+\.so', info.filename):
                abi = info.filename.split('/')[1]
                row = {'entry': info.filename, 'abi': abi, 'bytes': info.file_size}
                try:
                    with archive.open(info) as stream:
                        row.update(inspect_elf(stream, info.file_size, abi))
                except (ValueError, struct.error) as error:
                    row.update(abi_matches=False, error=str(error), all_load_alignments_at_least_16kb=None)
                native_libraries.append(row)
        # No archive extraction: paths and duplicates cannot overwrite host files.
        bad_crc = archive.testzip()
    return {'entries': len(names), 'duplicate_entries': duplicates, 'abis': abis,
            'models': models, 'matching_word_model_entries': [row['entry'] for row in models if row['sha256'] == expected_hash],
            'pngs': pngs, 'matching_native_fixture_entries': [row for row in pngs if row['sha256'] == expected_fixture_hash],
            'native_libraries': native_libraries,
            'elf_abi_mismatches': [row['entry'] for row in native_libraries if not row['abi_matches']],
            'elf_16kb_alignment_limitations': [row['entry'] for row in native_libraries if row['all_load_alignments_at_least_16kb'] is not True],
            'crc_error_entry': bad_crc}


def execute(name, command, output, environment):
    started = time.monotonic()
    error = None
    try:
        process = subprocess.run([str(arg) for arg in command], capture_output=True,
                                 text=True, encoding='utf-8', errors='replace', env=environment,
                                 timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        code, stdout, stderr = process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired as exception:
        def partial(value):
            return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''
        code, stdout, stderr, error = None, partial(exception.stdout), partial(exception.stderr), str(exception)
    except OSError as exception:
        code, stdout, stderr, error = None, '', '', str(exception)
    stdout_path, stderr_path = output / f'{name}.stdout.txt', output / f'{name}.stderr.txt'
    stdout_path.write_text(stdout, encoding='utf-8')
    stderr_path.write_text(stderr if error is None else stderr + '\n' + error, encoding='utf-8')
    return {'command': [str(arg) for arg in command], 'exit_code': code,
            'elapsed_seconds': round(time.monotonic() - started, 3), 'error': error,
            'stdout_file': stdout_path.name, 'stdout_sha256': sha(stdout_path),
            'stderr_file': stderr_path.name, 'stderr_sha256': sha(stderr_path)}, stdout, stderr


def verify(apk, repo, build_tools, java_home, output, model_directory='braille-word-reader-v1'):
    apk, repo, build_tools, java_home, output = (Path(value).resolve() for value in (apk, repo, build_tools, java_home, output))
    if output.exists():
        raise ValueError('Use a new output directory; existing evidence is never overwritten.')
    if model_directory not in ('braille-word-reader-v1', 'braille-word-reader-id-v5'):
        raise ValueError('Select a supported word-model asset directory.')
    manifest_path = repo / 'assets/models' / model_directory / 'manifest.json'
    java, signer = java_home / 'bin/java.exe', build_tools / 'lib/apksigner.jar'
    zipalign, aapt = build_tools / 'zipalign.exe', build_tools / 'aapt.exe'
    for path in (apk, manifest_path, java, signer, zipalign, aapt):
        if not path.is_file():
            raise ValueError(f'Required input/tool is missing: {path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    artifact = manifest['artifact']
    if not isinstance(artifact['file'], str) or Path(artifact['file']).name != artifact['file']:
        raise ValueError('Model artifact must name one file beside the model manifest.')
    model_path = (manifest_path.parent / artifact['file']).resolve()
    if not model_path.is_relative_to(manifest_path.parent.resolve()) or not model_path.is_file():
        raise ValueError('Model source is missing or outside the model directory.')
    if not re.fullmatch(r'[a-fA-F0-9]{64}', artifact['sha256']):
        raise ValueError('Manifest model hash is invalid.')
    expected_hash = artifact['sha256'].lower()
    fixture = manifest['nativeSelfTest']
    if not isinstance(fixture['file'], str) or Path(fixture['file']).name != fixture['file']:
        raise ValueError('Startup fixture must name one file beside the model manifest.')
    fixture_path = (manifest_path.parent / fixture['file']).resolve()
    if not fixture_path.is_relative_to(manifest_path.parent.resolve()) or not fixture_path.is_file():
        raise ValueError('Startup fixture is missing or outside the model directory.')
    if not re.fullmatch(r'[a-fA-F0-9]{64}', fixture['sha256']):
        raise ValueError('Manifest startup fixture hash is invalid.')
    expected_fixture_hash = fixture['sha256'].lower()
    with fixture_path.open('rb') as stream:
        fixture_dimensions = png_dimensions(stream.read(33))
    environment = os.environ.copy()
    environment['JAVA_HOME'] = str(java_home)
    environment['PATH'] = str(java_home / 'bin') + os.pathsep + environment.get('PATH', '')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 2, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Internal APK archive, signature and declared metadata only; no installation, runtime, physical camera or release-readiness evidence.',
              'apk': {'file': str(apk), 'bytes': apk.stat().st_size, 'sha256': sha(apk)},
              'workspace': str(repo), 'java_home': str(java_home), 'build_tools': str(build_tools),
              'source_sha256': {str(manifest_path.relative_to(repo)).replace('\\', '/'): sha(manifest_path),
                                str(model_path.relative_to(repo)).replace('\\', '/'): sha(model_path),
                                str(fixture_path.relative_to(repo)).replace('\\', '/'): sha(fixture_path)},
              'verifier_sha256': sha(Path(__file__)),
              'tool_sha256': {str(path): sha(path) for path in (java, signer, zipalign, aapt)},
              'expected': {'package': 'com.brailleme.app', 'version_name': '0.3.0', 'version_code': '3',
                           'min_sdk': 29, 'abis': EXPECTED_ABIS, 'word_model_sha256': expected_hash,
                           'native_fixture_sha256': expected_fixture_hash},
              'native_fixture_source': {'file': str(fixture_path), 'sha256': sha(fixture_path),
                                        'bytes': fixture_path.stat().st_size, **fixture_dimensions},
              'deployment_limitations': [], 'elf_16kb_segment_alignment_passed': None,
              'tools': {}, 'checks': {}, 'errors': [], 'runtime_assessed': False,
              'physical_camera_validated': False, 'release_readiness': 'not_assessed'}
    checks = report['checks']
    checks['workspace_model_matches_manifest'] = sha(model_path) == expected_hash
    checks['workspace_model_size_matches_manifest'] = model_path.stat().st_size == artifact.get('sizeBytes')
    checks['workspace_native_fixture_matches_manifest'] = sha(fixture_path) == expected_fixture_hash
    checks['workspace_native_fixture_dimensions_match_manifest'] = fixture_dimensions == {key: fixture.get(key) for key in ('width', 'height')}
    try:
        archive = inspect_archive(apk, expected_hash, expected_fixture_hash)
        report['archive'] = archive
        checks['zip_crc_valid'] = archive['crc_error_entry'] is None
        checks['zip_entries_unique'] = not archive['duplicate_entries']
        checks['archive_abis_match'] = archive['abis'] == EXPECTED_ABIS
        checks['embedded_word_model_matches_manifest'] = len(archive['matching_word_model_entries']) == 1
        matching_fixtures = archive['matching_native_fixture_entries']
        checks['embedded_native_fixture_matches_manifest'] = len(matching_fixtures) == 1 and matching_fixtures[0]['bytes'] == fixture_path.stat().st_size
        checks['native_elf64_little_endian_abis_match'] = bool(archive['native_libraries']) and not archive['elf_abi_mismatches']
        report['elf_16kb_segment_alignment_passed'] = bool(archive['native_libraries']) and not archive['elf_16kb_alignment_limitations']
        if archive['elf_16kb_alignment_limitations']:
            report['deployment_limitations'].append({
                'kind': 'elf_load_segment_alignment_below_16kb_or_unreadable',
                'libraries': archive['elf_16kb_alignment_limitations'],
                'meaning': '16 KB native segment alignment is not established. This is separate from ZIP alignment; internal 4 KB-page runtime must still be tested.',
            })
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error) as error:
        report['errors'].append(f'APK ZIP inspection failed: {error}')
        checks['archive_valid'] = False
    commands = [('java-version', [java, '-version']),
                ('apksigner', [java, '-jar', signer, 'verify', '--verbose', '--print-certs', apk]),
                ('zipalign', [zipalign, '-c', '-P', '16', '-v', '4', apk]),
                ('aapt-badging', [aapt, 'dump', 'badging', apk]),
                ('aapt-permissions', [aapt, 'dump', 'permissions', apk])]
    stdout_by_tool = {}
    for name, command in commands:
        result, stdout, stderr = execute(name, command, output, environment)
        report['tools'][name] = result
        checks[name + '_exit_zero'] = result['exit_code'] == 0
        stdout_by_tool[name] = stdout
        if name == 'java-version':
            report['java_version_output'] = stdout + stderr
            checks['java_major_17'] = bool(re.search(r'\bversion "17(?:\.|["-])', stdout + stderr))
        print(f'{name}: exit={result["exit_code"]}', flush=True)
    report['badging'] = parse_badging(stdout_by_tool['aapt-badging'])
    for field in ('package', 'version_name', 'version_code', 'min_sdk'):
        checks[field + '_matches'] = report['badging'][field] == report['expected'][field]
    checks['badging_abis_match'] = report['badging']['native_code'] == EXPECTED_ABIS
    permissions = parse_permissions(stdout_by_tool['aapt-permissions'])
    report['permissions'] = permissions
    report['forbidden_permissions_found'] = [value for value in permissions if value in FORBIDDEN_PERMISSIONS or value.startswith('android.permission.READ_MEDIA_')]
    checks['internet_and_storage_permissions_absent'] = not report['forbidden_permissions_found']
    checks['camera_permission_present'] = 'android.permission.CAMERA' in permissions
    certificates = re.findall(r'^Signer #\d+ certificate DN: (.+)$', stdout_by_tool['apksigner'], re.MULTILINE)
    report['signer_certificate_dns'] = certificates
    report['signer_certificate_sha256'] = re.findall(r'^Signer #\d+ certificate SHA-256 digest: ([a-fA-F0-9]+)$', stdout_by_tool['apksigner'], re.MULTILINE)
    checks['single_android_debug_certificate_dn'] = len(certificates) == 1 and bool(re.search(r'(?:^|,)\s*CN=Android Debug(?:,|$)', certificates[0]))
    checks['signer_certificate_hash_reported'] = len(report['signer_certificate_sha256']) == 1 and bool(re.fullmatch(r'[a-fA-F0-9]{64}', report['signer_certificate_sha256'][0]))
    checks['apk_unchanged_during_verification'] = sha(apk) == report['apk']['sha256']
    checks['workspace_inputs_unchanged'] = all(sha(repo / path) == value for path, value in report['source_sha256'].items())
    report['failed_checks'] = [name for name, value in checks.items() if not value]
    report['integrity_passed'] = bool(checks) and not report['failed_checks'] and not report['errors']
    report['status'] = 'passed_internal_apk_integrity' if report['integrity_passed'] else 'failed'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output / 'apk.sha256').write_text(report['apk']['sha256'] + '  ' + apk.name + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('apk', 'repo', 'build-tools', 'java-home', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-directory', choices=('braille-word-reader-v1', 'braille-word-reader-id-v5'),
                        default='braille-word-reader-v1', help='Word model expected inside this APK')
    args = parser.parse_args(argv)
    try:
        result = verify(args.apk, args.repo, args.build_tools, args.java_home, args.output, args.model_directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(json.dumps({'status': result['status'], 'apk_sha256': result['apk']['sha256'],
                      'failed_checks': result['failed_checks'], 'errors': result['errors'],
                      'elf_16kb_segment_alignment_passed': result['elf_16kb_segment_alignment_passed'],
                      'deployment_limitations': result['deployment_limitations'],
                      'report': str(args.output.resolve() / 'report.json')}), flush=True)
    return 0 if result['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
