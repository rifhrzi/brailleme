"""Verifier failure-path unit tests; no actual APK, signer or device proof."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile
import zlib

import verify_apk


BADGING = """package: name='com.brailleme.app' versionCode='3' versionName='0.3.0' platformBuildVersionName='16'
sdkVersion:'29'
targetSdkVersion:'36'
native-code: 'arm64-v8a' 'x86_64'
"""
SIGNER = """Verifies
Signer #1 certificate DN: C=US, O=Android, CN=Android Debug
Signer #1 certificate SHA-256 digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
"""


def png_fixture():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0\x12\x34\x56')) + chunk(b'IEND', b''))


def elf_fixture(machine, alignment=16384):
    ident = b'\x7fELF\x02\x01\x01' + b'\0' * 9
    header = struct.pack('<HHIQQQIHHHHHH', 3, machine, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    segment = struct.pack('<IIQQQQQQ', 1, 5, 0, 0, 0, 120, 120, alignment)
    return ident + header + segment


class ParsingTests(unittest.TestCase):
    def test_badging_keeps_exact_versions_and_declared_abis(self):
        result = verify_apk.parse_badging(BADGING)
        self.assertEqual(result, {'package': 'com.brailleme.app', 'version_name': '0.3.0',
                                 'version_code': '3', 'min_sdk': 29,
                                 'native_code': ['arm64-v8a', 'x86_64']})
        self.assertIsNone(verify_apk.parse_badging("targetSdkVersion:'29'")['min_sdk'])
        self.assertIsNone(verify_apk.parse_badging('')['version_name'])

    def test_sdk_limited_storage_declarations_are_not_ignored(self):
        result = verify_apk.parse_permissions("package: com.brailleme.app\n"
            "uses-permission: name='android.permission.CAMERA'\n"
            "uses-permission-sdk-23: name='android.permission.READ_EXTERNAL_STORAGE' maxSdkVersion='32'\n"
            "uses-permission: name='android.permission.INTERNET'\n"
            "permission: name='com.brailleme.app.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION'\n")
        self.assertEqual(result, ['android.permission.CAMERA', 'android.permission.INTERNET',
                                  'android.permission.READ_EXTERNAL_STORAGE'])

    def test_archive_checks_bytes_and_rejects_duplicate_entry_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            apk = Path(temporary) / 'fixture.zip'
            content = b'independent test model bytes'
            with zipfile.ZipFile(apk, 'w') as archive:
                archive.writestr('res/raw/word.tflite', content)
                archive.writestr('lib/arm64-v8a/libtest.so', b'arm')
                archive.writestr('lib/x86_64/libtest.so', b'x86')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    archive.writestr('res/raw/word.tflite', b'changed')
            result = verify_apk.inspect_archive(apk, hashlib.sha256(content).hexdigest(), 'f' * 64)
            self.assertEqual(result['matching_word_model_entries'], ['res/raw/word.tflite'])
            self.assertEqual(result['duplicate_entries'], ['res/raw/word.tflite'])
            self.assertEqual(result['abis'], ['arm64-v8a', 'x86_64'])

    def test_elf_class_byte_order_and_truncated_program_table_rejected(self):
        original = elf_fixture(183)
        invalid = [original[:4] + b'\x01' + original[5:],
                   original[:5] + b'\x02' + original[6:], original[:90]]
        for data in invalid:
            with self.subTest(data=data[:8]), self.assertRaises(ValueError):
                verify_apk.inspect_elf(io.BytesIO(data), len(data), 'arm64-v8a')

    def test_png_header_dimensions_and_crc_are_checked(self):
        self.assertEqual(verify_apk.png_dimensions(png_fixture()[:33]), {'width': 1, 'height': 1})
        changed = bytearray(png_fixture()[:33])
        changed[19] = 2
        with self.assertRaises(ValueError):
            verify_apk.png_dimensions(changed)


class VerificationTests(unittest.TestCase):
    def test_existing_output_rejected_before_inputs_or_tools(self):
        with tempfile.TemporaryDirectory() as temporary, patch('verify_apk.subprocess.run') as run:
            with self.assertRaisesRegex(ValueError, 'new output'):
                verify_apk.verify('missing.apk', '.', '.', '.', temporary)
            run.assert_not_called()

    def run_sample(self, signer_exit=0, fixture_mutation=None, source_mutation=False, machine=183, alignment=16384):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, sdk, jdk, output = [root / name for name in ('repo', 'sdk', 'jdk', 'result')]
            model_dir = repo / 'assets/models/braille-word-reader-v1'
            model_dir.mkdir(parents=True)
            model = model_dir / 'word-reader.tflite'
            model.write_bytes(b'unit-test-model')
            fixture = model_dir / 'native-word-fixture.png'
            fixture.write_bytes(png_fixture())
            (model_dir / 'manifest.json').write_text(json.dumps({'artifact': {
                'file': model.name, 'sizeBytes': model.stat().st_size,
                'sha256': verify_apk.sha(model)}, 'nativeSelfTest': {
                    'file': fixture.name, 'sha256': verify_apk.sha(fixture), 'width': 1, 'height': 1}}))
            if source_mutation:
                fixture.write_bytes(png_fixture() + b'changed')
            for file in [jdk / 'bin/java.exe', sdk / 'lib/apksigner.jar', sdk / 'zipalign.exe', sdk / 'aapt.exe']:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'not an executable; subprocess is mocked in this unit test')
            apk = root / 'fixture.zip'
            with zipfile.ZipFile(apk, 'w') as archive:
                archive.writestr('res/raw/word.tflite', model.read_bytes())
                if fixture_mutation != 'missing':
                    archive.writestr('res/drawable/nativewordfixture.png', png_fixture() + (b'changed' if fixture_mutation == 'tamper' else b''))
                if fixture_mutation == 'duplicate':
                    archive.writestr('assets/copy.png', png_fixture())
                for abi in verify_apk.EXPECTED_ABIS:
                    archive.writestr(f'lib/{abi}/libfixture.so', elf_fixture(machine if abi == 'arm64-v8a' else 62, alignment))
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                self.assertEqual(kwargs['env']['JAVA_HOME'], str(jdk))
                self.assertNotIn('shell', kwargs)
                if '-version' in command:
                    return subprocess.CompletedProcess(command, 0, '', 'openjdk version "17.0.20"\n')
                if '-jar' in command:
                    return subprocess.CompletedProcess(command, signer_exit, SIGNER, 'actual verification failed' if signer_exit else '')
                if 'badging' in command:
                    return subprocess.CompletedProcess(command, 0, BADGING, '')
                if 'permissions' in command:
                    return subprocess.CompletedProcess(command, 0, "uses-permission: name='android.permission.CAMERA'\n", '')
                self.assertEqual(command[1:6], ['-c', '-P', '16', '-v', '4'])
                return subprocess.CompletedProcess(command, 0, 'Verification successful', '')
            with patch('verify_apk.subprocess.run', side_effect=execute), patch('builtins.print'):
                result = verify_apk.verify(apk, repo, sdk, jdk, output)
            self.assertEqual(len(calls), 5)
            self.assertTrue(all('install' not in command for command in calls))
            saved = json.loads((output / 'report.json').read_text())
            self.assertEqual(saved, result)
            return result

    def test_failed_real_tool_exit_cannot_become_pass_from_success_text(self):
        result = self.run_sample(signer_exit=1)
        self.assertFalse(result['integrity_passed'])
        self.assertEqual(result['failed_checks'], ['apksigner_exit_zero'])
        self.assertEqual(result['tools']['apksigner']['exit_code'], 1)

    def test_missing_changed_or_duplicate_bundled_fixture_rejected(self):
        for mutation in ('missing', 'tamper', 'duplicate'):
            with self.subTest(mutation=mutation):
                result = self.run_sample(fixture_mutation=mutation)
                self.assertFalse(result['integrity_passed'])
                self.assertIn('embedded_native_fixture_matches_manifest', result['failed_checks'])

    def test_source_fixture_tamper_rejected_and_identity_bound(self):
        result = self.run_sample(source_mutation=True)
        self.assertFalse(result['integrity_passed'])
        self.assertIn('workspace_native_fixture_matches_manifest', result['failed_checks'])
        self.assertIn('assets/models/braille-word-reader-v1/native-word-fixture.png', result['source_sha256'])

    def test_wrong_elf_machine_in_correctly_named_abi_directory_rejected(self):
        result = self.run_sample(machine=62)
        self.assertFalse(result['integrity_passed'])
        self.assertEqual(result['failed_checks'], ['native_elf64_little_endian_abis_match'])
        self.assertEqual(result['archive']['elf_abi_mismatches'], ['lib/arm64-v8a/libfixture.so'])

    def test_4kb_elf_segments_are_separate_from_successful_zip_alignment(self):
        result = self.run_sample(alignment=4096)
        self.assertTrue(result['integrity_passed'])
        self.assertTrue(result['checks']['zipalign_exit_zero'])
        self.assertFalse(result['elf_16kb_segment_alignment_passed'])
        self.assertEqual(len(result['deployment_limitations'][0]['libraries']), 2)
        self.assertEqual(result['archive']['native_libraries'][0]['load_segments'][0]['alignment'], 4096)

    def test_16kb_headers_alone_do_not_claim_runtime_readiness(self):
        result = self.run_sample()
        self.assertTrue(result['integrity_passed'])
        self.assertTrue(result['elf_16kb_segment_alignment_passed'])
        self.assertFalse(result['runtime_assessed'])
        self.assertEqual(result['release_readiness'], 'not_assessed')

    def test_timeout_is_logged_without_a_success_exit_code(self):
        timeout = subprocess.TimeoutExpired(['tool'], 60, output=b'partial output', stderr=b'partial error')
        with tempfile.TemporaryDirectory() as temporary, patch('verify_apk.subprocess.run', side_effect=timeout):
            result, stdout, stderr = verify_apk.execute('timeout', ['tool'], Path(temporary), {})
            self.assertIsNone(result['exit_code'])
            self.assertIsNotNone(result['error'])
            self.assertEqual((stdout, stderr), ('partial output', 'partial error'))
            self.assertEqual((Path(temporary) / 'timeout.stdout.txt').read_text(), 'partial output')
            self.assertIn('timed out', (Path(temporary) / 'timeout.stderr.txt').read_text())


if __name__ == '__main__':
    unittest.main()
