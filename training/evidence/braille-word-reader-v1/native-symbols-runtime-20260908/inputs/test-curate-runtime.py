import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('curate_runtime', Path(__file__).with_name('curate-runtime.py'))
curator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curator)


class RuntimeCuratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.repo = self.base/'repo'
        self.repo.mkdir()
        self.source = self.base/'runtime600'
        self.source.mkdir()
        self.output = self.repo/'evidence'/'new'

    def run_curator(self, inputs=None, output=None):
        return curator.curate(inputs or [self.source], output or self.output, repository_root=self.repo)

    def test_copy_hashes_nested_files_and_declares_no_acceptance(self):
        (self.source/'frames').mkdir()
        original = self.source/'frames'/'sample.json'
        original.write_text('{"actual":true}')
        report = self.run_curator()
        saved = self.output/'inputs'/'runtime600'/'frames'/'sample.json'
        self.assertEqual(saved.read_bytes(), original.read_bytes())
        self.assertEqual(report['copiedFiles'][0]['sourceSha256'], hashlib.sha256(saved.read_bytes()).hexdigest())
        self.assertFalse(report['limits']['acceptanceInferred'])
        self.assertEqual(json.loads((self.output/'index.json').read_text())['summary']['copiedFiles'], 1)

    def test_conflicting_basenames_are_rejected_before_writes(self):
        second = self.base/'other'/'RUNTIME600'
        second.mkdir(parents=True)
        with self.assertRaises(curator.EvidenceError):
            self.run_curator([self.source, second])
        self.assertFalse(self.output.exists())

    def test_single_file_input_uses_its_basename(self):
        source = self.base/'integrity-v2.json'
        source.write_text('{"recorded":true}')
        report = self.run_curator([source])
        self.assertEqual((self.output/'inputs'/'integrity-v2.json').read_bytes(), source.read_bytes())
        self.assertEqual(report['summary']['copiedFiles'], 1)

    def test_large_file_is_hashed_and_skipped(self):
        large = self.source/'large.log'
        with large.open('wb') as stream:
            stream.seek(curator.MAX_FILE_BYTES)
            stream.write(b'x')
        report = self.run_curator()
        row = report['skippedFiles'][0]
        self.assertEqual(row['sizeBytes'], curator.MAX_FILE_BYTES+1)
        self.assertEqual(row['sourceSha256'], hashlib.sha256(large.read_bytes()).hexdigest())
        self.assertFalse((self.output/row['destination']).exists())
        self.assertEqual(report['summary']['copiedFiles'], 0)

    def test_small_apk_is_not_copied(self):
        package = self.base/'internal.APK'
        package.write_bytes(b'package')
        report = self.run_curator([package])
        self.assertEqual(report['skippedFiles'][0]['reason'], 'excluded_application_package')
        self.assertFalse((self.output/'inputs'/'internal.APK').exists())

    def test_output_outside_repository_is_rejected(self):
        with self.assertRaises(curator.EvidenceError):
            self.run_curator(output=self.base/'outside')
        self.assertFalse((self.base/'outside').exists())

    def test_recursive_containment_guard_rejects_outside_file(self):
        outside = self.base/'outside.log'
        outside.write_text('outside')
        with self.assertRaises(curator.EvidenceError):
            curator._existing(outside, self.source)

    def test_recursive_symlink_is_rejected_before_writes(self):
        outside = self.base/'outside.log'
        outside.write_text('outside')
        link = self.source/'redirect.log'
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'OS does not permit symlink creation: {error}')
        with self.assertRaises(curator.EvidenceError):
            self.run_curator()
        self.assertFalse(self.output.exists())

    def test_recursive_windows_reparse_attribute_is_rejected(self):
        redirect = self.source/'junction'
        redirect.mkdir()
        real_lstat = Path.lstat

        def with_reparse_attribute(path, *args, **kwargs):
            actual = real_lstat(path, *args, **kwargs)
            return SimpleNamespace(st_mode=actual.st_mode, st_file_attributes=0x400) if path == redirect else actual

        with mock.patch.object(Path, 'lstat', with_reparse_attribute):
            with self.assertRaises(curator.EvidenceError):
                self.run_curator()
        self.assertFalse(self.output.exists())

    def test_modified_copy_fails_without_complete_index(self):
        (self.source/'actual.log').write_text('original')
        real_copy = curator.shutil.copyfile

        def corrupt(source, destination, **kwargs):
            real_copy(source, destination, **kwargs)
            Path(destination).write_text('modified')

        with mock.patch.object(curator.shutil, 'copyfile', side_effect=corrupt):
            with self.assertRaises(curator.EvidenceError):
                self.run_curator()
        self.assertFalse((self.output/'index.json').exists())

    def test_new_input_file_during_copy_prevents_complete_index(self):
        (self.source/'actual.log').write_text('original')
        real_copy = curator.shutil.copyfile

        def add_file(source, destination, **kwargs):
            real_copy(source, destination, **kwargs)
            (self.source/'late.log').write_text('arrived after the snapshot')

        with mock.patch.object(curator.shutil, 'copyfile', side_effect=add_file):
            with self.assertRaises(curator.EvidenceError):
                self.run_curator()
        self.assertFalse((self.output/'index.json').exists())

    def test_existing_destination_and_relative_input_are_rejected(self):
        self.output.mkdir(parents=True)
        with self.assertRaises(curator.EvidenceError):
            self.run_curator()
        with self.assertRaises(curator.EvidenceError):
            self.run_curator([Path('relative.log')], self.repo/'another')

    def test_nonexistent_input_is_rejected_without_destination(self):
        with self.assertRaises(OSError):
            self.run_curator([self.base/'missing'])
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
