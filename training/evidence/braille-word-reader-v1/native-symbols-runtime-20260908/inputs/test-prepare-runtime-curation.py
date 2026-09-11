import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('prepare_curation', Path(__file__).with_name('prepare-runtime-curation.py'))
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


class PlannerTests(unittest.TestCase):
    def test_rejects_duplicate_basename_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'a'/'same.json', Path(tmp)/'b'/'SAME.json'
            first.parent.mkdir(); second.parent.mkdir()
            first.write_text('one'); second.write_text('two')
            with self.assertRaises(planner.curator.EvidenceError):
                planner.scan_inputs([first, second])

    def test_estimate_excludes_package_and_rejects_model_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'; folder.mkdir()
            (folder/'report.json').write_bytes(b'{}')
            (folder/'build.apk').write_bytes(b'12345')
            sources, files = planner.scan_inputs([folder])
            self.assertEqual(sources[0]['estimated_copy_bytes'], 2)
            self.assertEqual(sources[0]['indexed_only_bytes'], 5)
            self.assertEqual(len(files), 2)
            (folder/'model.tflite').write_bytes(b'no duplicate')
            with self.assertRaises(planner.curator.EvidenceError):
                planner.scan_inputs([folder])

    def test_changed_or_new_source_rejects_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'; folder.mkdir()
            source = folder/'log.txt'; source.write_text('original')
            sources, files = planner.scan_inputs([folder])
            plan = {'schema_version': 1, 'status': 'prepared_not_copied', 'output': str(planner.TARGET), 'inputs': sources, 'files': files}
            with patch.object(planner.curator, '_hash', wraps=planner.curator._hash):
                source.write_text('modified')
                with self.assertRaisesRegex(planner.curator.EvidenceError, 'changed after planning'):
                    planner.verify_plan(plan)


if __name__ == '__main__':
    unittest.main()
