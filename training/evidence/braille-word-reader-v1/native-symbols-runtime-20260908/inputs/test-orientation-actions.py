import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('orientation_actions', Path(__file__).with_name('orientation-actions.py'))
orientation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orientation)


class OrientationHelperTests(unittest.TestCase):
    def test_restores_missing_rotation_via_delete_and_preserves_accelerometer(self):
        helper = SimpleNamespace(mutation=Mock(), run=Mock(side_effect=['null', '1']), log=Mock())
        results = orientation.restore_settings(helper, {'user_rotation': 'null', 'accelerometer_rotation': '1'})
        self.assertTrue(all(row['restored'] for row in results.values()))
        self.assertEqual(helper.mutation.call_args_list[0].args, ('restore-user_rotation', 'shell', 'settings', 'delete', 'system', 'user_rotation'))
        self.assertEqual(helper.mutation.call_args_list[1].args, ('restore-accelerometer_rotation', 'shell', 'settings', 'put', 'system', 'accelerometer_rotation', '1'))

    def test_failed_first_restore_still_attempts_second_and_reports_mismatch(self):
        helper = SimpleNamespace(mutation=Mock(side_effect=[orientation.runtime.SmokeError('offline'), None]),
                                 run=Mock(return_value='0'), log=Mock())
        results = orientation.restore_settings(helper, {'user_rotation': '2', 'accelerometer_rotation': '1'})
        self.assertEqual(helper.mutation.call_count, 2)
        self.assertFalse(results['user_rotation']['restored'])
        self.assertFalse(results['accelerometer_rotation']['restored'])

    def test_pid_change_rejects_even_if_snapshot_ui_is_visible(self):
        helper = SimpleNamespace(snapshot=Mock(), observations=[{'pid': '222'}])
        with self.assertRaisesRegex(orientation.runtime.SmokeError, 'PID changed'):
            orientation.snapshot_same_pid(helper, 'rotated', '111')

    def test_installed_hash_and_path_reject_split_or_wrong_artifact(self):
        digest = 'a'*64
        path = '/data/app/~~Ab_cd==/com.brailleme.app-Zy_23==/base.apk'
        helper = SimpleNamespace(run=Mock(side_effect=['package:' + path, digest + '  ' + path]), log=Mock())
        self.assertEqual(orientation.installed_apk(helper, digest)['sha256'], digest)
        for values in [['package:' + path + '\npackage:/data/app/other.apk'],
                       ['package:' + path, 'b'*64 + '  ' + path]]:
            helper = SimpleNamespace(run=Mock(side_effect=values), log=Mock())
            with self.assertRaises(orientation.runtime.SmokeError):
                orientation.installed_apk(helper, digest)


if __name__ == '__main__':
    unittest.main()
