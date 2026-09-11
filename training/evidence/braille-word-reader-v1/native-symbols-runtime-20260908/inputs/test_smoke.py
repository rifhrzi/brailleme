"""Offline unit tests: no ADB process, emulator, or device is started."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import smoke


class SmokeTests(unittest.TestCase):
    def test_physical_serial_rejected_before_any_adb_call(self):
        with tempfile.TemporaryDirectory() as directory, patch('smoke.subprocess.run') as run:
            with self.assertRaisesRegex(smoke.SmokeError, 'perangkat fisik'):
                smoke.AdbSmoke('adb', 'R58M1234567', directory)
            run.assert_not_called()

    def test_serial_prefix_cannot_bypass_qemu_guard(self):
        replies = [subprocess.CompletedProcess([], 0, 'device\n', ''),
                   subprocess.CompletedProcess([], 0, '0\n', '')]
        with tempfile.TemporaryDirectory() as directory, patch('smoke.subprocess.run', side_effect=replies) as run:
            helper = smoke.AdbSmoke('adb', 'emulator-5554', directory)
            with self.assertRaisesRegex(smoke.SmokeError, 'bukan emulator'):
                helper.guard()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[0],
                             ['adb', '-s', 'emulator-5554', 'shell', 'getprop', 'ro.kernel.qemu'])
            self.assertNotIn('input', run.call_args.args[0])

    def test_exact_matching_ignores_hidden_disabled_and_outside_nodes(self):
        root = ET.fromstring('''<hierarchy><node bounds="[0,0][320,640]">
          <node text="Pindai kata" enabled="true" bounds="[20,400][300,470]"/>
          <node text="Pindai kata lagi" enabled="true" bounds="[20,480][300,550]"/>
          <node text="Pindai kata" enabled="false" bounds="[20,200][300,270]"/>
          <node visible-to-user="false" bounds="[0,0][320,640]">
            <node text="Pindai kata" bounds="[20,50][300,100]"/>
          </node>
          <node text="Pindai kata" bounds="[20,700][300,770]"/>
          <node bounds="[0,0][0,0]"><node text="Pindai kata" bounds="[20,80][300,120]"/></node>
        </node></hierarchy>''')
        found = smoke.find_visible(root, 'text', 'Pindai kata', True)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['bounds'], [20, 400, 300, 470])

    def test_content_description_is_exact_and_bounds_clip_to_visible_parent(self):
        root = ET.fromstring('''<hierarchy><node bounds="[0,100][320,300]">
          <node content-desc="Bacakan" bounds="[20,280][300,350]" enabled="true"/>
        </node></hierarchy>''')
        self.assertEqual(smoke.find_visible(root, 'content-desc', 'Bacakan')[0]['bounds'],
                         [20, 280, 300, 300])
        self.assertEqual(smoke.find_visible(root, 'content-desc', 'Baca'), [])

    def test_duplicate_visible_targets_are_rejected(self):
        root = ET.fromstring('''<hierarchy><node bounds="[0,0][320,640]">
          <node text="Bacakan" bounds="[20,100][300,170]"/>
          <node text="Bacakan" bounds="[20,200][300,270]"/>
        </node></hierarchy>''')
        with tempfile.TemporaryDirectory() as directory:
            helper = smoke.AdbSmoke('adb', 'emulator-5554', directory)
            with patch.object(helper, 'dump', return_value=(root, Path(directory)/'ui.xml')):
                with self.assertRaisesRegex(smoke.SmokeError, 'Label ambigu'):
                    helper.locate('text', 'Bacakan', 1, True)

    def test_polling_has_short_sleeps_and_a_deadline(self):
        clock = [0.]
        sleeps = []
        def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay
        with tempfile.TemporaryDirectory() as directory:
            helper = smoke.AdbSmoke('adb', 'emulator-5554', directory)
            with patch.object(helper, 'dump', return_value=(ET.fromstring('<hierarchy/>'), Path('ui.xml'))), \
                    patch('smoke.time.monotonic', side_effect=lambda: clock[0]), \
                    patch('smoke.time.sleep', side_effect=sleep):
                with self.assertRaisesRegex(smoke.SmokeError, 'Pengamatan berakhir'):
                    helper.locate('text', 'Pindai kata', 1.1, True)
            self.assertAlmostEqual(sum(sleeps), 1.1)
            self.assertLessEqual(max(sleeps), .5)

    def test_taps_use_observed_bounds_and_record_each_action(self):
        element = {'text': 'Pindai kata', 'bounds': [40, 200, 280, 260], 'enabled': True}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(smoke.AdbSmoke, 'guard'), \
                patch.object(smoke.AdbSmoke, 'locate', return_value=element), \
                patch.object(smoke.AdbSmoke, 'run') as run:
            code = smoke.main(['--serial', 'emulator-5554', '--output', directory,
                               'tap', '--text', 'Pindai kata', '--count', '2'])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args, ('shell', 'input', 'tap', '160', '230'))
            entries = [json.loads(line) for line in (Path(directory)/'actions.jsonl').read_text().splitlines()]
            self.assertEqual(len([entry for entry in entries if entry['action'] == 'tap']), 2)
            self.assertIn('no test pass decision', entries[-1]['interpretation'])

    def test_screenshot_uses_remote_file_and_pull_instead_of_binary_shell_redirection(self):
        calls = []
        def run(*arguments, **kwargs):
            calls.append(arguments)
            if arguments[0] == 'pull':
                Path(arguments[2]).write_bytes(b'\x89PNG\r\n\x1a\n')
            return ''
        with tempfile.TemporaryDirectory() as directory:
            helper = smoke.AdbSmoke('adb', 'emulator-5554', directory)
            with patch.object(helper, 'run', side_effect=run):
                result = helper.screenshot('capture')
            self.assertTrue(result.exists())
            self.assertEqual(calls[0][:3], ('shell', 'screencap', '-p'))
            self.assertTrue(calls[0][3].startswith('/sdcard/brailleme-smoke-'))
            self.assertEqual(calls[1][0], 'pull')
            self.assertEqual(calls[2][:3], ('shell', 'rm', '-f'))


if __name__ == '__main__':
    unittest.main()
