"""Offline harness checks; no emulator or app is started."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

spec = importlib.util.spec_from_file_location('runtime_actions', Path(__file__).with_name('runtime-actions.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def hierarchy(enabled='true'):
    return ET.fromstring(f'''<hierarchy><node bounds="[0,0][320,640]">
      <node class="android.view.View" content-desc="Pindai" enabled="true" bounds="[0,590][100,640]"/>
      <node class="android.widget.Button" content-desc="Pindai" enabled="{enabled}" bounds="[20,300][300,360]"/>
      <node class="android.widget.TextView" text="Mencari Braille" bounds="[20,380][300,420]"/>
    </node></hierarchy>''')


class HarnessTests(unittest.TestCase):
    DEX = '/data/local/tmp/brailleme-ui-dump-v3.dex'
    DEX_SHA = '79e234d57492167293889ef7df22815d53caa2f376b8e1ccaf1da031f4dee30a'

    def test_ui_dump_config_rejects_unpinned_or_shell_paths_before_output(self):
        cases = [(self.DEX, None), (None, self.DEX_SHA), (self.DEX + ';id', self.DEX_SHA),
                 ('/sdcard/helper.dex', self.DEX_SHA), (self.DEX, 'bad-hash')]
        for dex, digest in cases:
            with self.subTest(dex=dex, digest=digest), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)/'unused'
                with self.assertRaises(runtime.SmokeError):
                    runtime.Actions('adb', 'emulator-5554', output, 'com.brailleme.app/.MainActivity', dex, digest)
                self.assertFalse(output.exists())

    def test_remote_hash_mismatch_never_executes_ui_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity', self.DEX, self.DEX_SHA)
            with patch.object(helper, 'guard') as guard, patch.object(helper, 'run', return_value='0'*64 + '  ' + self.DEX), \
                 patch.object(runtime.subprocess, 'run') as execute:
                with self.assertRaisesRegex(runtime.SmokeError, 'hash mismatch'):
                    helper.dump()
            guard.assert_called_once()
            execute.assert_not_called()

    def test_no_idle_snapshot_keeps_real_enabled_state_and_checks_hash_each_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity', self.DEX, self.DEX_SHA)
            helper.deadline = time.monotonic() + 3
            tree = hierarchy('false')
            tree.set('snapshot-mode', 'no-idle')
            tree.set('root-wait-ms', '117')
            tree.set('started-uptime-ms', '123456')
            def pull(remote, local, deadline):
                local.write_text(ET.tostring(tree, encoding='unicode'), encoding='utf-8')
            with patch.object(helper, 'guard') as guard, \
                 patch.object(helper, 'run', side_effect=[self.DEX_SHA + '  ' + self.DEX, '0'*64 + '  ' + self.DEX]), \
                 patch.object(helper, 'pull_evidence', side_effect=pull), \
                 patch.object(runtime.subprocess, 'run', return_value=runtime.subprocess.CompletedProcess([], 0, 'UI_DUMP_WRITTEN nodes=4 durationMs=319\n', '')) as execute:
                root, path = helper.dump('real-state')
                self.assertFalse(runtime.observed(root)['shutter_enabled'])
                self.assertTrue(path.is_file())
                self.assertEqual(execute.call_args.args[0][:8], ['adb', '-s', 'emulator-5554', 'shell', 'env', 'CLASSPATH=' + self.DEX, 'app_process', '/system/bin'])
                self.assertGreater(execute.call_args.kwargs['timeout'], 0)
                self.assertLessEqual(execute.call_args.kwargs['timeout'], 3)
                with self.assertRaisesRegex(runtime.SmokeError, 'hash mismatch'):
                    helper.dump('changed-dex')
                execute.assert_called_once()
                self.assertEqual(guard.call_count, 2)
            events = [json.loads(line) for line in helper.log_path.read_text(encoding='utf-8').splitlines()]
            record = next(event for event in events if event['action'] == 'ui_snapshot_observed')
            self.assertEqual(record['dex_sha256'], self.DEX_SHA)
            self.assertEqual(record['root_wait_ms'], '117')

    def test_no_idle_failure_timeout_or_invalid_xml_never_returns_stale_tree(self):
        failures = [runtime.subprocess.CompletedProcess([], 1, '', 'No active root'),
                    runtime.subprocess.CompletedProcess([], 0, '', ''),
                    runtime.subprocess.TimeoutExpired(['adb'], 15),
                    runtime.subprocess.CompletedProcess([], 0, 'UI_DUMP_WRITTEN nodes=4 durationMs=319\n', '')]
        for result in failures:
            with self.subTest(result=str(result)), tempfile.TemporaryDirectory() as temporary:
                helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity', self.DEX, self.DEX_SHA)
                def pull(remote, local, deadline):
                    local.write_text('<hierarchy><node/></hierarchy>', encoding='utf-8')
                kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
                with patch.object(helper, 'guard'), patch.object(helper, 'run', return_value=self.DEX_SHA + '  ' + self.DEX), \
                     patch.object(helper, 'pull_evidence', side_effect=pull), patch.object(runtime.subprocess, 'run', **kwargs):
                    with self.assertRaises(runtime.SmokeError):
                        helper.dump()

    def test_stock_dump_is_unchanged_when_no_dex_is_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            expected = (hierarchy(), Path(temporary)/'ui.xml')
            with patch.object(runtime.AdbSmoke, 'dump', return_value=expected) as dump:
                self.assertEqual(helper.dump('stock', 123), expected)
            dump.assert_called_once_with('stock', 123)

    def test_exact_serial_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'unused'
            with self.assertRaises(runtime.SmokeError):
                runtime.Actions('adb', 'emulator-5556', output, 'com.brailleme.app/.MainActivity')
            self.assertFalse(output.exists())

    def test_wrong_avd_blocks_mutation_after_read_only_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            with patch.object(runtime.AdbSmoke, 'run', side_effect=['device', '1', 'Different_AVD\nOK']) as run:
                with self.assertRaises(runtime.SmokeError):
                    helper.mutation('tap', 'shell', 'input', 'tap', '10', '20')
            self.assertEqual([call.args for call in run.call_args_list], [('get-state',), ('shell', 'getprop', 'ro.kernel.qemu'), ('emu', 'avd', 'name')])

    def test_wrong_qemu_flag_blocks_even_matching_serial(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            with patch.object(runtime.AdbSmoke, 'run', side_effect=['device', '0']) as run:
                with self.assertRaises(runtime.SmokeError):
                    helper.mutation('font', 'shell', 'settings', 'put', 'system', 'font_scale', '2.0')
            self.assertEqual(run.call_count, 2)

    def test_valid_guard_precedes_each_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            with patch.object(runtime.AdbSmoke, 'run', side_effect=['device', '1', 'BrailleMe_API_36\nOK', '']) as run, patch('builtins.print'):
                helper.mutation('tap', 'shell', 'input', 'tap', '10', '20')
            self.assertEqual(run.call_args.args, ('shell', 'input', 'tap', '10', '20'))
            self.assertEqual(helper.action_counts, {'tap': 1})

    def test_shutter_selection_does_not_tap_the_identically_named_tab(self):
        self.assertEqual(runtime.button(hierarchy(), 'Pindai')['bounds'], [20, 300, 300, 360])
        self.assertIsNone(runtime.button(hierarchy('false'), 'Pindai'))
        self.assertFalse(runtime.observed(hierarchy('false'))['shutter_enabled'])

    def test_blank_disabled_shutter_causes_no_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            helper.deadline = time.monotonic() + 60
            with patch.object(helper, 'snapshot', return_value=hierarchy('false')), patch.object(helper, 'mutation') as mutation:
                helper.cycle(0)
            mutation.assert_not_called()
            self.assertEqual(helper.action_counts, {})

    def test_observation_records_exact_result_without_inventing_acceptance(self):
        root = hierarchy()
        ET.SubElement(root[0], 'node', {'content-desc':'Hasil: aab', 'class':'android.widget.TextView', 'bounds':'[20,440][300,490]'})
        result = runtime.observed(root)
        self.assertEqual(result['result_texts'], ['aab'])
        self.assertNotIn('passed', result)

    def test_snapshot_records_no_pid_and_no_log_without_hiding_cold_start_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            result = runtime.subprocess.CompletedProcess([], 1, stdout='', stderr='')
            with patch.object(helper, 'guard'), patch.object(helper, 'dump', return_value=(hierarchy(), Path(temporary)/'ui.xml')), \
                 patch.object(helper, 'screenshot', return_value=Path(temporary)/'ui.png'), \
                 patch.object(runtime.subprocess, 'run', return_value=result) as run, patch('builtins.print'):
                helper.snapshot('after-force-stop', require_app=False)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ['adb', '-s', 'emulator-5554', 'shell', 'pidof', runtime.PACKAGE])
            self.assertIsNone(helper.observations[0]['pid'])
            self.assertIsNone(helper.observations[0]['logcat'])
            rows = [json.loads(line) for line in helper.log_path.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(row['action'] == 'app_pid_observed' and row['pid'] is None for row in rows))

    def test_pid_preserves_real_adb_errors_and_ambiguous_processes(self):
        cases = [(1, '', 'error: device offline'), (127, '', 'pidof unavailable'),
                 (1, 'error: closed', ''), (0, '12 34\n', ''), (0, '', '')]
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            for code, stdout, stderr in cases:
                with self.subTest(code=code, stdout=stdout, stderr=stderr):
                    with patch.object(runtime.subprocess, 'run', return_value=runtime.subprocess.CompletedProcess([], code, stdout, stderr)):
                        with self.assertRaises(runtime.SmokeError):
                            helper.app_pid()

    def test_pid_preserves_timeout_and_records_existing_process_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = runtime.Actions('adb', 'emulator-5554', Path(temporary)/'run', 'com.brailleme.app/.MainActivity')
            helper.deadline = time.monotonic() + 3
            with patch.object(runtime.subprocess, 'run', side_effect=runtime.subprocess.TimeoutExpired(['adb'], 3)):
                with self.assertRaises(runtime.SmokeError):
                    helper.app_pid()
            with patch.object(runtime.subprocess, 'run', return_value=runtime.subprocess.CompletedProcess([], 0, '4321\n', '')) as run:
                self.assertEqual(helper.app_pid(), '4321')
                self.assertGreater(run.call_args.kwargs['timeout'], 0)
                self.assertLessEqual(run.call_args.kwargs['timeout'], 3)
            rows = [json.loads(line) for line in helper.log_path.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(row['action'] == 'adb_error' for row in rows))
            self.assertTrue(any(row['action'] == 'app_pid_observed' and row['pid'] == '4321' for row in rows))

    def run_mocked_main(self, folder, command, pids):
        """Exercise real readiness/summary control flow with only device boundaries mocked."""
        adb, output = folder / 'adb.exe', folder / 'observations'
        adb.write_bytes(b'offline-adb-placeholder-never-executed')
        with patch.object(runtime.Actions, 'guard'), \
             patch.object(runtime.Actions, 'run', return_value='versionName=0.3.0 versionCode=3'), \
             patch.object(runtime.Actions, 'dump', return_value=(hierarchy(), folder / 'ui.xml')), \
             patch.object(runtime.Actions, 'screenshot', return_value=folder / 'ui.png'), \
             patch.object(runtime.Actions, 'app_pid', side_effect=pids), \
             patch.object(runtime.Actions, 'pause'), patch('builtins.print'):
            code = runtime.main(['--adb', str(adb), '--serial', 'emulator-5554',
                                 '--component', 'com.brailleme.app/.MainActivity', '--output', str(output),
                                 '--duration', '20', '--interval', '10', command])
        return code, json.loads((output / 'summary.json').read_text(encoding='utf-8'))

    def test_stale_scanner_ui_without_pid_is_incomplete_for_prepare_and_exercise(self):
        for command, pids in [('prepare', [None, None]), ('exercise', [None]), ('exercise', ['4321', None])]:
            with self.subTest(command=command, pids=pids), tempfile.TemporaryDirectory() as temporary:
                code, summary = self.run_mocked_main(Path(temporary), command, pids)
                self.assertEqual(code, 1)
                self.assertEqual(summary['status'], 'action_or_observation_incomplete')
                self.assertEqual(summary['acceptanceResult'], 'not_assessed')
                self.assertIn('App process is missing', summary['error'])
                last = summary['observations'][-1]
                self.assertTrue(last['shutter_visible'])
                self.assertTrue(last['app_process_required'])
                self.assertIsNone(last['pid'])
                if len(pids) == 2 and pids[0] == '4321':
                    self.assertEqual(summary['action_counts']['Pindai'], 1)

    def test_initial_missing_pid_bootstraps_and_cold_start_preserves_pid_transition(self):
        for command, pids in [('prepare', [None, '5678']), ('cold-start', ['4321', None, '5678'])]:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temporary:
                code, summary = self.run_mocked_main(Path(temporary), command, pids)
                self.assertEqual(code, 0)
                self.assertEqual(summary['status'], 'completed_actions_not_assessed')
                self.assertEqual([row['pid'] for row in summary['observations']], pids)
                self.assertEqual(summary['observations'][-1]['pid'], '5678')
                self.assertTrue(summary['observations'][-1]['app_process_required'])
                self.assertEqual(summary['action_counts']['launch-existing-app'], 1)
                if command == 'cold-start':
                    self.assertEqual(summary['action_counts']['force-stop'], 1)
                    self.assertFalse(summary['observations'][1]['app_process_required'])


if __name__ == '__main__':
    unittest.main()
