import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from collect_runtime import RuntimeCollector, normalize_surface_layer, parse_memory, parse_metrics, parse_surface_latency, select_surface_layer
from smoke import SmokeError


TRACKING = {'schemaVersion': 1, 'elapsedMs': 10000, 'received': 300, 'admitted': 50,
            'busyDrops': 0, 'throttled': 250, 'disabledDrops': 0, 'runnerDrops': 0,
            'completed': 50, 'errors': 0, 'meanMs': 30, 'maxMs': 60,
            'latencyBuckets': [49, 1, 0, 0, 0, 0, 0]}


class ParsingTests(unittest.TestCase):
    def test_metrics_allowlist_drops_text_coordinates_and_unknown_fields(self):
        log = '1000.5 100 100 I ReactNativeJS: BRAILLE_WORD_TRACKING_METRICS ' + json.dumps({**TRACKING, 'text': 'secret', 'points': [1, 2]})
        result = parse_metrics(log)
        self.assertEqual(result['metrics']['completed'], 50)
        self.assertNotIn('text', result['metrics'])
        self.assertNotIn('points', result['metrics'])
        self.assertIsNone(parse_metrics(log.replace('"meanMs": 30', '"meanMs": "30"')))
        self.assertIsNone(parse_metrics('1000.5 ReactNativeJS: user text'))

    def test_photos_keep_only_numeric_timing_and_known_status(self):
        value = {'schemaVersion': 1, 'captureId': 3, 'elapsedMs': 432.5, 'status': 'success', 'text': 'secret'}
        result = parse_metrics('1000.6 I ReactNativeJS: BRAILLE_WORD_PHOTO_METRICS ' + json.dumps(value))
        self.assertEqual(result['metrics'], {key: value[key] for key in ('schemaVersion', 'captureId', 'elapsedMs', 'status')})

    def test_react_native_two_string_arguments_are_quoted_and_comma_separated(self):
        line = "1000.5 100 100 I ReactNativeJS: 'BRAILLE_WORD_TRACKING_METRICS', '" + json.dumps(TRACKING) + "'"
        result = parse_metrics(line)
        self.assertEqual(result['metrics']['completed'], 50)
        self.assertEqual(result['deviceEpochSeconds'], 1000.5)
        photo = {'schemaVersion': 1, 'captureId': 1, 'elapsedMs': 150, 'status': 'rejected'}
        line = "1000.6 I ReactNativeJS: 'BRAILLE_WORD_PHOTO_METRICS', '" + json.dumps(photo) + "'"
        self.assertEqual(parse_metrics(line)['metrics'], photo)

    def test_memory_supports_summary_and_old_total_rows(self):
        self.assertEqual(parse_memory(' TOTAL PSS: 12345 TOTAL RSS: 23456'), {'pssKiB': 12345, 'rssKiB': 23456})
        self.assertEqual(parse_memory(' TOTAL 12345 9000 2 3'), {'pssKiB': 12345, 'rssKiB': None})
        self.assertEqual(parse_memory('No process found'), {'pssKiB': None, 'rssKiB': None})

    def test_surface_uses_actual_presentation_column_and_discards_pending_frames(self):
        result = parse_surface_latency('16666666\n100 200 300\n110 200 310\n0 0 0\n1 9223372036854775807 2\n')
        self.assertEqual(result, {'refreshPeriodNs': 16666666, 'presentedNs': [200]})
        self.assertIsNone(parse_surface_latency('Permission denied'))

    def test_api36_requested_layer_wrapper_preserves_the_actual_buffer_name(self):
        name = 'com.brailleme.app/com.brailleme.app.MainActivity#246'
        for suffix in ('', ' parentId=245', ' parentId=245 z=0', ' z=4 parentId=245'):
            with self.subTest(suffix=suffix):
                self.assertEqual(normalize_surface_layer('RequestedLayerState{' + name + suffix + '}'), name)
        self.assertEqual(normalize_surface_layer(name), name)
        self.assertEqual(normalize_surface_layer('RequestedLayerState{broken parentId=1}'), 'RequestedLayerState{broken parentId=1}')

    def test_selects_exact_activity_buffer_among_api36_containers_and_input_sinks(self):
        lines = [
            'RequestedLayerState{ActivityRecord{abc u0 com.brailleme.app/.MainActivity t3}#240 parentId=1}',
            'RequestedLayerState{ActivityRecordInputSink com.brailleme.app/.MainActivity#243 parentId=240}',
            'RequestedLayerState{c7b5903 com.brailleme.app/com.brailleme.app.MainActivity#245 parentId=240}',
            'RequestedLayerState{com.brailleme.app/com.brailleme.app.MainActivity#246 parentId=245}',
        ]
        self.assertEqual(select_surface_layer([normalize_surface_layer(line) for line in lines], 'com.brailleme.app'),
                         'com.brailleme.app/com.brailleme.app.MainActivity#246')

    def test_ambiguous_buffers_or_only_containers_never_fabricate_a_surface_choice(self):
        package = 'com.brailleme.app'
        self.assertIsNone(select_surface_layer([f'{package}/{package}.MainActivity#246', f'{package}/{package}.MainActivity#247'], package))
        for name in (f'c7b5903 {package}/{package}.MainActivity#245',
                     f'ActivityRecordInputSink {package}/.MainActivity#243',
                     f'ActivityRecord{{abc u0 {package}/.MainActivity t3}}#240',
                     f'{package}.other/{package}.MainActivity#246'):
            with self.subTest(name=name):
                self.assertIsNone(select_surface_layer([name], package))

    def test_legacy_unwrapped_activity_and_unique_surface_view_remain_supported(self):
        package = 'com.brailleme.app'
        self.assertEqual(select_surface_layer([f'{package}/MainActivity'], package), f'{package}/MainActivity')
        surface = f'SurfaceView[{package}/{package}.MainActivity]#44'
        self.assertEqual(select_surface_layer([surface], package), surface)
        self.assertIsNone(select_surface_layer([surface, surface + '(BLAST)'], package))


class CollectionTests(unittest.TestCase):
    def test_guard_requires_explicit_emulator_and_qemu(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(SmokeError):
                RuntimeCollector('adb', 'physical-123', 'com.brailleme.app', output)
            collector = RuntimeCollector('adb', 'emulator-5554', 'com.brailleme.app', output)
            with patch.object(collector, 'run', side_effect=['device', '0']):
                with self.assertRaises(SmokeError):
                    collector.guard()

    def test_one_second_collection_is_read_only_and_labels_performance_unassessed(self):
        clock = [0.0]
        latency_calls = [0]
        commands = []

        def run(command, **_kwargs):
            commands.append(command)
            args = command[3:]
            if args == ['get-state']:
                stdout = 'device\n'
            elif args == ['shell', 'getprop', 'ro.kernel.qemu']:
                stdout = '1\n'
            elif args == ['shell', 'pidof', 'com.brailleme.app']:
                stdout = '123\n'
            elif args == ['shell', 'date', '+%s']:
                stdout = '1000\n'
            elif args == ['shell', 'dumpsys', 'SurfaceFlinger', '--list']:
                stdout = 'RequestedLayerState{com.brailleme.app/com.brailleme.app.MainActivity#246 parentId=245}\n'
                stdout += 'RequestedLayerState{c7b5903 com.brailleme.app/com.brailleme.app.MainActivity#245 parentId=240}\n'
            elif args[:4] == ['shell', 'dumpsys', 'SurfaceFlinger', '--latency']:
                self.assertEqual(args[4], shlex.quote('com.brailleme.app/com.brailleme.app.MainActivity#246'))
                latency_calls[0] += 1
                stdout = '16666666\n0 100 0\n0 200 0\n' + ('0 300 0\n' if latency_calls[0] > 1 else '')
            elif args == ['shell', 'dumpsys', 'media.camera']:
                stdout = 'Camera service diagnostic fixture\n'
            elif args == ['shell', 'dumpsys', 'meminfo', 'com.brailleme.app']:
                stdout = 'TOTAL PSS: 50000 TOTAL RSS: 60000\n'
            elif args[0] == 'logcat':
                stdout = '1000.5 I ReactNativeJS: BRAILLE_WORD_TRACKING_METRICS ' + json.dumps(TRACKING) + '\n'
                stdout += '1000.6 I ReactNativeJS: BRAILLE_WORD_PHOTO_METRICS ' + json.dumps({'schemaVersion': 1, 'captureId': 1, 'elapsedMs': 150, 'status': 'rejected'})
            else:
                raise AssertionError(f'Unexpected command {command}')
            return subprocess.CompletedProcess(command, 0, stdout, '')

        with tempfile.TemporaryDirectory() as output, patch('collect_runtime.subprocess.run', side_effect=run), \
                patch('collect_runtime.time.monotonic', side_effect=lambda: clock[0]), \
                patch('collect_runtime.time.sleep', side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)), patch('builtins.print'):
            collector = RuntimeCollector('adb', 'emulator-5554', 'com.brailleme.app', output)
            self.assertEqual(collector.collect(duration=1), 0)
            summary = json.loads((Path(output) / 'summary.json').read_text())
            metadata = json.loads((Path(output) / 'metadata.json').read_text())
            self.assertEqual(metadata['surfaceLayer'], 'com.brailleme.app/com.brailleme.app.MainActivity#246')
            self.assertEqual(summary['collectionStatus'], 'duration_completed')
            self.assertEqual(summary['acceptanceResult'], 'not_assessed')
            self.assertEqual(summary['presentedFramesObserved'], 1)
            self.assertEqual(summary['trackingReports'], 1)
            self.assertEqual(summary['photoLatenciesMs'], [150])
            self.assertTrue(all(command[:3] == ['adb', '-s', 'emulator-5554'] for command in commands))
            self.assertFalse(any(argument in ('install', 'push', 'input', 'am', '-c', '--latency-clear') for command in commands for argument in command))


if __name__ == '__main__':
    unittest.main()
