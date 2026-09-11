"""Meaningful offline fixtures for counter resets and complete photo latency accounting."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('analyze-runtime.py')
SPEC = importlib.util.spec_from_file_location('runtime_analysis', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def tracking(epoch, received, buckets, elapsed=10000):
    counts = {key: 0 for key in analysis.COUNTERS}
    counts.update(received=received, admitted=sum(buckets), completed=sum(buckets))
    return {'tag': analysis.TRACKING, 'deviceEpochSeconds': epoch, 'elapsedSeconds': epoch,
            'metrics': {'schemaVersion': 1, 'elapsedMs': elapsed, **counts, 'meanMs': 20, 'maxMs': 70,
                        'latencyBuckets': buckets, 'latencyUpperBoundsMs': analysis.BOUNDS}}


def photo(epoch, latency, status='success', capture=1):
    return {'tag': analysis.PHOTO, 'deviceEpochSeconds': epoch, 'elapsedSeconds': epoch,
            'metrics': {'schemaVersion': 1, 'captureId': capture, 'elapsedMs': latency, 'status': status}}


class AnalysisTests(unittest.TestCase):
    def test_cumulative_snapshots_are_subtracted_per_reset_not_summed(self):
        rows = [tracking(100, 100, [4, 1, 0, 0, 0, 0, 0]),
                tracking(110, 200, [7, 3, 0, 0, 0, 0, 0], 20000),
                tracking(130, 50, [1, 1, 0, 0, 0, 0, 0]),
                tracking(140, 100, [2, 3, 0, 0, 0, 0, 0], 20000)]
        issues = []
        actual = analysis.tracking_summary(rows, issues)
        self.assertEqual(issues, [])
        self.assertEqual(actual['inferred_counter_resets'], 1)
        self.assertEqual(actual['observed_delta_counts_lower_bound']['received'], 150)
        self.assertEqual(actual['observed_delta_buckets_lower_bound'], [4, 4, 0, 0, 0, 0, 0])
        self.assertEqual(actual['callback_rate_lower_bound_hz'], 150 / 40)
        self.assertEqual(actual['segments'][0]['last_cumulative_buckets'], [7, 3, 0, 0, 0, 0, 0])
        self.assertEqual(actual['observed_delta_histogram_p95_bounds']['lower_ms'], 50)
        self.assertEqual(actual['observed_delta_histogram_p95_bounds']['upper_ms'], 100)

    def test_reporting_clock_restart_does_not_create_counter_reset(self):
        rows = [tracking(100, 100, [1, 0, 0, 0, 0, 0, 0], 60000),
                tracking(110, 200, [3, 0, 0, 0, 0, 0, 0], 10000)]
        actual = analysis.tracking_summary(rows, [])
        self.assertEqual(actual['inferred_counter_resets'], 0)
        self.assertEqual(actual['reporting_elapsed_clock_restarts_without_counter_reset'], 1)
        self.assertEqual(actual['callback_rate_lower_bound_hz'], 10)

    def test_zero_or_backwards_device_time_produces_no_rate(self):
        for end in (99, 100):
            with self.subTest(end=end):
                issues = []
                actual = analysis.tracking_summary([tracking(100, 1, [1, 0, 0, 0, 0, 0, 0]),
                                                     tracking(end, 5, [2, 0, 0, 0, 0, 0, 0])], issues)
                self.assertIsNone(actual['callback_rate_lower_bound_hz'])
                self.assertIsNone(actual['segments'][0]['callback_rate_lower_bound_hz'])
                self.assertTrue(issues)

    def test_photo_p95_includes_rejected_and_error_and_repeated_capture_ids(self):
        rows = [photo(index, 10) for index in range(18)] + [photo(18, 1000, 'rejected'), photo(19, 2000, 'error')]
        summary = {'photoReports': 20, 'trackingReports': 0, 'memorySamples': 0, 'surfaceSamples': 0,
                   'photoLatenciesMs': [10] * 18 + [1000, 2000], 'collectionStatus': 'duration_completed', 'elapsedSeconds': 600}
        actual = analysis.summarize(rows + [copy.deepcopy(rows[0])], {'requestedSeconds': 600}, summary, [])
        self.assertEqual(actual['photos']['observed_terminal_records'], 20)
        self.assertEqual(actual['photos']['p95_rank'], 19)
        self.assertEqual(actual['photos']['nearest_rank_p95_all_statuses_ms'], 1000)
        self.assertEqual(actual['photos']['max_all_statuses_ms'], 2000)
        self.assertEqual(actual['photos']['statuses']['success']['count'], 18)
        self.assertEqual(actual['photos']['statuses']['rejected']['count'], 1)
        self.assertEqual(actual['photos']['statuses']['error']['count'], 1)
        self.assertEqual(actual['collection']['duplicate_metric_records_ignored'], 1)

    def test_single_report_has_no_observed_delta_or_rate(self):
        actual = analysis.tracking_summary([tracking(100, 100, [2, 0, 0, 0, 0, 0, 0])], [])
        self.assertEqual(actual['observed_delta_counts_lower_bound']['received'], 0)
        self.assertIsNone(actual['observed_delta_histogram_p95_bounds'])
        self.assertIsNone(actual['callback_rate_lower_bound_hz'])
        self.assertEqual(analysis.histogram_percentile([0, 0, 0, 0, 0, 0, 2])['upper_ms'], None)

    def test_invalid_histogram_and_malformed_rows_are_visible(self):
        bad = tracking(100, 5, [1, 0, 0, 0, 0, 0, 0])
        bad['metrics']['completed'] = 3
        issues = []
        tracks, photos, _ = analysis.metric_rows([bad, photo(200, -1), {'unexpected': True}], issues)
        self.assertEqual((tracks, photos), ([], []))
        self.assertEqual(len(issues), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.jsonl'
            path.write_text('{"memory":{}}\n{"broken"\n', encoding='utf-8')
            self.assertEqual(len(analysis.load_lines(path, issues)), 1)
            self.assertIn('samples.jsonl:2:', issues[-1])

    def test_pss_extremes_surface_proxy_and_incomplete_duration(self):
        rows = [{'memory': {'elapsedSeconds': 20, 'pssKiB': 125}},
                {'memory': {'elapsedSeconds': 0, 'pssKiB': 100}},
                {'memory': {'elapsedSeconds': 10, 'pssKiB': 150}},
                {'elapsedSeconds': 1, 'surface': {'newPresentedFrames': 5, 'presentedFramesObserved': 5, 'refreshPeriodNs': 16666667}},
                {'elapsedSeconds': 2, 'surface': {'newPresentedFrames': 8, 'presentedFramesObserved': 13, 'refreshPeriodNs': 16666667}}]
        issues = []
        actual = analysis.summarize(rows, {'requestedSeconds': 600}, {'elapsedSeconds': 20,
                                    'collectionStatus': 'app_process_changed', 'surfaceHistoryGaps': 2}, issues)
        self.assertEqual(actual['memory']['first_pss_kib'], 100)
        self.assertEqual(actual['memory']['last_pss_kib'], 125)
        self.assertEqual(actual['memory']['max_pss_kib'], 150)
        self.assertEqual(actual['surface']['observed_composition_fps_proxy'], .65)
        self.assertEqual(actual['surface']['metric_kind'], 'surface_composition_proxy')
        self.assertTrue(any('not reached' in value for value in issues))
        self.assertTrue(any('app_process_changed' in value for value in issues))

    def test_cli_binds_apk_hash_and_never_overwrites_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, actions, apk = root / 'runtime', root / 'actions', root / 'actual.apk'
            runtime.mkdir()
            actions.mkdir()
            apk.write_bytes(b'fixture-apk-not-an-installed-artifact')
            output = root / 'analysis.json'
            args = ['--runtime', str(runtime), '--actions', str(actions), '--apk', str(apk),
                    '--apk-sha256', '0' * 64, '--output', str(output)]
            with self.assertRaises(SystemExit):
                analysis.main(args)
            self.assertFalse(output.exists())
            args[7] = analysis.sha(apk)
            self.assertEqual(analysis.main(args), 1)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['acceptanceResult'], 'not_assessed')
            self.assertFalse(report['physical_camera_validated'])
            self.assertFalse(report['apk']['installed_apk_verified_by_this_analyzer'])
            before = output.read_bytes()
            with self.assertRaises(SystemExit):
                analysis.main(args)
            self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
