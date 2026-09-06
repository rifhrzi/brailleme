"""Guard reporting against denominator drift, stale evidence and misleading gates; no ML."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import word_report as report


class WordReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = report.EVIDENCE/'failed-selected-v15'
        cls.original = {name: report.read_json(cls.directory/name)[0] for name in report.REPORT_NAMES}
        cls.fixtures = {name: (*report.read_json(path), path) for name,path in [
            ('external',report.EVIDENCE/'data-manifests/data__external-words.json.gz'),
            ('validation',report.EVIDENCE/'data-manifests/data__validation-stress-selection.json.gz'),
            ('repetition',report.EVIDENCE/'repetition-fixtures.json'),
            ('rejection',report.EVIDENCE/'rejection-fixtures.json')]}
        cls.model = cls.original['export-report.json']['artifact_sha256']
        cls.checkpoint = cls.original['export-report.json']['source_checkpoint_sha256']

    def setUp(self):
        self.reports = copy.deepcopy(self.original)

    def validate(self):
        return report.validate(self.reports, self.fixtures, self.model, self.checkpoint)

    def test_actual_v15_rows_keep_all_rejections_and_separate_regressions(self):
        result = self.validate()
        self.assertEqual((result['words']['total'],result['words']['accepted'],result['words']['correct']), (607,353,325))
        self.assertEqual(sum(result['reasons'].values()),254)
        self.assertEqual(result['by_length']['12']['total'],2)
        self.assertEqual(result['by_length']['12']['accepted'],0)
        self.assertIsNone(result['by_length']['12']['accepted_accuracy'])
        self.assertEqual(result['negatives']['total'],44)
        self.assertEqual((result['legacy']['total'],result['legacy']['correct']),(66,20))
        self.assertEqual(result['diagnostics']['rejection']['accepted'],2)
        self.assertFalse(result['offline_gates'])

    def test_report_import_and_validation_do_not_load_ml(self):
        # Discovery can already have imported Torch for unrelated loss tests.
        # Verify this report's complete validation in a fresh process instead.
        code = '''
import sys
import word_report as r
directory = r.EVIDENCE/'failed-selected-v15'
reports = {name:r.read_json(directory/name)[0] for name in r.REPORT_NAMES}
fixtures = {name:(*r.read_json(path),path) for name,path in [
    ('external',r.EVIDENCE/'data-manifests/data__external-words.json.gz'),
    ('validation',r.EVIDENCE/'data-manifests/data__validation-stress-selection.json.gz'),
    ('repetition',r.EVIDENCE/'repetition-fixtures.json'),
    ('rejection',r.EVIDENCE/'rejection-fixtures.json')]}
r.validate(reports,fixtures,reports['export-report.json']['artifact_sha256'],
           reports['export-report.json']['source_checkpoint_sha256'])
assert not {'torch','tensorflow','keras'}.intersection(sys.modules)
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(report.__file__).parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_aggregate_must_match_actual_rows(self):
        self.reports['public-benchmark.json']['whole_words']['accepted'] -= 1
        with self.assertRaisesRegex(ValueError,'inconsistent accepted'):
            self.validate()

    def test_length_denominator_cannot_exclude_rejections(self):
        self.reports['public-benchmark.json']['whole_words']['by_length']['12']['total'] = 0
        with self.assertRaisesRegex(ValueError,'inconsistent total'):
            self.validate()

    def test_rejection_reason_summary_cannot_replace_quality_rejections(self):
        reasons = self.reports['public-benchmark.json']['whole_words']['rejection_reasons']
        reasons['low_confidence'] -= 1
        reasons['no_grid'] += 1
        with self.assertRaisesRegex(ValueError,'rejection reasons'):
            self.validate()

    def test_fixture_row_must_not_be_dropped_or_duplicated(self):
        rows = self.reports['public-benchmark.json']['words']
        rows[1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'duplicated'):
            self.validate()

    def test_same_count_but_changed_expected_label_is_rejected(self):
        self.reports['public-benchmark.json']['words'][0]['expected'] = 'AAAAA'
        with self.assertRaisesRegex(ValueError,'frozen fixture'):
            self.validate()

    def test_all44_negatives_are_required_even_if_aggregate_is_lowered(self):
        benchmark = self.reports['public-benchmark.json']
        benchmark['negative_rows'].pop()
        benchmark['negative_regression']['total'] = 43
        with self.assertRaisesRegex(ValueError,'44 negatives'):
            self.validate()

    def test_all66_legacy_blocks_are_required(self):
        self.reports['public-benchmark.json']['legacy_rows'].pop()
        with self.assertRaisesRegex(ValueError,'66 legacy'):
            self.validate()

    def test_export_model_hash_must_match_actual_model(self):
        self.reports['export-report.json']['artifact_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'Model hash differs'):
            self.validate()

    def test_export_validation_fixture_cannot_be_stale(self):
        self.reports['export-report.json']['parity_fixture_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'fixture hash differs'):
            self.validate()

    def test_selected_checkpoint_binding_is_required(self):
        self.reports['checkpoint-selection.json']['selected_checkpoint_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'Checkpoint hash differs'):
            self.validate()

    def test_diagnostics_cannot_use_different_thresholds(self):
        self.reports['rejection-diagnostics.json']['thresholds']['dotConfidence'] = .69
        with self.assertRaisesRegex(ValueError,'Thresholds differ'):
            self.validate()

    def test_diagnostic_correctness_cannot_be_overstated(self):
        self.reports['rejection-diagnostics.json']['rows'][6]['correct'] = True
        with self.assertRaisesRegex(ValueError,'correctness flag differs'):
            self.validate()

    def test_diagnostic_png_bytes_are_checked(self):
        real_digest = report.digest
        with patch.object(report,'digest',side_effect=lambda p: '0'*64 if Path(p).name=='style-01-a.png' else real_digest(p)):
            with self.assertRaisesRegex(ValueError,'fixture PNG changed'):
                self.validate()

    def test_numeric_gate_cannot_be_marked_passed_for_failed_words(self):
        self.reports['public-benchmark.json']['numerical_internal_gates_pass'] = True
        with self.assertRaisesRegex(ValueError,'numerical gate disagrees'):
            self.validate()

    def test_render_default_delivery_is_pending_and_12_letter_precision_is_na(self):
        result = self.validate()
        output = report.EVIDENCE/'unused-test-summary.md'
        text = report.render(result,self.reports,{name:value[1] for name,value in self.fixtures.items()},
                             {name:self.directory/name for name in report.REPORT_NAMES},output,'unit-test',{})
        self.assertIn('| 12 | 2 | 0 | 0 | 0.00% | N/A | 2 |',text)
        self.assertIn('APK internal: belum disertakan',text)
        self.assertIn('Perangkat fisik: pending',text)
        self.assertIn('Gerbang angka offline: GAGAL',text)
        self.assertIn('53.54%',text)
        self.assertIn('92.07%',text)


if __name__ == '__main__':
    unittest.main()
