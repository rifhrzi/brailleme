"""Denominator, frozen-selection and no-repeat guards without model inference."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_v4_test_artifacts import (POSITIVE_GROUPS,acceptance_gates,
    authorize_artifacts,group_metrics,native_parity,sha,write_new)


def pair(source,group,expected,predicted):
    return (dict(source=source,symbol_group=group,expected=expected),
            dict(source=source,group=group,expected=expected,predicted=predicted,
                 reason='' if predicted else 'low_confidence',expected_cells=2,detected_cells=2))


class DenominatorTests(unittest.TestCase):
    def test_rejections_stay_in_positive_denominator_and_invalid_accepts_fail(self):
        pairs=[pair('a','capital_word','AB','AB'),pair('b','capital_word','CD',''),pair('c','rejection','','a')]
        result=group_metrics([p[0] for p in pairs],[p[1] for p in pairs])
        self.assertEqual(result['capital_word']['positive_total'],2)
        self.assertEqual(result['capital_word']['exact_accuracy'],.5)
        self.assertEqual(result['capital_word']['accepted_precision'],1.)
        self.assertEqual(result['rejection']['false_accepts'],1)
        self.assertFalse(acceptance_gates(result)['mandatory_invalid_sequences_all_rejected'])

    def test_single_letters_cannot_hide_failed_multicharacter_or_missing_groups(self):
        pairs=[]
        for group in POSITIVE_GROUPS:
            pairs += [pair(group+str(i),group,'a','a') for i in range(9)]
            pairs.append(pair(group+'long',group,'abc',''))
        pairs.append(pair('reject','rejection','',''))
        groups=group_metrics([p[0] for p in pairs],[p[1] for p in pairs])
        self.assertEqual(groups['capital_word']['exact_accuracy'],.9)
        gates=acceptance_gates(groups)
        self.assertTrue(gates['coverage_complete'])
        self.assertFalse(gates['exact80_each_positive_group_and_multichar'])
        del groups['punctuation']
        self.assertFalse(acceptance_gates(groups)['coverage_complete'])

    def test_exact80_is_separate_from_precision_and_invalid_guards(self):
        pairs=[]
        for group in POSITIVE_GROUPS:
            pairs.extend(pair(group+str(i),group,'abc','abc' if i<8 else 'wrong') for i in range(10))
        pairs.append(pair('reject','rejection','','bad'))
        result=acceptance_gates(group_metrics([p[0] for p in pairs],[p[1] for p in pairs]))
        self.assertTrue(result['exact80_each_positive_group_and_multichar'])
        self.assertFalse(result['accepted_precision95_each_positive_group_and_multichar'])
        self.assertFalse(result['mandatory_invalid_sequences_all_rejected'])
        self.assertFalse(result['combined_acceptance_passed'])

    def test_each_punctuation_has_its_own_denominator(self):
        pairs=[pair('a','punctuation','ab!','ab!'),pair('b','punctuation','ab?',''),pair('c','punctuation','(ab)','(ab)')]
        detail=group_metrics([p[0] for p in pairs],[p[1] for p in pairs])['punctuation']['by_punctuation']
        self.assertEqual(detail['!']['positive_total'],1)
        self.assertEqual(detail['?']['exact_accuracy'],0.)
        self.assertEqual(detail['(']['positive_total'],1)
        self.assertIsNone(detail['.']['exact_accuracy'])

    def test_missing_reordered_or_duplicate_decisions_cannot_pass(self):
        pairs=[pair('a','numbers','12','12'),pair('b','numbers','34','34')]
        rows,decisions=[p[0] for p in pairs],[p[1] for p in pairs]
        with self.assertRaises(ValueError):group_metrics(rows,decisions[:1])
        with self.assertRaises(ValueError):group_metrics(rows,list(reversed(decisions)))
        with self.assertRaises(ValueError):group_metrics(rows+[rows[0]],decisions+[decisions[0]])

    def test_native_parity_checks_reason_and_detected_cells_not_just_text(self):
        decision=pair('a','numbers','12','12')[1]
        changed=dict(decision,detected_cells=3,reason='different')
        report=native_parity([changed],[decision])
        self.assertEqual(report['status'],'failed')
        self.assertEqual(set(report['mismatches'][0]['differences']),{'reason','detected_cells'})
        with self.assertRaises(ValueError):native_parity([], [decision])


class BindingTests(unittest.TestCase):
    def fixture(self,root):
        run,export=root/'run-final',root/'export-final'
        run.mkdir();export.mkdir()
        def save(path,value):path.write_text(json.dumps(value))
        baseline=root/'deployed-baseline.tflite';baseline.write_bytes(b'unchanged baseline')
        checkpoint=run/'best-word.keras';checkpoint.write_bytes(b'chosen checkpoint')
        model=export/'word-reader.tflite';model.write_bytes(b'converted checkpoint')
        arrays=export/'torch-oracle.npz';arrays.write_bytes(b'validation oracle')
        for name,value in (('validation-symbols.json',{'split':'validation'}),('test-symbols.json',{'split':'test'}),('test-image-hashes.json',{})):
            save(root/name,value)
        source_dir=Path(__file__).resolve().parents[1]/'scripts'
        config=dict(thresholds={'proposalScore':.25},input_sha256={str(root/name):sha(root/name) for name in ('validation-symbols.json','test-symbols.json','test-image-hashes.json')},source_sha256={name:sha(source_dir/name) for name in ('word_decode_id_v4.py','word_symbols_id_v4.py','word_symbols.py','word_data.py')})
        save(run/'training-config.json',config)
        selection=dict(sha256=sha(checkpoint),selection_scope='validation_only',test_opened=False,training_config_sha256=sha(run/'training-config.json'))
        save(run/'selection.json',selection)
        reservation=dict(run=str(run),checkpoint_sha256=sha(checkpoint),selection_sha256=sha(run/'selection.json'))
        save(root/'test-evaluation-reservation.json',reservation)
        save(run/'test-evaluation.json',dict(checkpoint_sha256=sha(checkpoint),reservation=reservation,thresholds=config['thresholds'],test_manifest_sha256=sha(root/'test-symbols.json'),decisions=[]))
        save(root/'final-test-evaluation.json',dict(run=str(run),checkpoint_sha256=sha(checkpoint),report_sha256=sha(run/'test-evaluation.json')))
        oracle=dict(checkpoint_sha256=sha(checkpoint),thresholds=config['thresholds'],validation_sha256=sha(root/'validation-symbols.json'),oracle_sha256=sha(arrays),source_sha256=config['source_sha256'])
        save(export/'torch-oracle.json',oracle)
        save(export/'export-parity.json',dict(oracle,status='passed',allclose=True,all_finite=True,decision_mismatches=[],quality_effective_decision_mismatches=[],tflite_sha256=sha(model),bytes=model.stat().st_size))
        save(run/'protected-before.json',{str(baseline):sha(baseline)})
        return root,run,export,baseline

    def test_no_reservation_blocks_before_test_files_are_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run-final';run.mkdir()
            with patch('word_symbol_v4_test_artifacts.read') as reader:
                with self.assertRaisesRegex(ValueError,'Completed native TEST'):
                    authorize_artifacts(root,run,root/'export',root/'baseline')
                reader.assert_not_called()

    def test_completed_reservation_export_and_checkpoint_binding_pass_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary))
            self.assertEqual(authorize_artifacts(*args)['selection']['selection_scope'],'validation_only')

    def test_different_reserved_run_changed_checkpoint_or_failed_export_is_denied(self):
        for mutation in ('run','checkpoint','export','baseline','thresholds','test'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                args=self.fixture(Path(temporary));root,run,export,baseline=args
                if mutation=='run':
                    path=root/'test-evaluation-reservation.json';value=json.loads(path.read_text());value['run']=str(root/'other');path.write_text(json.dumps(value))
                elif mutation=='checkpoint':(run/'best-word.keras').write_bytes(b'another model')
                elif mutation=='export':
                    path=export/'export-parity.json';value=json.loads(path.read_text());value['status']='failed';path.write_text(json.dumps(value))
                elif mutation=='baseline':baseline.write_bytes(b'changed baseline')
                elif mutation=='thresholds':
                    path=run/'test-evaluation.json';value=json.loads(path.read_text());value['thresholds']={'proposalScore':.1};path.write_text(json.dumps(value))
                else:(root/'test-symbols.json').write_text('{}')
                with self.assertRaises(ValueError):authorize_artifacts(*args)

    def test_started_or_finished_comparison_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.fixture(Path(temporary));run=args[1]
            marker=run/'test-artifact-comparison-started.json'
            write_new(marker,{'a':1})
            with self.assertRaises(FileExistsError):write_new(marker,{'a':2})
            self.assertEqual(json.loads(marker.read_text()),{'a':1})
            with self.assertRaisesRegex(ValueError,'never silently repeat'):authorize_artifacts(*args)


if __name__=='__main__':
    unittest.main()
