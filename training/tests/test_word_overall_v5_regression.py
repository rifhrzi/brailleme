"""CPU-only historical regression bookkeeping; synthetic records, no inference."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import word_overall_v5_evaluate as evaluate

THRESHOLDS=dict(proposalScore=.25,acceptScore=.25,dotConfidence=.7)


def cached_rows():
    result={}
    for name,count in (('external607',607),('legacy405',405),('negative44',44)):
        rows=[]
        for index in range(count):
            expected='' if name=='negative44' else 'a'
            good=(name=='external607' and index<606) or (name=='legacy405' and index<355)
            rows.append(dict(source=f'{name}/page',image=f'{name}/{index}.png',expected=expected,
                             selection=[.1,.2,.9,.8],image_sha256='image hash',baseline=dict(text='a' if good else '',reason='',cells=int(good))))
        result[name]=rows
    return result


def cache_document():
    return dict(baseline_sha256=evaluate.OLD_BASELINE_SHA256,thresholds=THRESHOLDS,
                decoder_version='six-dot-id-v4-experimental',input_sha256={'bound-old-file':'digest'},
                results={name:dict(rows=rows) for name,rows in cached_rows().items()})


class RegressionTests(unittest.TestCase):
    def test_cached_baseline_is_hash_bound_and_has_three_fixed_denominators(self):
        document=cache_document();path=Path('existing-bound-report.json')
        with patch.object(evaluate,'read',return_value=document),patch.object(evaluate,'verify_hashes') as check:
            result=evaluate.load_cached_regression(path,THRESHOLDS)
        self.assertEqual({name:len(rows) for name,rows in result.items()},dict(external607=607,legacy405=405,negative44=44))
        self.assertEqual(check.call_args_list,[call({path:evaluate.OLD_REGRESSION_SHA256}),call(document['input_sha256'])])

    def test_wrong_baseline_decoder_thresholds_or_denominators_fail(self):
        source=cache_document()
        alternatives=[]
        for field,value in [('baseline_sha256','wrong'),('decoder_version','old parser'),('thresholds',dict(THRESHOLDS,dotConfidence=.6))]:
            item=deepcopy(source);item[field]=value;alternatives.append(item)
        item=deepcopy(source);item['results']['external607']['rows'].pop();alternatives.append(item)
        item=deepcopy(source);item['results']['legacy405']['rows'][1]=item['results']['legacy405']['rows'][0];alternatives.append(item)
        item=deepcopy(source);item['results']['negative44']['rows'][0]['baseline']['text']=None;alternatives.append(item)
        for index,document in enumerate(alternatives):
            with self.subTest(index=index),patch.object(evaluate,'read',return_value=document),patch.object(evaluate,'verify_hashes'):
                with self.assertRaises(ValueError):evaluate.load_cached_regression(Path('report.json'),THRESHOLDS)

    def test_reservation_failure_prevents_regression_engine_creation(self):
        method=dict(primary_engine='tflite')
        with patch.object(evaluate,'verify_method',return_value=method),patch.object(evaluate,'reserve_stage',side_effect=ValueError('stage already reserved')),patch.object(evaluate,'make_predictor') as factory:
            with self.assertRaisesRegex(ValueError,'already reserved'):
                evaluate.regression(SimpleNamespace(experiment=Path('.'),device='cpu'))
            factory.assert_not_called()

    def test_real_stage_bookkeeping_separate_denominators_precision_and_loss_gain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            method=dict(checkpoint_sha256='checkpoint',tflite_sha256='tflite',primary_engine='torch',selected_policy='cascade',
                        regression_reference='old-report.json',thresholds=THRESHOLDS)
            method_path=root/'final-method-selection.json';method_path.write_text(json.dumps(method))
            original=method_path.read_bytes()
            identity=dict(method_sha256=evaluate.sha(method_path),checkpoint_sha256='checkpoint',tflite_sha256='tflite',primary_engine='torch',selected_policy='cascade')
            (root/'test-evaluation-reservation.json').write_text(json.dumps(identity))
            (root/'final-test-evaluation.json').write_text(json.dumps(dict(method_sha256=evaluate.sha(method_path))))
            events=[]
            def factory(engine,frozen,device):
                self.assertEqual(engine,'tflite')
                self.assertTrue((root/'regression-tflite-started.json').is_file())
                self.assertEqual(frozen,method);events.append('engine')
                return object(),dict(engine='synthetic_test_double',device='cpu')
            def infer(rows,predictor,frozen,*,allow_resize=False):
                self.assertTrue(allow_resize);self.assertEqual(frozen['selected_policy'],'cascade')
                name=rows[0]['source'].split('/')[0];events.append(name)
                outputs=[]
                for index,row in enumerate(rows):
                    if name=='external607': text='' if index==0 else 'b' if index==1 else 'a'
                    elif name=='legacy405': text='a' if index<356 else ''
                    else: text='a' if index==0 else ''
                    outputs.append(dict(source=row['source'],expected=row['expected'],predicted=text,reason='',route='matched_logit_four'))
                return outputs,dict(test_double=True)
            args=SimpleNamespace(experiment=root,device='cpu')
            with patch.object(evaluate,'verify_method',return_value=method) as checked,patch.object(evaluate,'load_cached_regression',return_value=cached_rows()),patch.object(evaluate,'make_predictor',side_effect=factory),patch.object(evaluate,'infer_rows',side_effect=infer):
                with redirect_stdout(io.StringIO()):
                    evaluate.regression(args)
                self.assertEqual(checked.call_count,2)
                with self.assertRaises(FileExistsError):evaluate.regression(args)
            report=json.loads((root/'historical-regression.json').read_text())
            self.assertEqual(events,['engine','external607','legacy405','negative44'])
            self.assertEqual(method_path.read_bytes(),original)
            self.assertNotIn('target80_met',report)
            self.assertNotIn('overall_exact_accuracy',report)
            public=report['results']['external607'];legacy=report['results']['legacy405'];negative=report['results']['negative44']
            self.assertEqual(public['candidate']['positive_total'],607)
            self.assertEqual((public['candidate']['exact_correct'],public['candidate']['accepted_positive_total']),(605,606))
            self.assertAlmostEqual(public['candidate']['overall_exact_accuracy'],605/607)
            self.assertAlmostEqual(public['candidate']['accepted_precision'],605/606)
            self.assertEqual((public['lost'],public['gained']),(2,1));self.assertTrue(public['flags']['exact_regressed'])
            self.assertEqual(legacy['candidate']['positive_total'],405)
            self.assertEqual((legacy['lost'],legacy['gained']),(0,1));self.assertFalse(legacy['regression_observed'])
            self.assertEqual(negative['candidate']['invalid_total'],44)
            self.assertEqual(negative['candidate']['positive_total'],0)
            self.assertIsNone(negative['candidate']['overall_exact_accuracy'])
            self.assertEqual(negative['candidate']['invalid_false_accepts'],1)
            self.assertTrue(negative['flags']['invalid_false_accepts_regressed'])


if __name__=='__main__':unittest.main()
