"""Reject contradictory lineage, tensor and diagnostic conversion evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SOURCE))
from word_fine_validation import digest, require_fine_reports, validated_fine_branch
from word_bn_affine_parity import decision_rows
from word_decode import DEFAULT_THRESHOLDS


class FineProofTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.init_path, self.proof_path = self.root/'init.json', self.root/'proof.json'
        self.initial = {'status': 'passed', 'ordinary_frozen_batchnorm_layers': 52,
                        'inherited_array_count': 280, 'array_count': 283, 'parameters': 830539,
                        'zero_branch_gradients': {'loss': .1, 'lateral_l1': .2, 'depthwise_l1': 0.},
                        'after_isolated_lateral_step_gradients': {'loss': .09, 'lateral_l1': .2, 'depthwise_l1': .03}}
        for key in ('parent_checkpoint', 'initial_checkpoint', 'nonzero_diagnostic_checkpoint'):
            path = self.root/(key+'.keras'); path.write_bytes(key.encode())
            self.initial[key] = str(path)
            self.initial['nonzero_diagnostic_sha256' if key == 'nonzero_diagnostic_checkpoint' else key+'_sha256'] = digest(path)
        for key in ('nonzero_diagnostic_must_never_be_resumed', 'all_inherited_arrays_and_trainability_equal',
                    'initial_train_eval_parent_outputs_equal', 'initial_save_reload_values_and_outputs_equal',
                    'diagnostic_save_reload_outputs_equal', 'diagnostic_only_new_lateral_changed', 'saved_parent_and_initial_unchanged'):
            self.initial[key] = True
        self.initial['source_sha256'] = {name: digest(SOURCE/name) for name in ('word_fine_branch.py', 'word_fine_init.py', 'word_model.py', 'word_data.py')}
        static = self.root/'static.json'
        static.write_text(json.dumps({'parent': {'sha256': self.initial['parent_checkpoint_sha256']}}))
        self.initial.update(static_audit=str(static), static_audit_sha256=digest(static))
        selected = self.root/'selected.json'
        selected.write_text(json.dumps({'checkpoint_sha256': self.initial['parent_checkpoint_sha256'], 'thresholds': DEFAULT_THRESHOLDS}))
        images = np.zeros((6, 192, 480, 3), np.float32)
        values = np.full((6, 48, 120, 11), .05, np.float32)
        changed = values.copy(); changed[0, 0, 0, 5] += .005
        oracle = self.root/'initialization-oracle.npz'
        np.savez_compressed(oracle, images=images, parent=values, initial=values, nonzero=changed)
        delta = float(np.max(np.abs(changed-values)))
        self.initial.update(oracle_sha256=digest(oracle), nonzero_max_output_delta=delta)
        self.proof = {'status': 'passed', 'oracle_sha256': digest(oracle),
                      'selected_validation_report': str(selected), 'selected_validation_sha256': digest(selected),
                      'thresholds': DEFAULT_THRESHOLDS, 'absolute_tolerance': 1e-4, 'relative_tolerance': 1e-4,
                      'nonzero_tf_output_delta': delta, 'nonzero_tflite_output_delta': delta,
                      'conversion_probes_must_not_be_activated_or_resumed': True, 'trained_candidate_requires_full405_parity': True,
                      'torch_parent_decisions': decision_rows(values, images, DEFAULT_THRESHOLDS), 'probes': {}}
        self.proof['source_sha256'] = {name: digest(SOURCE/name) for name in ('word_fine_parity.py', 'word_bn_affine_parity.py', 'word_decode.py', 'word_benchmark.py', 'benchmark_angelina.py', 'word_data.py')}
        for label, output, key in [('initial', values, 'initial_checkpoint_sha256'), ('nonzero', changed, 'nonzero_diagnostic_sha256')]:
            artifact = self.root/f'{label}-conversion-probe.tflite'; artifact.write_bytes(label.encode())
            row = {'checkpoint_sha256': self.initial[key], 'samples': 6, 'operators': ['CONV_2D', 'DEPTHWISE_CONV_2D'],
                   'artifact_sha256': digest(artifact), 'decisions': decision_rows(output, images, DEFAULT_THRESHOLDS),
                   'input': {'count': 1, 'shape': [1, 192, 480, 3], 'dtype': 'float32'},
                   'output': {'count': 1, 'shape': [1, 48, 120, 11], 'dtype': 'float32'}}
            for name in ('builtins_only', 'all_finite', 'eager_train_eval_exact', 'all_moments_and_weights_unchanged', 'raw_and_quality_effective_decisions_equal'):
                row[name] = True
            for name in ('torch_vs_tf_max_abs', 'tf_vs_tflite_max_abs', 'torch_vs_tflite_max_abs', 'compiled_vs_eager_max_abs'):
                row[name] = 1.5e-4  # valid upper bound for atol+rtol on probabilities
            self.proof['probes'][label] = row

    def write(self, initial=None, proof=None):
        initial, proof = initial or self.initial, deepcopy(proof or self.proof)
        self.init_path.write_text(json.dumps(initial))
        proof['initialization_report_sha256'] = digest(self.init_path)
        self.proof_path.write_text(json.dumps(proof))

    def test_consistent_proof_is_accepted_and_diagnostic_cannot_resume(self):
        self.write()
        require_fine_reports(Path(self.initial['initial_checkpoint']), self.init_path, self.proof_path)
        with self.assertRaisesRegex(ValueError, 'exact initial'):
            require_fine_reports(Path(self.initial['nonzero_diagnostic_checkpoint']), self.init_path, self.proof_path)

    def test_declared_pass_cannot_hide_numeric_shape_or_decision_failures(self):
        mutations = [lambda p: p.update(nonzero_tflite_output_delta=1e-6),
                     lambda p: p.update(nonzero_tf_output_delta=float('nan')),
                     lambda p: p['probes']['nonzero'].update(torch_vs_tflite_max_abs=.5),
                     lambda p: p['probes']['initial'].update(compiled_vs_eager_max_abs=float('nan')),
                     lambda p: p['probes']['nonzero']['output'].update(dtype='float16'),
                     lambda p: p['probes']['initial']['input'].update(count=2),
                     lambda p: p['probes']['nonzero'].update(operators=['CUSTOM']),
                     lambda p: p['probes']['initial'].update(all_moments_and_weights_unchanged=False),
                     lambda p: p['probes']['nonzero']['decisions'][0]['raw'].update(reason='clipped'),
                     lambda p: p['source_sha256'].pop('word_bn_affine_parity.py'),
                     lambda p: p.update(trained_candidate_requires_full405_parity=False)]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                proof = deepcopy(self.proof); mutate(proof); self.write(proof=proof)
                with self.assertRaises(ValueError):
                    require_fine_reports(Path(self.initial['initial_checkpoint']), self.init_path, self.proof_path)

    def test_unproven_model_cannot_bypass_opt_in(self):
        class Layer:
            name = 'fpn_2_zero'
        class Model:
            layers = [Layer()]
        with self.assertRaisesRegex(ValueError, 'both explicit'):
            validated_fine_branch(Model(), None, None, None)
        Model.layers = []
        self.assertEqual(validated_fine_branch(Model(), None, None, None), {'enabled': False})


if __name__ == '__main__':
    unittest.main()
