"""Guard the resumed-training objective that previously changed silently."""
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('KERAS_TORCH_DEVICE', 'cpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from word_train import parse_args


class TrainingConfigurationTest(unittest.TestCase):
    base = ['--data', 'unused-data', '--output', 'unused-output']

    def test_resume_requires_explicit_geometry_loss(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            parse_args(self.base+['--resume', 'parent.keras'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('--resume requires explicit --geometry-loss', errors.getvalue())

    def test_parent_relative_objective_is_preserved(self):
        args = parse_args(self.base+['--resume', 'parent.keras', '--geometry-loss', 'relative_size_huber'])
        self.assertEqual(args.geometry_loss, 'relative_size_huber')

    def test_new_model_keeps_original_default(self):
        self.assertEqual(parse_args(self.base).geometry_loss, 'normalized_huber')

    def test_vertical_stretch_is_an_explicit_illumination_ablation(self):
        self.assertFalse(parse_args(self.base).vertical_stretch_augmentation)
        enabled = self.base+['--illumination-augmentation', '--vertical-stretch-augmentation']
        self.assertTrue(parse_args(enabled).vertical_stretch_augmentation)
        for argv in (self.base+['--vertical-stretch-augmentation'], enabled+['--horizontal-mirror-augmentation']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_bn_affine_requires_paired_proofs_and_preserves_the_parent(self):
        resume = self.base+['--resume', 'initial.keras', '--geometry-loss', 'relative_size_huber']
        proofs = ['--bn-affine-initialization', 'init.json', '--bn-affine-conversion', 'proof.json']
        self.assertIsNone(parse_args(resume).bn_affine_initialization)
        self.assertEqual(parse_args(resume+proofs).bn_affine_conversion, Path('proof.json'))
        for argv in (resume+proofs[:2], resume+proofs[2:], self.base+proofs,
                     resume+proofs+['--image-normalization', 'channel_gray_std05'],
                     resume+proofs+['--geometry-head-initialization', 'geometry.json']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_stride2_requires_paired_proofs_and_preserves_the_base_ablation(self):
        resume = self.base+['--resume', 'initial.keras', '--geometry-loss', 'relative_size_huber']
        proofs = ['--fine-branch-initialization', 'init.json', '--fine-branch-conversion', 'proof.json']
        self.assertIsNone(parse_args(resume).fine_branch_initialization)
        self.assertEqual(parse_args(resume+proofs).fine_branch_conversion, Path('proof.json'))
        for argv in (resume+proofs[:2], resume+proofs[2:], self.base+proofs,
                     resume+proofs+['--image-normalization', 'channel_gray_std05'],
                     resume+proofs+['--geometry-head-initialization', 'geometry.json'],
                     resume+proofs+['--bn-affine-initialization', 'bn.json', '--bn-affine-conversion', 'bn-proof.json']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == '__main__':
    unittest.main()
