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


if __name__ == '__main__':
    unittest.main()
