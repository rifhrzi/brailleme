"""Check truthful normalization identity and isolated initialization controls."""
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('KERAS_BACKEND','torch');os.environ.setdefault('KERAS_TORCH_DEVICE','cpu')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_normalization_transition import training_normalization_metadata,validated_normalization_transition
from word_train import parse_args


class NormalizationTransitionTests(unittest.TestCase):
    def test_std05_metadata_remains_byte_identical(self):
        identity,text=training_normalization_metadata({'kind':'per_channel_standardize_grayscale','targetStd':.05,'stdFloor':.001})
        self.assertEqual(identity,'channel_gray_std05')
        self.assertEqual(text,'RGB float32 /255; internal per-image per-channel mean and population std (ddof=0, std floor .001), mean standardized channels, .5+.05*gray clipped [0,1], repeat RGB, then Rescaling(2,-1)')

    def test_std15_metadata_reports_actual_scale(self):
        identity,text=training_normalization_metadata({'kind':'per_channel_standardize_grayscale','targetStd':.15,'stdFloor':.001})
        self.assertEqual(identity,'channel_gray_std15');self.assertIn('.5+.15*gray',text);self.assertNotIn('.05',text)

    def test_raw_metadata_remains_identical(self):
        self.assertEqual(training_normalization_metadata({'kind':'none'}),('raw','RGB float32 /255; internal Rescaling(2,-1)'))

    def test_nonfinite_scale_cannot_be_reported(self):
        for value in (float('nan'),float('inf'),0.,-.15):
            with self.assertRaises(ValueError):training_normalization_metadata({'kind':'per_channel_standardize_grayscale','targetStd':value,'stdFloor':.001})

    def test_nondefault_normalization_requires_explicit_proof(self):
        self.assertEqual(validated_normalization_transition(None,None,None,None,{'kind':'none'}),{'enabled':False})
        self.assertEqual(validated_normalization_transition(None,None,None,None,{'targetStd':.05}),{'enabled':False})
        with self.assertRaises(ValueError):validated_normalization_transition(None,None,None,None,{'targetStd':.15})
        for a,b in ((None,'proof.json'),('initial.json',None)):
            with self.assertRaises(ValueError):validated_normalization_transition(None,None,a,b,{'targetStd':.15})

    def test_cli_requires_exact_pair_and_isolated_profile(self):
        base=['--data','unused','--output','unused','--resume','initial.keras','--geometry-loss','relative_size_huber',
            '--illumination-augmentation','--dsbi-visible-partial-supervision','--dsbi-point-metadata','points.json',
            '--base-train-manifest','base.json','--supplemental-source-index','source.json','--supplemental-negative-repeat','2']
        pair=['--normalization-initialization','initial.json','--normalization-runtime-proof','proof.json']
        self.assertIsNone(parse_args(base).normalization_initialization)
        self.assertEqual(parse_args(base+pair).normalization_runtime_proof,Path('proof.json'))
        for argv in (base+pair[:2],base+pair[2:]):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(argv)
        for extra in (['--anchor-zoom-cap'],['--horizontal-aspect-augmentation'],['--vertical-stretch-augmentation'],['--horizontal-mirror-augmentation'],['--image-normalization','raw'],['--camera-augmentation'],['--half-turn-augmentation'],['--polarity-augmentation'],['--partial-geometry-supervision'],['--geometry-head-initialization','geometry.json'],['--bn-affine-initialization','bn.json','--bn-affine-conversion','bn-proof.json'],['--fine-branch-initialization','fine.json','--fine-branch-conversion','fine-proof.json']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(base+pair+extra)


if __name__=='__main__':unittest.main()
