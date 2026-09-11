"""New numeric support must not be mislabeled as rejection failure."""
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_rejection_fixtures import candidates
from word_symbols import LATIN_MASKS


class SymbolDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(1000,500),'white')
        self.supported=set(LATIN_MASKS)|{32,60}

    def boxes(self,count):
        return np.asarray([[250+i*30,220,270+i*30,260] for i in range(count)],float)

    def test_66_and_67_are_not_unsupported_pattern_negatives(self):
        for masks in ([60,11,11],[60,11,27]):
            self.assertTrue(list(candidates(self.image,self.boxes(3),np.asarray(masks),'unsupported_pattern')))
            self.assertFalse(list(candidates(self.image,self.boxes(3),np.asarray(masks),'unsupported_pattern',self.supported)))

    def test_numeric_indicator_does_not_count_toward_12_character_limit(self):
        masks=np.asarray([60]+[1]*12)
        self.assertTrue(list(candidates(self.image,self.boxes(13),masks,'over_limit')))
        self.assertFalse(list(candidates(self.image,self.boxes(13),masks,'over_limit',self.supported)))


if __name__=='__main__':
    unittest.main()
