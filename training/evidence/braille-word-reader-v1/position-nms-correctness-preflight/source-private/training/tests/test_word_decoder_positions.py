"""Real neighboring cells must not compete before positional box NMS."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_decode import decode, prepare_calibration_outputs, DEFAULT_THRESHOLDS

THRESHOLDS=dict(DEFAULT_THRESHOLDS,proposalScore=.25,acceptScore=.25,dotConfidence=.7)


def tensor(cells):
    output=np.zeros((48,120,11),np.float32)
    for cx,width,mask,score in cells:
        gx=int(cx//4)
        output[24,gx]=[score,cx/4-gx,0,width/480,12/192]+[float(bool(mask&(1<<i))) for i in range(6)]
    return output


class WordDecoderPositionTests(unittest.TestCase):
    def test_disjoint_adjacent_letters_and_repeats_with_equal_or_unequal_scores(self):
        for masks,text in [([1,3],'AB'),([1,1],'AA')]:
            for scores in [(1,1),(.99,.98),(.98,.99)]:
                with self.subTest(text=text,scores=scores):
                    output=tensor([(100,4,masks[0],scores[0]),(104.5,4,masks[1],scores[1])])
                    self.assertEqual(decode(output,THRESHOLDS)['text'],text)

    def test_overlapping_duplicate_is_removed_but_nms_chain_retains_disjoint_ends(self):
        duplicate=tensor([(100,12,1,.99),(104,12,1,.99)])
        self.assertEqual(decode(duplicate,THRESHOLDS)['text'],'A')
        chain=tensor([(100,7,1,.99),(104,7,3,.98),(108,7,9,.97)])
        self.assertEqual(decode(chain,THRESHOLDS)['text'],'AC')

    def test_weak_uncertain_or_unsupported_adjacent_cell_rejects_whole_word(self):
        output=tensor([(100,4,1,.99),(104.5,4,3,.3)])
        self.assertEqual(decode(output,dict(THRESHOLDS,acceptScore=.4))['reason'],'low_confidence')
        output[24,26,0]=.98;output[24,26,6]=.5
        self.assertEqual(decode(output,THRESHOLDS)['reason'],'low_confidence')
        output[24,26,5:]=0
        self.assertEqual(decode(output,THRESHOLDS)['reason'],'unsupported_pattern')

    def test_outside_guide_neighbor_cannot_hide_valid_cell_but_intersection_rejects(self):
        output=tensor([(100,4,1,.9),(104.5,4,3,.99)])
        self.assertEqual(decode(output,THRESHOLDS,[.20,.45,.213,.55])['text'],'A')
        self.assertEqual(decode(output,THRESHOLDS,[.20,.45,.22,.55])['reason'],'clipped')

    def test_calibration_cache_retains_nonmaxima_and_matches_raw_threshold_trials(self):
        raw=tensor([(100,4,1,.99),(104.5,4,3,.98)])[None]
        original=raw.copy();cached=prepare_calibration_outputs(raw)
        np.testing.assert_array_equal(raw,original)
        self.assertEqual(float(cached[0,24,26,0]),float(raw[0,24,26,0]))
        for proposal in (.15,.25,.35):
            threshold=dict(THRESHOLDS,proposalScore=proposal,acceptScore=max(.4,proposal))
            self.assertEqual(decode(raw[0],threshold),decode(cached[0],threshold))
        malformed=raw.copy();malformed[0,0,0,0]=np.nan
        np.testing.assert_array_equal(prepare_calibration_outputs(malformed),malformed)


if __name__=='__main__':unittest.main()
