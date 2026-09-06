"""Partial source geometry must never supervise an unseen dot value."""
import os
from pathlib import Path
import sys
import unittest

os.environ['KERAS_BACKEND']='torch'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

import numpy as np
import torch
from word_data import make_target
from word_model import word_relative_loss


class WordLossTests(unittest.TestCase):
    def test_partial_objectness_and_geometry_have_gradients_but_six_bits_do_not(self):
        target=make_target(np.asarray([[200.,-20.,240.,60.],[300.,60.,340.,120.]]),np.asarray([1,3]),partial_geometry_supervision=True)
        prediction=torch.full((1,48,120,11),.5,requires_grad=True)
        loss=word_relative_loss(torch.from_numpy(target[None]),prediction)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        partial=prediction.grad[0,5,55]
        self.assertNotEqual(float(partial[0]),0.)
        self.assertGreater(float(partial[1:5].abs().sum()),0.)
        self.assertEqual(float(partial[5:].abs().sum()),0.)
        self.assertGreater(float(prediction.grad[...,5:].abs().sum()),0.)

    def test_unknown_partial_mask_does_not_affect_training_loss(self):
        boxes=np.asarray([[200.,-20.,240.,60.]])
        first=make_target(boxes,np.asarray([1]),partial_geometry_supervision=True)
        second=make_target(boxes,np.asarray([63]),partial_geometry_supervision=True)
        np.testing.assert_array_equal(first,second)
        prediction=torch.full((1,48,120,11),.5)
        self.assertEqual(float(word_relative_loss(torch.from_numpy(first[None]),prediction)),float(word_relative_loss(torch.from_numpy(second[None]),prediction)))


if __name__=='__main__':
    unittest.main()
