"""Small arithmetic/rejection tests; never alter a checkpoint or run inference."""
from fractions import Fraction
import unittest
import numpy as np
from average import mean_arrays

class MeanTests(unittest.TestCase):
    def test_signed_cancellation_and_extremes_use_float64(self):
        arrays=[np.asarray(v,np.float32) for v in ([1e30,3,1e-35],[1,-2,2e-35],[-1e30,5,3e-35])]
        got=mean_arrays(arrays)
        # First coordinate specifies ordered float64 addition, not exact-real
        # summation: the tiny middle value is rounded away at this magnitude.
        self.assertEqual(got[0],0)
        for i in (1,2):
            rational=sum((Fraction.from_float(float(a[i])) for a in arrays),Fraction())/3
            self.assertEqual(got[i].tobytes(),np.float32(float(rational)).tobytes())
        self.assertEqual(got.dtype,np.dtype('float32'))
    def test_same_values_are_stable_and_inputs_unchanged(self):
        a=np.array([[-3.5,0,1.125]],np.float32);copies=[a.copy() for _ in range(3)]
        np.testing.assert_array_equal(mean_arrays(copies),a)
        for value in copies:np.testing.assert_array_equal(value,a)
    def test_count_shape_dtype_and_nonfinite_rejected(self):
        a=np.ones((2,3),np.float32)
        for arrays in ([a,a],[a,a,a,a],[a,a,a.astype(np.float64)],[a,a,np.ones((3,2),np.float32)],
                       [a,a,np.full_like(a,np.nan)],[a,a,np.full_like(a,np.inf)]):
            with self.assertRaises(ValueError):mean_arrays(arrays)
    def test_large_finite_inputs_do_not_overflow_before_division(self):
        a=np.array([np.finfo(np.float32).max],np.float32)
        np.testing.assert_array_equal(mean_arrays([a,a,a]),a)
if __name__=='__main__':unittest.main(verbosity=2)
