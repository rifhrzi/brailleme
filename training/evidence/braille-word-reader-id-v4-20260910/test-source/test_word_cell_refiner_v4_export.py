"""Strict tiny-CNN architecture and validation-crop provenance before export."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_cell_refiner_v4_export import SHAPES,load_weights,read_crops,sha,validate_probabilities


class RefinerExportTests(unittest.TestCase):
    def test_exact_finite_float32_architecture_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'weights.npz'
            weights={name:np.zeros(shape,np.float32) for name,shape in SHAPES.items()}
            np.savez(path,**weights)
            self.assertEqual(set(load_weights(path)),set(SHAPES))
            for mutation in ('missing','shape','dtype','nonfinite'):
                changed=dict(weights)
                name='classifier.2.bias'
                if mutation=='missing':del changed[name]
                elif mutation=='shape':changed[name]=np.zeros(7,np.float32)
                elif mutation=='dtype':changed[name]=np.zeros(6,np.float64)
                else:changed[name]=np.full(6,np.nan,np.float32)
                np.savez(path,**changed)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):load_weights(path)

    def test_validation_predicted_crops_bind_manifest_detector_and_crop_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            validation,detector,source,fixture=[root/name for name in ['validation.json','detector.keras','crop.py','crops.npz']]
            validation.write_text(json.dumps({'split':'validation'}))
            detector.write_bytes(b'frozen detector');source.write_bytes(b'crop protocol')
            values=dict(inputs=np.zeros((2,64,40,1),np.float32),scope='validation_predicted_boxes',
                        validation_sha256=sha(validation),detector_sha256=sha(detector),crop_code_sha256=sha(source))
            np.savez(fixture,**values)
            self.assertEqual(read_crops(fixture,validation,detector,source).shape,(2,64,40,1))
            for key in ('scope','validation_sha256','detector_sha256','crop_code_sha256'):
                np.savez(fixture,**dict(values,**{key:'incorrect'}))
                with self.subTest(key=key),self.assertRaises(ValueError):read_crops(fixture,validation,detector,source)
            np.savez(fixture,**values)
            validation.write_text(json.dumps({'split':'test'}))
            with self.assertRaisesRegex(ValueError,'never TRAIN or TEST'):read_crops(fixture,validation,detector,source)

    def test_probability_count_range_and_finiteness_are_required(self):
        valid=np.zeros((2,6),np.float32)
        validate_probabilities(valid,2)
        for values,count in [(valid,1),(valid+1.01,2),(valid-1,2),(valid*np.nan,2)]:
            with self.assertRaises(ValueError):validate_probabilities(values,count)


if __name__=='__main__':
    unittest.main()
