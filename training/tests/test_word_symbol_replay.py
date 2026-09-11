"""Cached output rows must belong to the actual TFLite artifact and fixture."""
import io
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from word_symbol_replay import validate_oracle_binding


class ReplayBindingTests(unittest.TestCase):
    def validate(self,**changes):
        values=dict(oracle_kind='tflite',model_sha256='a'*64,checkpoint_sha256='b'*64,fixture_sha256='c'*64)
        values.update(changes)
        stream=io.BytesIO();np.savez(stream,**values);stream.seek(0)
        with np.load(stream) as archive:
            validate_oracle_binding(archive,'a'*64,'b'*64,'c'*64)

    def test_actual_binding_passes(self):
        self.validate()

    def test_torch_or_wrong_model_checkpoint_fixture_cannot_be_replayed(self):
        for field,value in [('oracle_kind','torch'),('model_sha256','d'*64),('checkpoint_sha256','d'*64),('fixture_sha256','d'*64)]:
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,field):
                self.validate(**{field:value})

    def test_unbound_legacy_cache_fails(self):
        stream=io.BytesIO();np.savez(stream,outputs=np.zeros((1,48,120,11),np.float32));stream.seek(0)
        with np.load(stream) as archive,self.assertRaisesRegex(ValueError,'oracle_kind'):
            validate_oracle_binding(archive,'a'*64,'b'*64,'c'*64)


if __name__=='__main__':
    unittest.main()
