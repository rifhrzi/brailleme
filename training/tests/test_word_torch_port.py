"""Stored-oracle and roundtrip checks for the Keras-free Torch execution port."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
import zipfile

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from word_torch_port import load_keras_checkpoint, save_keras_checkpoint, parse_checkpoint, joint_word_loss

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO / 'training/checkpoints/braille-word-reader-v1/best-word.keras'
EXPERIMENT = Path('C:/bm-symbols-20260908')
ORACLE = EXPERIMENT / 'deployment-baseline/backend-fullprecision.npz'
FIXTURES = EXPERIMENT / 'legacy-selection-fixtures.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PortableTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model = load_keras_checkpoint(CHECKPOINT).to(cls.device).eval()

    def test_eight_archived_inputs_match_frozen_torch_backend(self):
        if not ORACLE.exists() or not FIXTURES.exists():
            self.skipTest('Retained deployment baseline is unavailable')
        reference = np.load(ORACLE)
        self.assertEqual(str(reference['checkpoint_sha256']), digest(CHECKPOINT))
        self.assertEqual(str(reference['fixture_sha256']), digest(FIXTURES))
        records = json.loads(FIXTURES.read_text())['records']
        indices = [0, 1, 2, 49, 100, 200, 300, 404]
        images = np.asarray([np.asarray(Image.open(records[index]['image']).convert('RGB'))
                             for index in indices], dtype=np.float32) / 255.
        with torch.no_grad():
            actual = self.model(torch.from_numpy(images).to(self.device)).cpu().numpy()
        np.testing.assert_allclose(actual, reference['outputs'][indices], rtol=0, atol=1e-4)

    def test_checkpoint_roundtrip_updates_weights_and_keeps_original_architecture(self):
        path = Path(tempfile.gettempdir()) / f'brailleme-torch-port-test-{uuid.uuid4().hex}.keras'
        original_digest = digest(CHECKPOINT)
        model = load_keras_checkpoint(CHECKPOINT).eval()
        config, weights = parse_checkpoint(CHECKPOINT)
        self.assertEqual(weights[('mobilenetv2_half', 'Conv1')][0].shape, (3, 3, 3, 16))
        self.assertEqual(weights[('dot_probabilities',)][1].shape, (6,))
        try:
            with torch.no_grad():
                for operation in model.modules():
                    if getattr(operation, 'keras_name', '') == 'dot_probabilities':
                        operation.weights[1].add_(.125)
            save_keras_checkpoint(model, CHECKPOINT, path)
            with self.assertRaises(FileExistsError):
                save_keras_checkpoint(model, CHECKPOINT, path)
            saved_config, saved_weights = parse_checkpoint(path)
            self.assertEqual(saved_config['config'], config['config'])
            self.assertEqual(saved_config['compile_config'], {})
            np.testing.assert_array_equal(saved_weights[('dot_probabilities',)][1], weights[('dot_probabilities',)][1] + .125)
            generator = torch.Generator().manual_seed(802026)
            inputs = torch.rand((1, 192, 480, 3), generator=generator)
            with torch.no_grad():
                np.testing.assert_array_equal(model(inputs).numpy(), load_keras_checkpoint(path).eval()(inputs).numpy())
            self.assertEqual(digest(CHECKPOINT), original_digest)
        finally:
            if path.exists():
                path.unlink()

    def test_backbone_freeze_preserves_original_frozen_bn_and_backward_is_finite(self):
        model = load_keras_checkpoint(CHECKPOINT)
        full = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        model.freeze_backbone()
        partial = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertGreater(full, partial)
        self.assertGreater(partial, 0)
        for operation in model.modules():
            if getattr(operation, 'kind', '') == 'BatchNormalization':
                self.assertTrue(all(not parameter.requires_grad for parameter in operation.weights))
        model.freeze_backbone(False)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), full)
        model.freeze_backbone().to(self.device).train()
        inputs = torch.full((1, 192, 480, 3), .5, device=self.device)
        output = model(inputs)
        truth = torch.zeros_like(output)
        truth[0, 24, 60] = output.new_tensor([1., .5, .5, .05, .2, 1, 0, 0, 0, 0, 0])
        loss = joint_word_loss(truth, output)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_unknown_dots_and_ignored_slots_do_not_receive_gradients(self):
        truth = torch.zeros((1, 2, 2, 11))
        truth[0, 0, 0] = torch.tensor([1., .2, .3, .1, .2, -1, -1, -1, -1, -1, -1])
        truth[0, 0, 1, 0] = -1
        prediction = torch.full_like(truth, .5, requires_grad=True)
        joint_word_loss(truth, prediction).backward()
        self.assertGreater(float(prediction.grad[0, 0, 0, :5].abs().sum()), 0)
        self.assertEqual(float(prediction.grad[0, 0, 0, 5:].abs().sum()), 0)
        self.assertEqual(float(prediction.grad[0, 0, 1].abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
