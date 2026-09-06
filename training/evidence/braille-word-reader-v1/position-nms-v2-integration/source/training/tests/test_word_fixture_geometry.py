"""Startup-fixture crop/pixel parity against the actual TypeScript photo pipeline."""
import base64
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from word_fixture import fixture_descriptor, runtime_geometry, check_source_input
from word_data import make_target


class WordFixtureGeometryTests(unittest.TestCase):
    def test_real_rounding_boundary_cannot_use_different_cached_crop(self):
        # Exact annotation geometry of natural validation KOTIKI, with generated
        # pixels. Its normalized source ROI rounds the right edge one pixel wider.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (400, 400)).save(root/"source.png")
            record = {"source": "source.png", "image": "unused.png",
                      "crop": [72, 160, 282, 245], "guide": [72., 175.625, 282., 229.375]}
            descriptor = fixture_descriptor(record, root)
            geometry = runtime_geometry([descriptor])["fixtures"][0]
            self.assertEqual(descriptor["expected_local_model_crop"], [42, 42, 252, 127])
            self.assertEqual(geometry["crop"], [42, 42, 253, 127])
            self.assertEqual(check_source_input(record, descriptor, geometry, root, root),
                             {"eligible": False, "reason": "runtime_integer_crop_differs"})
            # Rejection occurs before even opening a nonexistent cached image.

    def test_matching_crop_checks_actual_source_bytes_and_detects_stale_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = np.random.default_rng(41).integers(0, 256, (192, 480, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(root/"source.png")
            Image.fromarray(rgb).save(root/"cached.png")
            record = {"source": "source.png", "image": "cached.png",
                      "crop": [0, 0, 480, 192], "guide": [0., 60., 480., 132.]}
            descriptor = fixture_descriptor(record, root)
            geometry = runtime_geometry([descriptor])["fixtures"][0]
            self.assertEqual(geometry["crop"], [0, 0, 480, 192])
            self.assertTrue(check_source_input(record, descriptor, geometry, root, root)["eligible"])
            rgb[90, 200, 1] ^= 128
            Image.fromarray(rgb).save(root/"cached.png")
            self.assertEqual(check_source_input(record, descriptor, geometry, root, root)["reason"],
                             "runtime_source_pixels_differ")

    def test_runtime_quality_and_tensor_are_checked_without_model_inference(self):
        rgb = np.random.default_rng(9).integers(0, 256, (192, 480, 3), dtype=np.uint8)
        descriptor = {"width": 480, "height": 192, "roi": {"x": 0., "y": .3125, "width": 1., "height": .375}}
        candidates = [{**descriptor, "rgb_base64": base64.b64encode(pixels.tobytes()).decode("ascii")}
                      for pixels in [rgb, np.zeros_like(rgb)]]
        valid, blank = runtime_geometry(candidates)["fixtures"]
        self.assertEqual(valid["input_float32_sha256"], hashlib.sha256((rgb.astype(np.float32)/255.).tobytes()).hexdigest())
        self.assertEqual(valid["selection"], descriptor["roi"])
        self.assertEqual(valid["quality_reason"], "")
        self.assertEqual(blank["quality_reason"], "dark")
        self.assertIsNone(blank["input_float32_sha256"])

    def test_vertical_short_guide_expands_horizontally_and_clamps_to_source(self):
        descriptor = {"width": 400, "height": 200, "roi": {"x": .025, "y": .3, "width": .1, "height": .4}}
        geometry = runtime_geometry([descriptor])["fixtures"][0]
        self.assertEqual(geometry["crop"], [0, 60, 200, 140])
        self.assertEqual(geometry["selection"], {"x": .05, "y": 0., "width": .2, "height": 1.})

    def test_saved_output_runs_actual_decoder_with_runtime_selection(self):
        # Synthetic model output tests the reference wiring; it is not evidence
        # that a trained model recognizes this randomly generated image.
        rgb = np.random.default_rng(9).integers(0, 256, (192, 480, 3), dtype=np.uint8)
        output = make_target(np.asarray([[210., 70., 235., 115.]]), np.asarray([1]))
        # Only exact centers are ideal predicted objects, not Gaussian loss weights.
        output[..., 0] = np.where(output[..., 0] >= .999, output[..., 0], 0)
        descriptor = {"width": 480, "height": 192, "roi": {"x": 0., "y": .3125, "width": 1., "height": .375},
                      "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
                      "output_float32_base64": base64.b64encode(output.tobytes()).decode("ascii")}
        actual = runtime_geometry([descriptor])["fixtures"][0]
        self.assertEqual(actual["decoded_text"], "A")
        self.assertEqual(actual["rejection"], "")
        self.assertEqual(actual["cached_output_sha256"], hashlib.sha256(output.tobytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
