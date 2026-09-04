from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_angelina import (  # noqa: E402
    CellRecord,
    parse_annotations,
    preprocess_cell,
    read_split,
    resolve_within,
    square_crop_box,
    summarize_page_accuracy,
)


class AngelinaBenchmarkTests(unittest.TestCase):
    def test_rejects_path_outside_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "keluar dari root dataset"):
                resolve_within(root, "../outside.jpg")

    def test_reads_split_and_maps_only_a_z_masks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subset = root / "uploaded"
            subset.mkdir()
            image_path = subset / "page.labeled.jpg"
            Image.new("RGB", (100, 100), "white").save(image_path)
            (subset / "page.labeled.csv").write_text(
                "0.1;0.1;0.2;0.3;1\n0.3;0.1;0.4;0.3;63\n",
                encoding="utf-8",
            )
            (subset / "test.txt").write_text(
                "page.labeled.jpg\n", encoding="utf-8"
            )

            image_paths = read_split(root, "uploaded/test.txt")
            records, excluded, errors = parse_annotations(root, image_paths)

            self.assertEqual([record.label for record in records], ["a"])
            self.assertEqual(excluded[63], 1)
            self.assertEqual(errors, [])

    def test_square_crop_stays_inside_image(self):
        box = square_crop_box(100, 80, (0.0, 0.0, 0.1, 0.2))
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[2], 100)
        self.assertLessEqual(box[3], 80)
        self.assertEqual(box[2] - box[0], box[3] - box[1])

    def test_preprocessing_matches_model_contract(self):
        image = Image.new("RGB", (100, 100), (64, 128, 255))
        record = CellRecord(
            "test", "page.jpg", 1, "a", 1, 0.2, 0.2, 0.4, 0.6
        )
        tensor = preprocess_cell(image, record)
        self.assertEqual(tensor.shape, (28, 28, 3))
        self.assertEqual(str(tensor.dtype), "float32")
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_summarizes_pages_below_internal_target(self):
        summary = summarize_page_accuracy(
            [
                {"source_image": "page-b.jpg", "correct": True},
                {"source_image": "page-a.jpg", "correct": True},
                {"source_image": "page-a.jpg", "correct": False},
                {"source_image": "page-b.jpg", "correct": True},
            ]
        )

        self.assertEqual(summary["page_count"], 2)
        self.assertEqual(summary["pages_below_target_count"], 1)
        self.assertEqual(summary["minimum_page_accuracy"], 0.5)
        self.assertEqual(
            summary["pages_below_target"],
            [
                {
                    "source_image": "page-a.jpg",
                    "sample_count": 2,
                    "correct_count": 1,
                    "top1_accuracy": 0.5,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
