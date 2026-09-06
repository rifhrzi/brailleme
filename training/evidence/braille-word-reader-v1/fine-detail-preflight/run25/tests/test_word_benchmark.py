"""Public report reasons must preserve the actual image-quality rejection."""
from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from word_benchmark import quality, summarize_public_words
from word_data import make_target
from word_decode import decode, summarize


def word_output(masks):
    boxes = np.asarray([[180 + index * 42, 65, 208 + index * 42, 115] for index in range(len(masks))])
    return make_target(boxes, np.asarray(masks))


class WordBenchmarkTests(unittest.TestCase):
    def test_quality_placeholders_keep_final_reasons_without_changing_word_metrics(self):
        quality_reasons = [quality(np.full((8, 8, 3), value, np.float32)) for value in (0., 1., .5)]
        self.assertEqual(quality_reasons, ["dark", "glare", "low_confidence"])
        quality_reasons += ["", "", ""]
        one = word_output([1])
        raw_outputs = [one, one, one, np.zeros_like(one), one, word_output([1, 3])]
        records = [{"expected": text} for text in ("A", "A", "A", "A", "A", "AC")]
        predictions, rows = [], []
        for output, reason in zip(raw_outputs, quality_reasons):
            result = {"text": "", "reason": reason} if reason else decode(output)
            rows.append({"predicted": result["text"], "rejection": result["reason"]})
            predictions.append(np.zeros_like(output) if reason else output)

        prior = summarize(predictions, records)
        actual = summarize_public_words(predictions, records, None, rows)
        self.assertEqual(prior["rejection_reasons"], {"no_grid": 4})
        self.assertEqual(actual["rejection_reasons"], {"dark": 1, "glare": 1, "low_confidence": 1, "no_grid": 1})
        self.assertEqual(actual["rejection_reasons"], dict(Counter(row["rejection"] for row in rows if not row["predicted"])))
        self.assertEqual({key: value for key, value in actual.items() if key != "rejection_reasons"},
                         {key: value for key, value in prior.items() if key != "rejection_reasons"})
        self.assertEqual((actual["total"], actual["accepted"], actual["correct"]), (6, 2, 1))
        self.assertEqual((actual["by_length"]["1"]["total"], actual["by_length"]["1"]["accepted"], actual["by_length"]["1"]["correct"]), (5, 1, 1))
        self.assertEqual((actual["by_length"]["2"]["total"], actual["by_length"]["2"]["accepted"], actual["by_length"]["2"]["correct"]), (1, 1, 0))

    def test_reports_without_quality_rejection_are_unchanged(self):
        predictions = [word_output([1]), np.zeros_like(word_output([1]))]
        records = [{"expected": "A"}, {"expected": "AB"}]
        results = [decode(output) for output in predictions]
        rows = [{"predicted": result["text"], "rejection": result["reason"]} for result in results]
        self.assertEqual(summarize_public_words(predictions, records, None, rows), summarize(predictions, records))


if __name__ == "__main__":
    unittest.main()
