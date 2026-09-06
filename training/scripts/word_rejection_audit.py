"""Check frozen rejection fixture pixels, source annotations, and selection math."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

LETTER_MASKS = {1, 3, 9, 25, 17, 11, 27, 19, 10, 26, 5, 7, 13,
                29, 21, 15, 31, 23, 14, 30, 37, 39, 58, 45, 61, 53}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("C:/bm-datasets/AngelinaDataset"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    split = json.loads(args.source_split.read_text())
    assert sha256(args.source_split) == manifest["source_split_sha256"]
    assert not manifest["model_predictions_used"]
    assert not manifest["used_for_checkpoint_or_threshold_selection"]
    rows, max_coordinate_error = [], 0.
    for record in manifest["records"]:
        image_path = args.manifest.parent / record["image"]
        assert sha256(image_path) == record["image_sha256"]
        with Image.open(image_path) as image:
            actual = np.asarray(image.convert("RGB"))
        assert actual.shape == (192, 480, 3)
        assert record["expected_reject"] and record["expected"] == ""
        selection = np.asarray(record["selection"])
        assert np.isfinite(selection).all() and np.all(selection >= 0) and np.all(selection <= 1)
        assert np.all(selection[2:] > selection[:2])
        if record["category"] == "blank":
            assert np.all(actual == record["generation"]["rgb"])
            rows.append({"id": record["id"], "pixels_exact": True, "annotation_checks": "generated_blank"})
            continue
        source = record["source"]
        assert source in split["images"]["validation"]
        assert source not in split["images"]["train"] and source not in split["images"]["external"]
        source_path = args.dataset / source
        assert sha256(source_path) == record["source_sha256"]
        assert sha256(source_path.with_suffix(".csv")) == record["annotation_sha256"]
        with Image.open(source_path) as opened:
            original = opened.convert("RGB")
        annotations = np.loadtxt(source_path.with_suffix(".csv"), delimiter=";", ndmin=2)
        source_boxes = annotations[:, :4] * np.asarray(list(original.size) * 2)
        crop, guide = np.asarray(record["crop"]), np.asarray(record["guide"])
        assert crop[0] >= 0 and crop[1] >= 0 and crop[2] <= original.width and crop[3] <= original.height
        patch = original.crop(tuple(crop))
        crop_path = args.manifest.parent / record["source_crop_image"]
        assert sha256(crop_path) == record["source_crop_sha256"]
        with Image.open(crop_path) as saved:
            assert np.array_equal(np.asarray(patch), np.asarray(saved.convert("RGB")))
        assert np.array_equal(actual, np.asarray(patch.resize((480, 192), Image.Resampling.BILINEAR)))
        offset = np.tile(crop[:2], 2)
        size = np.tile(crop[2:] - crop[:2], 2)
        center = (guide[:2] + guide[2:]) / 2
        guide_width, guide_height = guide[2:] - guide[:2]
        context_width = max(float(guide_width), float(guide_height) * 2.5)
        half = np.array([context_width, context_width / 2.5]) / 2
        assert crop.tolist() == [math.floor(center[0] - half[0]), math.floor(center[1] - half[1]),
                                 math.ceil(center[0] + half[0]), math.ceil(center[1] + half[1])]
        assert 1 <= guide_width / guide_height <= 8
        np.testing.assert_allclose(selection, (guide - offset) / size, atol=0, rtol=0)
        all_indices = np.flatnonzero((source_boxes[:, 0] < crop[2]) & (source_boxes[:, 2] > crop[0])
                                    & (source_boxes[:, 1] < crop[3]) & (source_boxes[:, 3] > crop[1]))
        assert (all_indices + 1).tolist() == record["context_annotation_rows"]
        context_boxes = source_boxes[all_indices]
        context_full = ((context_boxes[:, 0] >= crop[0]) & (context_boxes[:, 1] >= crop[1])
                        & (context_boxes[:, 2] <= crop[2]) & (context_boxes[:, 3] <= crop[3]))
        assert context_full.tolist() == record["context_complete_in_crop"]
        for prefix in ("", "context_"):
            indices = np.asarray(record[prefix + "annotation_rows"]) - 1
            assert annotations[indices, 4].astype(int).tolist() == record[prefix + "masks"]
            np.testing.assert_allclose(record[prefix + "source_boxes"], source_boxes[indices], atol=0, rtol=0)
            expected = (source_boxes[indices] - offset) * (np.tile([480, 192], 2) / size)
            max_coordinate_error = max(max_coordinate_error, float(np.max(np.abs(expected - record[prefix + "boxes"]))))
            np.testing.assert_allclose(expected, record[prefix + "boxes"], atol=0, rtol=0)
        selected = source_boxes[np.asarray(record["annotation_rows"]) - 1]
        in_guide = ((source_boxes[:, 0] < guide[2]) & (source_boxes[:, 2] > guide[0])
                    & (source_boxes[:, 1] < guide[3]) & (source_boxes[:, 3] > guide[1]))
        assert set(np.flatnonzero(in_guide) + 1) == set(record["annotation_rows"])
        whole = ((selected[:, 0] >= guide[0]) & (selected[:, 1] >= guide[1])
                 & (selected[:, 2] <= guide[2]) & (selected[:, 3] <= guide[3]))
        if record["category"] == "selection_cut":
            assert int((~whole).sum()) == 1 and not whole[-1]
            assert record["cut_annotation_row"] == record["annotation_rows"][-1]
            assert guide[2] == (selected[-1, 0] + selected[-1, 2]) / 2
            assert all(mask in LETTER_MASKS for mask in record["masks"])
        else:
            assert whole.all()
        if record["category"] == "unsupported_pattern":
            unsupported = [mask for mask in record["masks"] if mask not in LETTER_MASKS]
            assert len(unsupported) == 1 and unsupported[0].bit_count() >= 3
            assert len(record["masks"]) >= 3
        elif record["category"] == "multiple_rows":
            assert len(set(record["line_ids"])) == 2
            assert all(mask in LETTER_MASKS for mask in record["masks"])
            levels = sorted(set(record["line_ids"]))
            top = selected[np.asarray(record["line_ids"]) == levels[0]]
            bottom = selected[np.asarray(record["line_ids"]) == levels[1]]
            assert bottom[:, 1].min() > top[:, 3].max()
        elif record["category"] == "over_limit":
            assert len(record["masks"]) > 12 and len(set(record["line_ids"])) == 1
        rows.append({"id": record["id"], "pixels_exact": True, "annotation_checks": "passed",
                     "source": source, "selected_cells": len(selected), "complete_in_guide": int(whole.sum()),
                     "source_sha256": record["source_sha256"], "image_sha256": record["image_sha256"]})
    assert Counter(record["category"] for record in manifest["records"]) == {
        "unsupported_pattern": 3, "multiple_rows": 3, "selection_cut": 3, "over_limit": 3, "blank": 2}
    report = {"scope": "Fixture provenance, geometry, and pixel integrity only; no model inference",
              "status": "passed", "manifest_sha256": sha256(args.manifest), "audit_script_sha256": sha256(__file__),
              "records": len(rows), "source_photos": len({r["source"] for r in rows if "source" in r}),
              "all_source_photos_validation_only": True, "all_pixels_exact": True,
              "max_box_coordinate_error_pixels": max_coordinate_error,
              "no_model_or_threshold_selection": True, "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
