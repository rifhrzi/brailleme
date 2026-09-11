"""Freeze small rejection diagnostics from validation photographs, before inference.

This is a labeled regression set, not an independent accuracy holdout. Selection
uses source annotations and geometry only; this script never imports a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from benchmark_angelina import LETTER_BY_MASK
from word_data import lines_from_boxes, word_groups

WIDTH, HEIGHT = 480, 192
CATEGORIES = ("unsupported_pattern", "multiple_rows", "selection_cut", "over_limit")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intersects(boxes, region):
    return ((boxes[:, 0] < region[2]) & (boxes[:, 2] > region[0])
            & (boxes[:, 1] < region[3]) & (boxes[:, 3] > region[1]))


def complete(boxes, region):
    return ((boxes[:, 0] >= region[0]) & (boxes[:, 1] >= region[1])
            & (boxes[:, 2] <= region[2]) & (boxes[:, 3] <= region[3]))


def guide_for(boxes, indices):
    selected = boxes[indices]
    lower, upper = selected[:, :2].min(axis=0), selected[:, 2:].max(axis=0)
    cell_height = float(np.median(selected[:, 3] - selected[:, 1]))
    extent = upper - lower + [cell_height * .4, cell_height * .3]
    extent[0] = max(extent[0], extent[1])
    # Stay visibly inside the application's 8:1 maximum rather than depending
    # on floating-point cancellation at exactly 8:1.
    extent[1] = max(extent[1], extent[0] / 7.9)
    center = (lower + upper) / 2
    return np.concatenate((center - extent / 2, center + extent / 2))


def context_for(guide, size):
    """Same aspect-2.5 context rule as word fixtures; outward integer crop."""
    center = (guide[:2] + guide[2:]) / 2
    guide_size = guide[2:] - guide[:2]
    width = max(float(guide_size[0]), float(guide_size[1]) * 2.5)
    extent = np.array([width, width / 2.5])
    lower, upper = center - extent / 2, center + extent / 2
    crop = [math.floor(lower[0]), math.floor(lower[1]), math.ceil(upper[0]), math.ceil(upper[1])]
    if crop[0] < 0 or crop[1] < 0 or crop[2] > size[0] or crop[3] > size[1]:
        return None
    return crop


def row_ids(boxes):
    ids = {}
    lines = lines_from_boxes(boxes)
    lines.sort(key=lambda line: float(np.median((boxes[line, 1] + boxes[line, 3]) / 2)))
    for line_id, line in enumerate(lines):
        for index in line:
            ids[index] = line_id
    return ids


def candidates(image, boxes, masks, category, supported_masks=None):
    supported_masks=set(LETTER_BY_MASK) if supported_masks is None else set(supported_masks)
    groups = word_groups(boxes)
    groups.sort(key=lambda group: (min(boxes[group, 1]), min(boxes[group, 0])))
    rows = row_ids(boxes)
    choices = []
    if category == "multiple_rows":
        # Adjacent naturally photographed rows; no translation/composition.
        eligible = [group for group in groups if 2 <= len(group) <= 5
                    and all(int(masks[i]) in LETTER_BY_MASK for i in group)]
        for first in eligible:
            for second in eligible:
                if rows[second[0]] != rows[first[0]] + 1:
                    continue
                if min(boxes[first, 2].max(), boxes[second, 2].max()) <= max(boxes[first, 0].min(), boxes[second, 0].min()):
                    continue
                choices.append((first + second, None))
    else:
        for group in groups:
            if category == "unsupported_pattern":
                unsupported = [i for i in group if int(masks[i]) not in supported_masks]
                if not (3 <= len(group) <= 7 and len(unsupported) == 1
                        and int(masks[unsupported[0]]).bit_count() >= 3):
                    continue
                choices.append((group, None))
            elif category == "selection_cut":
                if not (3 <= len(group) <= 7 and all(int(masks[i]) in LETTER_BY_MASK for i in group)):
                    continue
                last = group[-1]
                # Cutting between the two occupied dot columns, with other
                # complete cells still visible, is unambiguous in row context.
                if int(masks[last]) & 7 and int(masks[last]) & 56:
                    choices.append((group, last))
            elif category == "over_limit" and len(group) >= 13:
                if {32,60}.issubset(supported_masks) and sum(int(masks[i]) not in (32,60) for i in group)<13:
                    continue
                choices.append((group, None))
    for indices, cut_index in choices:
        guide = guide_for(boxes, indices)
        expected_indices = set(indices)
        if set(np.flatnonzero(intersects(boxes, guide))) != expected_indices:
            continue
        if not complete(boxes[indices], guide).all():
            continue
        if cut_index is not None:
            guide[2] = (boxes[cut_index, 0] + boxes[cut_index, 2]) / 2
        crop = context_for(guide, image.size)
        if crop is None:
            continue
        if not 1 <= (guide[2] - guide[0]) / (guide[3] - guide[1]) <= 8:
            continue
        actual = set(np.flatnonzero(intersects(boxes, guide)))
        if actual != expected_indices:
            continue
        partial = {i for i in actual if not complete(boxes[[i]], guide)[0]}
        if partial != ({cut_index} if cut_index is not None else set()):
            continue
        if category == "multiple_rows" and len({rows[i] for i in actual}) != 2:
            continue
        yield {"indices": indices, "guide": guide, "crop": crop,
               "row_ids": rows, "cut_index": cut_index}


def write_record(output, source, image, boxes, masks, choice, category, ordinal, dataset, supported_masks=None):
    supported_masks=set(LETTER_BY_MASK) if supported_masks is None else set(supported_masks)
    stem = f"{category}-{ordinal:02d}"
    crop, guide, indices = choice["crop"], choice["guide"], choice["indices"]
    patch = image.crop(crop)
    patch.save(output / f"{stem}-source-crop.png")
    model_image = patch.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
    model_image.save(output / f"{stem}.png")
    crop_origin = np.array(crop[:2] * 2)
    crop_extent = np.array([crop[2] - crop[0], crop[3] - crop[1]] * 2)
    scale = np.array([WIDTH, HEIGHT] * 2) / crop_extent
    context_indices = np.flatnonzero(intersects(boxes, crop))
    record = {
        "id": stem, "category": category, "expected": "", "expected_reject": True,
        "image": f"{stem}.png", "image_sha256": sha256(output / f"{stem}.png"),
        "source_crop_image": f"{stem}-source-crop.png",
        "source_crop_sha256": sha256(output / f"{stem}-source-crop.png"),
        "source": source, "source_sha256": sha256(dataset / source),
        "annotation_sha256": sha256((dataset / source).with_suffix(".csv")),
        "source_size": list(image.size), "source_split": "validation",
        "crop": crop, "guide": guide.tolist(),
        "guideAspect": float((guide[2] - guide[0]) / (guide[3] - guide[1])),
        "selection": ((guide - crop_origin) / crop_extent).tolist(),
        "annotation_rows": [int(i) + 1 for i in indices],
        "source_boxes": boxes[indices].tolist(),
        "boxes": ((boxes[indices] - crop_origin) * scale).tolist(),
        "masks": [int(masks[i]) for i in indices],
        "line_ids": [choice["row_ids"][i] for i in indices],
        "context_annotation_rows": (context_indices + 1).tolist(),
        "context_source_boxes": boxes[context_indices].tolist(),
        "context_boxes": ((boxes[context_indices] - crop_origin) * scale).tolist(),
        "context_masks": masks[context_indices].tolist(),
        "context_complete_in_crop": complete(boxes[context_indices], crop).tolist(),
        "cut_annotation_row": None if choice["cut_index"] is None else int(choice["cut_index"]) + 1,
        "unsupported_annotation_rows": [int(i) + 1 for i in indices if int(masks[i]) not in supported_masks],
        "natural_pixels_only": True, "model_predictions_used": False,
    }
    preview = model_image.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle(tuple(np.asarray(record["selection"]) * ([WIDTH, HEIGHT] * 2)), outline="red", width=2)
    for box, mask in zip(record["boxes"], record["masks"]):
        draw.rectangle(tuple(box), outline="blue", width=1)
        draw.text((box[0], max(0, box[1] - 10)), str(mask), fill="blue")
    preview.save(output / f"{stem}-review.png")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("C:/bm-datasets/AngelinaDataset"))
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument('--indonesian-symbols',action='store_true',help='Exclude newly supported indicator patterns from unsupported diagnostics; overlimit counts output characters')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output folder; never overwrite frozen diagnostic fixtures")
    split = json.loads(args.source_split.read_text())
    revision = subprocess.check_output(["git", "-C", str(args.dataset), "rev-parse", "HEAD"], text=True).strip()
    if revision != split["dataset_revision"]:
        raise ValueError("Dataset revision differs from the frozen split")
    validation = set(split["images"]["validation"])
    if validation & (set(split["images"]["train"]) | set(split["images"]["external"])):
        raise ValueError("Source split overlap")
    args.output.mkdir(parents=True)
    pages = {}
    for source in sorted(validation):
        with Image.open(args.dataset / source) as opened:
            image = opened.convert("RGB")
        annotations = np.loadtxt((args.dataset / source).with_suffix(".csv"), delimiter=";", ndmin=2)
        boxes = annotations[:, :4] * np.array(list(image.size) * 2)
        masks = annotations[:, 4].astype(int)
        pages[source] = image, boxes, masks
    records = []
    supported_masks=set(LETTER_BY_MASK)|({32,60} if args.indonesian_symbols else set())
    for category in CATEGORIES:
        # Prefer one case from each held family, then a third distinct photo.
        selected_sources, used_families = set(), set()
        for require_new_family in (True, False):
            for source, (image, boxes, masks) in pages.items():
                family = "/".join(source.split("/")[:2])
                if source in selected_sources or (require_new_family and family in used_families):
                    continue
                choice = next(candidates(image, boxes, masks, category,supported_masks), None)
                if choice is None:
                    continue
                selected_sources.add(source)
                used_families.add(family)
                records.append(write_record(args.output, source, image, boxes, masks,
                                            choice, category, len(selected_sources), args.dataset,supported_masks))
                if len(selected_sources) == 3:
                    break
            if len(selected_sources) == 3:
                break
        if len(selected_sources) != 3:
            raise ValueError(f"Fewer than three eligible distinct validation photos for {category}")
    for name, level in (("blank-white", 255), ("blank-gray", 128)):
        image = Image.new("RGB", (WIDTH, HEIGHT), (level,) * 3)
        image.save(args.output / f"{name}.png")
        records.append({"id": name, "category": "blank", "expected": "", "expected_reject": True,
                        "image": f"{name}.png", "image_sha256": sha256(args.output / f"{name}.png"),
                        "selection": [.1, .25, .9, .75], "boxes": [], "masks": [],
                        "generation": {"kind": "constant_rgb", "rgb": [level] * 3, "size": [WIDTH, HEIGHT]},
                        "model_predictions_used": False})
    manifest = {
        "schema_version": 1, "scope": "held_validation_photo_rejection_diagnostics_and_generated_blanks",
        "dataset_revision": revision, "source_split_sha256": sha256(args.source_split),
        "generator_sha256": sha256(__file__),
        "selection_helper_sha256": sha256(Path(__file__).with_name("word_data.py")),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": pillow_version},
        "model_predictions_used": False, "frozen_before_final_model_inference": True,
        "used_for_checkpoint_or_threshold_selection": False,
        "is_independent_holdout": False, "is_natural_whole_word_benchmark": False,
        "selection_method": "Annotation/geometry-only, lexicographically first eligible photo and top-to-bottom eligible group per held family, then a third distinct photo. Unsupported: one non-A-Z mask with at least three dots plus 2-6 A-Z neighbors. Multiple rows: two adjacent naturally photographed rows with 2-5 A-Z cells per group. Selection cut: terminal two-column cell bisected by the guide. Over limit: a complete annotation-space-bounded group of at least 13 consecutive cells; non-A-Z cells are retained and explicitly recorded as additional rejection conditions.",
        "preprocessing": {"image_mode": "RGB", "crop": "Pillow integer XYXY crop, no padding", "resize": "Pillow BILINEAR to 480x192", "normalization": "float32/255", "boxes": "(source_xyxy - [crop_left,crop_top]*2) * [480/crop_width,192/crop_height]*2", "selection": "(source_guide_xyxy - [crop_left,crop_top]*2) / [crop_width,crop_height]*2", "orientation": "source JPEG decoded as stored; no rotation or EXIF transpose", "other_image_changes": "none"},
        "expected_behavior": "Every case rejects the entire capture. Any accepted text is a failure; do not require a specific rejection reason when multiple conditions apply. No trimming, dictionary repair, or dropping doubtful cells.",
        "category_counts": dict(Counter(record["category"] for record in records)), "records": records,
    }
    serialized = json.dumps(manifest, indent=2) + "\n"
    if args.indonesian_symbols:
        manifest['symbol_protocol']='indonesian-basic-v1'
        manifest['selection_method']=manifest['selection_method'].replace('one non-A-Z mask','one mask outside A-Z/32/60').replace('at least 13 consecutive cells','at least 13 output-bearing cells excluding indicators32/60')
        serialized=json.dumps(manifest,indent=2)+'\n'
    (args.output / "manifest.json").write_text(serialized, encoding="utf-8")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(serialized, encoding="utf-8")
    print(json.dumps({"records": len(records), "categories": manifest["category_counts"],
                      "manifest_sha256": sha256(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
