"""Compose frozen repetition diagnostics from annotated TRAINING photo cells.

These synthetic compositions are decoder/model regressions, never external-word
or physical-camera accuracy evidence. Source selection uses annotations and
geometry only; this script does not load a model or read predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

WORDS = ("A", "MAMA", "KAKAK", "MAMAMAMAMAMA")
MASKS = {"A": 1, "K": 5, "M": 13}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cells(dataset, source):
    """First geometrically isolated A/K/M cells from a single photograph."""
    path = dataset / source
    image = Image.open(path).convert("RGB")
    rows = np.loadtxt(path.with_suffix(".csv"), delimiter=";", ndmin=2)
    boxes = rows[:, :4] * np.asarray([image.width, image.height] * 2)
    masks = rows[:, 4].astype(int)
    heights = boxes[:, 3] - boxes[:, 1]
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    gaps = []
    for index, center in enumerate(centers):
        dx = centers[:, 0] - center[0]
        candidates = dx[(dx > 0) & (dx < heights[index] * 2.5)
                        & (abs(centers[:, 1] - center[1]) < heights[index] * .4)]
        if len(candidates):
            gaps.append(float(candidates.min() / heights[index]))
    if not gaps:
        return None
    gaps.sort()
    typical_height = float(np.median(heights))
    pitch_ratio = float(np.median(gaps[:max(1, math.ceil(len(gaps) * .6))]))
    pitch = max(1, round(pitch_ratio * typical_height))
    tile_height = math.ceil(typical_height * 1.25)
    selected = {}
    for letter, mask in MASKS.items():
        for index in np.flatnonzero(masks == mask):
            box = boxes[index]
            if not .85 * typical_height <= heights[index] <= 1.15 * typical_height:
                continue
            cx, cy = centers[index]
            left, top = math.floor(cx - pitch / 2), math.floor(cy - tile_height / 2)
            crop = [left, top, left + pitch, top + tile_height]
            if left < 0 or top < 0 or crop[2] > image.width or crop[3] > image.height:
                continue
            if not (box[0] >= left + 1 and box[1] >= top + 1
                    and box[2] <= crop[2] - 1 and box[3] <= crop[3] - 1):
                continue
            intersects = (boxes[:, 0] < crop[2]) & (boxes[:, 2] > left) & (boxes[:, 1] < crop[3]) & (boxes[:, 3] > top)
            if set(np.flatnonzero(intersects)) != {int(index)}:
                continue
            selected[letter] = {"source": source, "annotation_row": int(index) + 1,
                                "mask": mask, "crop": crop, "source_box": box.tolist()}
            break
    if len(selected) != len(MASKS):
        return None
    return image, selected, pitch, tile_height, pitch_ratio


def compose(image, selected, pitch, tile_height, word):
    tiles = {letter: image.crop(cell["crop"]) for letter, cell in selected.items()}
    border_pixels = []
    for tile in tiles.values():
        pixels = np.asarray(tile)
        border_pixels.extend([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    paper = tuple(np.median(np.concatenate(border_pixels), axis=0).astype(np.uint8).tolist())
    local_boxes = []
    for position, letter in enumerate(word):
        cell = selected[letter]
        local_boxes.append(np.asarray(cell["source_box"]) - np.asarray(cell["crop"][:2] * 2)
                           + np.asarray([position * pitch, 0] * 2))
    local_boxes = np.asarray(local_boxes)
    x0, y0 = local_boxes[:, :2].min(axis=0)
    x1, y1 = local_boxes[:, 2:].max(axis=0)
    height = float(np.median(local_boxes[:, 3] - local_boxes[:, 1]))
    guide_width = max(float(x1 - x0) + height * .6, float(y1 - y0) * 1.25)
    guide_height = max(float(y1 - y0) * 1.25, guide_width / 8)
    context_width = max(guide_width, guide_height * 2.5)
    canvas_width, canvas_height = math.ceil(context_width), math.ceil(context_width / 2.5)
    offset_x = round(canvas_width / 2 - (x0 + x1) / 2)
    offset_y = round(canvas_height / 2 - (y0 + y1) / 2)
    canvas = Image.new("RGB", (canvas_width, canvas_height), paper)
    source_cells = []
    for position, letter in enumerate(word):
        cell = selected[letter]
        paste = [offset_x + position * pitch, offset_y]
        # Only the outer paper margin is blended. Annotated cell pixels retain
        # alpha255 and remain identical to the original photograph before resize.
        alpha = np.full((tile_height, pitch), 255, dtype=np.uint8)
        alpha[[0, -1], :] = 0
        alpha[:, [0, -1]] = 0
        canvas.paste(tiles[letter], tuple(paste), Image.fromarray(alpha))
        source_cells.append(dict(cell, letter=letter, position=position, paste_xy=paste))
    actual_boxes = local_boxes + np.asarray([offset_x, offset_y] * 2)
    x0, y0 = actual_boxes[:, :2].min(axis=0)
    x1, y1 = actual_boxes[:, 2:].max(axis=0)
    cx, cy = canvas_width / 2, canvas_height / 2
    guide = [float(cx - guide_width / 2), float(cy - guide_height / 2),
             float(cx + guide_width / 2), float(cy + guide_height / 2)]
    assert guide[0] >= 0 and guide[1] >= 0 and guide[2] <= canvas_width and guide[3] <= canvas_height
    assert np.all(actual_boxes[:, 0] >= guide[0]) and np.all(actual_boxes[:, 2] <= guide[2])
    assert np.all(actual_boxes[:, 1] >= guide[1]) and np.all(actual_boxes[:, 3] <= guide[3])
    # Verify original occupied-cell pixels were neither redrawn nor recolored.
    for cell in source_cells:
        source_box = cell["source_box"]
        left, top, right, bottom = map(round, source_box)
        dx, dy = np.asarray(cell["paste_xy"]) - np.asarray(cell["crop"][:2])
        assert image.crop((left, top, right, bottom)).tobytes() == canvas.crop((left + dx, top + dy, right + dx, bottom + dy)).tobytes()
    scale = np.asarray([480 / canvas_width, 192 / canvas_height] * 2)
    return canvas, {"expected": word, "selection": (np.asarray(guide) / np.asarray([canvas_width, canvas_height] * 2)).tolist(),
                    "guide": guide, "guideAspect": guide_width / guide_height,
                    "canvas_size": [canvas_width, canvas_height], "boxes": (actual_boxes * scale).tolist(),
                    "masks": [MASKS[letter] for letter in word], "source_cells": source_cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("C:/bm-datasets/AngelinaDataset"))
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--styles", type=int, default=3)
    args = parser.parse_args()
    if args.styles < 1:
        raise ValueError("At least one training style is required")
    split = json.loads(args.source_split.read_text())
    revision = subprocess.check_output(["git", "-C", str(args.dataset), "rev-parse", "HEAD"], text=True).strip()
    if revision != split["dataset_revision"]:
        raise ValueError("Dataset revision differs from frozen training source split")
    train = set(split["images"]["train"])
    if train & (set(split["images"]["validation"]) | set(split["images"]["external"])):
        raise ValueError("Training source overlap with validation/external split")
    args.output.mkdir(parents=True, exist_ok=True)
    styles, records, used_families = [], [], set()
    for source in sorted(train):
        family = "/".join(source.split("/")[:2])
        if family in used_families:
            continue
        selected = select_cells(args.dataset, source)
        if selected is None:
            continue
        image, cells, pitch, tile_height, pitch_ratio = selected
        style_id = f"style-{len(styles) + 1:02d}"
        used_families.add(family)
        styles.append({"id": style_id, "source": source, "source_sha256": sha256(args.dataset / source),
                       "annotation_sha256": sha256((args.dataset / source).with_suffix(".csv")),
                       "source_cells": cells, "pitch_pixels": pitch, "tile_height_pixels": tile_height,
                       "observed_pitch_to_cell_height": pitch_ratio})
        sheet = Image.new("RGB", (500, len(WORDS) * 220), "white")
        draw = ImageDraw.Draw(sheet)
        for row, word in enumerate(WORDS):
            canvas, record = compose(image, cells, pitch, tile_height, word)
            stem = f"{style_id}-{word.lower()}"
            canvas.save(args.output / f"{stem}-composition.png")
            model_image = canvas.resize((480, 192), Image.Resampling.BILINEAR)
            model_image.save(args.output / f"{stem}.png")
            record.update(id=stem, image=f"{stem}.png", composition_image=f"{stem}-composition.png",
                          image_sha256=sha256(args.output / f"{stem}.png"),
                          composition_sha256=sha256(args.output / f"{stem}-composition.png"), style_id=style_id)
            assert len(record["masks"]) == len(word) == len(record["boxes"]) == len(record["source_cells"])
            records.append(record)
            preview = model_image.copy()
            preview_draw = ImageDraw.Draw(preview)
            selection = record["selection"]
            preview_draw.rectangle(tuple(np.asarray(selection) * [480, 192, 480, 192]), outline="red", width=1)
            sheet.paste(preview, (10, row * 220 + 23))
            draw.text((10, row * 220 + 4), f"{word} | {source[-52:]}", fill="black")
        sheet.save(args.output / f"{style_id}-review.png")
        if len(styles) == args.styles:
            break
    if len(styles) != args.styles:
        raise ValueError("Not enough training families with isolated A/K/M cells")
    manifest = {"schema_version": 1, "scope": "synthetic_composition_from_training_only",
                "not_external_or_physical_accuracy_evidence": True, "model_predictions_used": False,
                "dataset_revision": revision, "source_split_sha256": sha256(args.source_split),
                "generator_sha256": sha256(__file__), "words": list(WORDS),
                "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pillow": pillow_version},
                "model_input": {"width": 480, "height": 192, "color": "RGB", "normalization": "float32/255"},
                "selection_method": "Lexicographically first eligible training source per family; first annotation-order A/K/M cell with complete surrounding paper and no intersecting annotated neighbor. No image quality or model output selection.",
                "composition_method": "Unmodified photo cells from one source per style, translated to a shared baseline at observed source cell pitch. Outer one-pixel paper border blends into median border paper color. Repeated letters reuse identical source pixels at different positions. Final context uses Pillow BILINEAR480x192 and the deployed adaptive1..8 guide geometry.",
                "styles": styles, "records": records}
    serialized = json.dumps(manifest, indent=2) + "\n"
    (args.output / "manifest.json").write_text(serialized, encoding="utf-8")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(serialized, encoding="utf-8")
    print(f"Frozen {len(records)} synthetic repetition fixtures from {len(styles)} training photographs")
    print(f"Manifest SHA256: {sha256(args.output / 'manifest.json')}")


if __name__ == "__main__":
    main()
