#!/usr/bin/env python3
"""Verify the promoted Keras and TFLite artifacts on one A-Z fixture each."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import tensorflow as tf

LABELS = "abcdefghijklmnopqrstuvwxyz"
INPUT_SIZE = (28, 28)
MAX_ABSOLUTE_DELTA = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixture(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            INPUT_SIZE, Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def select_fixtures(dataset_dir: Path) -> list[tuple[str, Path]]:
    image_paths = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    selected: list[tuple[str, Path]] = []
    for label in LABELS:
        path = next(
            (item for item in image_paths if item.name[:1].lower() == label),
            None,
        )
        if path is None:
            raise ValueError(f"Fixture kelas {label} tidak ditemukan di {dataset_dir}")
        selected.append((label, path))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    keras_path = args.keras.expanduser().resolve()
    tflite_path = args.tflite.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    fixtures = select_fixtures(dataset_dir)
    batch = np.stack([load_fixture(path) for _, path in fixtures])

    keras_model = tf.keras.models.load_model(keras_path)
    keras_probabilities = keras_model.predict(batch, verbose=0)

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    tflite_probabilities: list[np.ndarray] = []
    for tensor in batch:
        interpreter.set_tensor(input_details["index"], tensor[None, ...])
        interpreter.invoke()
        tflite_probabilities.append(
            interpreter.get_tensor(output_details["index"])[0].copy()
        )
    tflite_batch = np.stack(tflite_probabilities)

    samples = []
    for index, (expected_label, path) in enumerate(fixtures):
        keras_values = keras_probabilities[index]
        tflite_values = tflite_batch[index]
        keras_index = int(np.argmax(keras_values))
        tflite_index = int(np.argmax(tflite_values))
        maximum_delta = float(np.max(np.abs(keras_values - tflite_values)))
        samples.append(
            {
                "fixtureLabel": expected_label,
                "fixtureFile": path.name,
                "fixtureSha256": sha256_file(path),
                "kerasTop1": LABELS[keras_index],
                "tfliteTop1": LABELS[tflite_index],
                "top1Match": keras_index == tflite_index,
                "kerasConfidence": float(keras_values[keras_index]),
                "tfliteConfidence": float(tflite_values[tflite_index]),
                "maximumAbsoluteProbabilityDelta": maximum_delta,
            }
        )

    maximum_delta = max(
        sample["maximumAbsoluteProbabilityDelta"] for sample in samples
    )
    top1_match_count = sum(1 for sample in samples if sample["top1Match"])
    passed = top1_match_count == len(samples) and maximum_delta <= MAX_ABSOLUTE_DELTA
    report = {
        "schemaVersion": 1,
        "purpose": "Keras-to-TFLite conversion parity for the promoted candidate",
        "preprocessing": {
            "resize": [28, 28],
            "resizeInterpolation": "Pillow bilinear",
            "channelOrder": "RGB",
            "dtype": "float32",
            "normalization": "pixel / 255.0",
        },
        "artifacts": {
            "kerasSha256": sha256_file(keras_path),
            "tfliteSha256": sha256_file(tflite_path),
        },
        "sampleCount": len(samples),
        "top1MatchCount": top1_match_count,
        "maximumAllowedAbsoluteProbabilityDelta": MAX_ABSOLUTE_DELTA,
        "maximumObservedAbsoluteProbabilityDelta": maximum_delta,
        "passed": passed,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sampleCount",
        "top1MatchCount",
        "maximumObservedAbsoluteProbabilityDelta",
        "passed",
    )}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
