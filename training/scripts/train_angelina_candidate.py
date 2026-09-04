#!/usr/bin/env python3
"""Train an isolated BrailleMe experiment with Angelina plus the Kaggle source.

The output stays under ``.braille-training-work`` and is never copied into the
application automatically. Angelina has no explicit root LICENSE at the pinned
revision, so this is an evaluation experiment, not a redistributable release
artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from benchmark_angelina import (
    CONFIDENCE_THRESHOLD,
    INTERNAL_ACCURACY_TARGET,
    MODEL_INPUT_SIZE,
    MODEL_LABELS,
    PINNED_REVISION,
    TfliteRunner,
    batched_images,
    evaluate_cells,
    parse_annotations,
    read_split,
    save_json,
    sha256_file,
    validate_dataset_revision,
)


SEED = 42
ANGELINA_TRAIN_SPLITS = ("books/train.txt", "handwritten/train.txt")
ANGELINA_VALIDATION_SPLITS = ("books/val.txt", "handwritten/val.txt")
ANGELINA_EXTERNAL_TEST_SPLITS = ("uploaded/test.txt",)
MAX_EPOCHS = 12
BATCH_SIZE = 64


def set_determinism() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
    if enable_determinism is not None:
        enable_determinism()


def load_records(dataset_root: Path, splits: Sequence[str]):
    image_paths_by_split = {
        split: read_split(dataset_root, split) for split in splits
    }
    partition_by_image = {
        path.resolve(): split
        for split, paths in image_paths_by_split.items()
        for path in paths
    }
    image_paths = [path for paths in image_paths_by_split.values() for path in paths]
    records, excluded, errors = parse_annotations(
        dataset_root, image_paths, partition_by_image
    )
    if errors:
        raise ValueError("Anotasi Angelina tidak valid:\n- " + "\n- ".join(errors[:20]))
    return records, excluded


def records_to_uint8(dataset_root: Path, records, name: str):
    tensors: list[np.ndarray] = []
    labels: list[int] = []
    label_indexes = {label: index for index, label in enumerate(MODEL_LABELS)}
    for index, (record, tensor) in enumerate(
        batched_images(dataset_root, records), start=1
    ):
        tensors.append(np.rint(tensor * 255).astype(np.uint8))
        labels.append(label_indexes[record.label])
        if index % 10000 == 0:
            print(f"Prepared {index}/{len(records)} {name} cells...", flush=True)
    return np.stack(tensors), np.asarray(labels, dtype=np.int64)


def kaggle_to_uint8(dataset_root: Path):
    image_paths = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    tensors: list[np.ndarray] = []
    labels: list[int] = []
    for path in image_paths:
        label = path.name[:1].lower()
        if label not in MODEL_LABELS:
            continue
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                MODEL_INPUT_SIZE, Image.Resampling.BILINEAR
            )
            tensors.append(np.asarray(image, dtype=np.uint8))
        labels.append(MODEL_LABELS.index(label))
    if not tensors:
        raise ValueError(f"Dataset Kaggle tidak menghasilkan gambar A-Z: {dataset_root}")
    return np.stack(tensors), np.asarray(labels, dtype=np.int64)


def create_model():
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 3), name="braille_input"),
            layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_1"),
            layers.MaxPooling2D((2, 2), name="pool_1"),
            layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
            layers.MaxPooling2D((2, 2), name="pool_2"),
            layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_3"),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_1"),
            layers.Dropout(0.35, name="dropout"),
            layers.Dense(26, activation="softmax", name="predictions"),
        ],
        name="braille_cell_classifier_angelina_experiment",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_keras(model, tensors: np.ndarray, labels: np.ndarray):
    probabilities = model.predict(tensors.astype(np.float32) / 255.0, verbose=0)
    predicted = probabilities.argmax(axis=1)
    report = classification_report(
        labels,
        predicted,
        labels=list(range(26)),
        target_names=list(MODEL_LABELS),
        output_dict=True,
        zero_division=0,
    )
    confidence = probabilities.max(axis=1)
    accepted = confidence >= CONFIDENCE_THRESHOLD
    supported = [
        label for label in MODEL_LABELS if float(report[label]["support"]) > 0
    ]
    return {
        "sample_count": int(len(labels)),
        "supported_class_count": len(supported),
        "missing_classes": sorted(set(MODEL_LABELS) - set(supported)),
        "top1_accuracy": float(report["accuracy"]),
        "macro_f1_all_26_classes": float(report["macro avg"]["f1-score"]),
        "minimum_supported_class_recall": min(
            float(report[label]["recall"]) for label in supported
        ),
        "accepted_count": int(accepted.sum()),
        "coverage_at_0_90": float(accepted.mean()),
        "accepted_accuracy": (
            float((predicted[accepted] == labels[accepted]).mean())
            if accepted.any()
            else None
        ),
        "classification_report": report,
    }


def build_parser():
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angelina-dir",
        type=Path,
        default=os.environ.get("BRAILLE_ANGELINA_DATASET_DIR"),
    )
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=os.environ.get("BRAILLE_DATASET_DIR"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / ".braille-training-work/angelina-finetune",
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.angelina_dir is None or args.kaggle_dir is None:
        raise SystemExit(
            "Isi BRAILLE_ANGELINA_DATASET_DIR dan BRAILLE_DATASET_DIR."
        )
    if not 1 <= args.epochs <= MAX_EPOCHS:
        raise ValueError(f"--epochs harus antara 1 dan {MAX_EPOCHS}.")

    set_determinism()
    angelina_root = args.angelina_dir.expanduser().resolve()
    kaggle_root = args.kaggle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = validate_dataset_revision(angelina_root, PINNED_REVISION)

    train_records, train_excluded = load_records(
        angelina_root, ANGELINA_TRAIN_SPLITS
    )
    validation_records, validation_excluded = load_records(
        angelina_root, ANGELINA_VALIDATION_SPLITS
    )
    test_records, test_excluded = load_records(
        angelina_root, ANGELINA_EXTERNAL_TEST_SPLITS
    )

    train_images, train_labels = records_to_uint8(
        angelina_root, train_records, "Angelina train"
    )
    kaggle_images, kaggle_labels = kaggle_to_uint8(kaggle_root)
    train_images = np.concatenate([train_images, kaggle_images], axis=0)
    train_labels = np.concatenate([train_labels, kaggle_labels], axis=0)
    validation_images, validation_labels = records_to_uint8(
        angelina_root, validation_records, "Angelina validation"
    )
    test_images, test_labels = records_to_uint8(
        angelina_root, test_records, "Angelina external test"
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=np.arange(26), y=train_labels
    )
    class_weights = {
        index: float(weight) for index, weight in enumerate(class_weights_array)
    }
    training_generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=4,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.05,
        brightness_range=(0.85, 1.15),
        fill_mode="reflect",
    ).flow(
        train_images,
        train_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    model = create_model()
    best_model_path = output_dir / "best_candidate.keras"
    history = model.fit(
        training_generator,
        epochs=args.epochs,
        validation_data=(
            validation_images.astype(np.float32) / 255.0,
            validation_labels,
        ),
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                best_model_path,
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ],
        verbose=2,
    )

    validation_metrics = evaluate_keras(
        model, validation_images, validation_labels
    )
    test_metrics = evaluate_keras(model, test_images, test_labels)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    tflite_path = output_dir / "braille_model_angelina_experiment.tflite"
    tflite_path.write_bytes(tflite_bytes)

    # Re-run the external test through the exact TFLite artifact to catch
    # conversion or label-order regressions.
    tflite_runner = TfliteRunner(tflite_path)
    _, tflite_test_metrics, _ = evaluate_cells(
        tflite_runner, angelina_root, test_records
    )

    report = {
        "schema_version": 1,
        "status": "experimental_not_for_redistribution",
        "dataset": {
            "angelina_revision": revision,
            "angelina_train_splits": list(ANGELINA_TRAIN_SPLITS),
            "angelina_validation_splits": list(ANGELINA_VALIDATION_SPLITS),
            "angelina_external_test_splits": list(ANGELINA_EXTERNAL_TEST_SPLITS),
            "kaggle_path": str(kaggle_root),
            "train_counts": {
                label: int((train_labels == index).sum())
                for index, label in enumerate(MODEL_LABELS)
            },
            "train_excluded_non_a_z": int(sum(train_excluded.values())),
            "validation_excluded_non_a_z": int(sum(validation_excluded.values())),
            "test_excluded_non_a_z": int(sum(test_excluded.values())),
        },
        "training": {
            "seed": SEED,
            "requested_epochs": args.epochs,
            "completed_epochs": len(history.history["loss"]),
            "batch_size": BATCH_SIZE,
            "class_weights": {
                MODEL_LABELS[index]: weight
                for index, weight in class_weights.items()
            },
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
        },
        "validation": validation_metrics,
        "external_test_keras": test_metrics,
        "external_test_tflite": tflite_test_metrics,
        "artifact": {
            "tflite_path": str(tflite_path),
            "tflite_sha256": sha256_file(tflite_path),
        },
        "decision": {
            "internal_75_percent_target": INTERNAL_ACCURACY_TARGET,
            "target_passed": tflite_test_metrics["top1_accuracy"]
            >= INTERNAL_ACCURACY_TARGET,
            "copy_to_application": False,
            "reason": (
                "Kandidat tetap terisolasi sampai benchmark ditinjau, lisensi "
                "Angelina jelas, dan validasi kamera native tersedia."
            ),
        },
    }
    save_json(output_dir / "training_report.json", report)
    print(json.dumps(report["decision"], indent=2), flush=True)
    print(
        json.dumps(
            {
                "validation_accuracy": validation_metrics["top1_accuracy"],
                "external_test_accuracy": tflite_test_metrics["top1_accuracy"],
                "external_test_coverage_at_0_90": tflite_test_metrics[
                    "coverage_at_threshold"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
