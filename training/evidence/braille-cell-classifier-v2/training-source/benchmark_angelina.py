#!/usr/bin/env python3
"""Evaluate the frozen BrailleMe TFLite model on external Angelina pages.

This script deliberately does not train or tune the model. The predefined
upstream partitions are treated as one external, page-level benchmark because
none were used by BrailleMe. Results remain segmented by upstream partition.
This is not a replacement for native-camera or independently collected physical
validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


REPOSITORY_URL = "https://github.com/IlyaOvodov/AngelinaDataset"
PINNED_REVISION = "e13038d483bb58e17046d640e078dc4768276e9b"
DEFAULT_SPLITS = (
    "books/train.txt",
    "books/val.txt",
    "handwritten/train.txt",
    "handwritten/val.txt",
    "uploaded/test.txt",
)
DEFAULT_NEGATIVE_SPLIT = "not_braille/train.txt"
MODEL_LABELS = tuple("abcdefghijklmnopqrstuvwxyz")
MODEL_INPUT_SIZE = (28, 28)
CELL_CROP_SCALE = 1.25
NEGATIVE_CENTER_CROP_FRACTION = 0.55
CONFIDENCE_THRESHOLD = 0.90
INTERNAL_ACCURACY_TARGET = 0.75
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Angelina CSV labels are six-dot bitmasks. These are the English A-Z patterns
# declared in AngelinaDataset/src/letters.py, converted to integer bitmasks.
LETTER_BY_MASK = {
    1: "a",
    3: "b",
    9: "c",
    25: "d",
    17: "e",
    11: "f",
    27: "g",
    19: "h",
    10: "i",
    26: "j",
    5: "k",
    7: "l",
    13: "m",
    29: "n",
    21: "o",
    15: "p",
    31: "q",
    23: "r",
    14: "s",
    30: "t",
    37: "u",
    39: "v",
    58: "w",
    45: "x",
    61: "y",
    53: "z",
}


@dataclass(frozen=True)
class CellRecord:
    partition: str
    source_image: str
    annotation_row: int
    label: str
    mask: int
    left: float
    top: float
    right: float
    bottom: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_within(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path.replace("/", os.sep)).resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise ValueError(f"Path keluar dari root dataset: {relative_path}")
    return candidate


def read_split(dataset_root: Path, split_relative_path: str) -> list[Path]:
    split_path = resolve_within(dataset_root, split_relative_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split Angelina tidak ditemukan: {split_path}")

    split_root = split_path.parent
    image_paths: list[Path] = []
    for line_number, raw_line in enumerate(
        split_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        entry = raw_line.strip()
        if not entry:
            continue
        image_path = resolve_within(split_root, entry)
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Ekstensi gambar tidak didukung pada {split_path}:{line_number}: {entry}"
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Gambar pada {split_path}:{line_number} tidak ditemukan: {image_path}"
            )
        image_paths.append(image_path)

    if not image_paths:
        raise ValueError(f"Split Angelina kosong: {split_path}")
    if len(image_paths) != len(set(image_paths)):
        raise ValueError(f"Split Angelina memuat path gambar duplikat: {split_path}")
    return image_paths


def annotation_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".csv")


def parse_annotations(
    dataset_root: Path,
    image_paths: Sequence[Path],
    partition_by_image: dict[Path, str] | None = None,
) -> tuple[list[CellRecord], Counter[int], list[str]]:
    records: list[CellRecord] = []
    excluded_masks: Counter[int] = Counter()
    errors: list[str] = []

    for image_path in image_paths:
        relative_image = image_path.relative_to(dataset_root).as_posix()
        annotation_path = annotation_path_for(image_path)
        if not annotation_path.is_file():
            errors.append(f"Anotasi tidak ditemukan: {relative_image}")
            continue

        with annotation_path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source, delimiter=";")
            for row_number, row in enumerate(reader, start=1):
                if not row or all(not value.strip() for value in row):
                    continue
                if len(row) != 5:
                    errors.append(
                        f"Kolom anotasi bukan 5: {relative_image}:{row_number}"
                    )
                    continue
                try:
                    left, top, right, bottom = (float(value) for value in row[:4])
                    mask = int(row[4])
                except ValueError:
                    errors.append(
                        f"Nilai anotasi tidak valid: {relative_image}:{row_number}"
                    )
                    continue

                if not (
                    0 <= left < right <= 1
                    and 0 <= top < bottom <= 1
                ):
                    errors.append(
                        f"Bounding box di luar 0..1: {relative_image}:{row_number}"
                    )
                    continue

                label = LETTER_BY_MASK.get(mask)
                if label is None:
                    excluded_masks[mask] += 1
                    continue
                records.append(
                    CellRecord(
                        partition=(partition_by_image or {}).get(
                            image_path.resolve(), "unspecified"
                        ),
                        source_image=relative_image,
                        annotation_row=row_number,
                        label=label,
                        mask=mask,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                    )
                )

    return records, excluded_masks, errors


def square_crop_box(
    image_width: int,
    image_height: int,
    normalized_box: tuple[float, float, float, float],
    scale: float = CELL_CROP_SCALE,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = normalized_box
    x0 = left * image_width
    y0 = top * image_height
    x1 = right * image_width
    y1 = bottom * image_height
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * scale
    side = max(1.0, min(side, float(image_width), float(image_height)))

    crop_left = min(max(0.0, center_x - side / 2), image_width - side)
    crop_top = min(max(0.0, center_y - side / 2), image_height - side)
    crop_right = crop_left + side
    crop_bottom = crop_top + side
    return (
        int(round(crop_left)),
        int(round(crop_top)),
        int(round(crop_right)),
        int(round(crop_bottom)),
    )


def preprocess_cell(image: Image.Image, record: CellRecord) -> np.ndarray:
    crop_box = square_crop_box(
        image.width,
        image.height,
        (record.left, record.top, record.right, record.bottom),
    )
    crop = image.crop(crop_box).resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32) / 255.0


def preprocess_negative(image: Image.Image) -> np.ndarray:
    side = max(
        1,
        int(min(image.width, image.height) * NEGATIVE_CENTER_CROP_FRACTION),
    )
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    crop = image.crop((left, top, left + side, top + side)).resize(
        MODEL_INPUT_SIZE, Image.Resampling.BILINEAR
    )
    return np.asarray(crop, dtype=np.float32) / 255.0


class TfliteRunner:
    def __init__(self, model_path: Path):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        input_detail = self.interpreter.get_input_details()[0]
        output_detail = self.interpreter.get_output_details()[0]
        if list(input_detail["shape"]) != [1, 28, 28, 3]:
            raise ValueError(f"Kontrak input TFLite tidak cocok: {input_detail['shape']}")
        if list(output_detail["shape"]) != [1, 26]:
            raise ValueError(f"Kontrak output TFLite tidak cocok: {output_detail['shape']}")
        if input_detail["dtype"] != np.float32 or output_detail["dtype"] != np.float32:
            raise ValueError("Benchmark hanya mendukung model float32 BrailleMe saat ini.")
        self.input_index = int(input_detail["index"])
        self.output_index = int(output_detail["index"])

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        batch = np.expand_dims(tensor, axis=0).astype(np.float32, copy=False)
        self.interpreter.set_tensor(self.input_index, batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0].copy()


def current_git_revision(dataset_root: Path) -> str | None:
    if not (dataset_root / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def validate_dataset_revision(dataset_root: Path, expected_revision: str) -> str:
    observed_revision = current_git_revision(dataset_root)
    if observed_revision is None:
        raise ValueError(
            "Root Angelina harus berupa checkout Git agar revision dapat diverifikasi."
        )
    if observed_revision != expected_revision.lower():
        raise ValueError(
            "Revision Angelina tidak cocok: "
            f"expected {expected_revision}, observed {observed_revision}"
        )
    return observed_revision


def batched_images(
    dataset_root: Path, records: Sequence[CellRecord]
) -> Iterable[tuple[CellRecord, np.ndarray]]:
    active_relative_path: str | None = None
    active_image: Image.Image | None = None
    try:
        for record in records:
            if record.source_image != active_relative_path:
                if active_image is not None:
                    active_image.close()
                active_relative_path = record.source_image
                active_image = Image.open(
                    resolve_within(dataset_root, record.source_image)
                ).convert("RGB")
            assert active_image is not None
            yield record, preprocess_cell(active_image, record)
    finally:
        if active_image is not None:
            active_image.close()


def evaluate_cells(
    runner: TfliteRunner,
    dataset_root: Path,
    records: Sequence[CellRecord],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    predictions: list[dict[str, object]] = []
    expected: list[str] = []
    predicted: list[str] = []

    for record, tensor in batched_images(dataset_root, records):
        probabilities = runner.predict(tensor)
        predicted_index = int(np.argmax(probabilities))
        predicted_label = MODEL_LABELS[predicted_index]
        confidence = float(probabilities[predicted_index])
        expected.append(record.label)
        predicted.append(predicted_label)
        predictions.append(
            {
                "source_image": record.source_image,
                "partition": record.partition,
                "annotation_row": record.annotation_row,
                "expected_label": record.label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": predicted_label == record.label,
                "accepted_at_0_90": confidence >= CONFIDENCE_THRESHOLD,
            }
        )
        if len(predictions) % 5000 == 0:
            print(
                f"Evaluated {len(predictions)}/{len(records)} Angelina A-Z cells...",
                flush=True,
            )

    report = classification_report(
        expected,
        predicted,
        labels=list(MODEL_LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(expected, predicted, labels=list(MODEL_LABELS))
    accepted = [item for item in predictions if item["accepted_at_0_90"]]
    accepted_correct = sum(bool(item["correct"]) for item in accepted)
    metrics = {
        "sample_count": len(predictions),
        "source_page_count": len({item["source_image"] for item in predictions}),
        "top1_accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "minimum_per_class_recall": min(
            float(report[label]["recall"]) for label in MODEL_LABELS
        ),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "accepted_count": len(accepted),
        "coverage_at_threshold": len(accepted) / len(predictions),
        "accepted_accuracy": (
            accepted_correct / len(accepted) if accepted else None
        ),
        "internal_top1_accuracy_target": INTERNAL_ACCURACY_TARGET,
        "internal_target_passed": float(report["accuracy"])
        >= INTERNAL_ACCURACY_TARGET,
    }
    metrics_by_partition = {
        partition: summarize_predictions(
            [item for item in predictions if item["partition"] == partition]
        )
        for partition in sorted({str(item["partition"]) for item in predictions})
    }
    return predictions, metrics, {
        "labels": list(MODEL_LABELS),
        "matrix": matrix.tolist(),
        "classification_report": report,
        "metrics_by_partition": metrics_by_partition,
    }


def summarize_predictions(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    expected = [str(item["expected_label"]) for item in rows]
    predicted = [str(item["predicted_label"]) for item in rows]
    report = classification_report(
        expected,
        predicted,
        labels=list(MODEL_LABELS),
        output_dict=True,
        zero_division=0,
    )
    supported_labels = [
        label for label in MODEL_LABELS if float(report[label]["support"]) > 0
    ]
    accepted = [item for item in rows if item["accepted_at_0_90"]]
    return {
        "sample_count": len(rows),
        "source_page_count": len({item["source_image"] for item in rows}),
        "supported_class_count": len(supported_labels),
        "missing_classes": sorted(set(MODEL_LABELS) - set(supported_labels)),
        "top1_accuracy": float(report["accuracy"]),
        "macro_f1_all_26_classes": float(report["macro avg"]["f1-score"]),
        "minimum_supported_class_recall": min(
            float(report[label]["recall"]) for label in supported_labels
        ),
        "accepted_count": len(accepted),
        "coverage_at_0_90": len(accepted) / len(rows),
        "accepted_accuracy": (
            sum(bool(item["correct"]) for item in accepted) / len(accepted)
            if accepted
            else None
        ),
    }


def evaluate_negatives(
    runner: TfliteRunner, dataset_root: Path, image_paths: Sequence[Path]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for image_path in image_paths:
        with Image.open(image_path) as source:
            tensor = preprocess_negative(source.convert("RGB"))
        probabilities = runner.predict(tensor)
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])
        predictions.append(
            {
                "source_image": image_path.relative_to(dataset_root).as_posix(),
                "predicted_label": MODEL_LABELS[predicted_index],
                "confidence": confidence,
                "false_accept_at_0_90": confidence >= CONFIDENCE_THRESHOLD,
            }
        )
    false_accept_count = sum(
        bool(item["false_accept_at_0_90"]) for item in predictions
    )
    return predictions, {
        "scene_count": len(predictions),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "false_accept_count": false_accept_count,
        "false_accept_rate": false_accept_count / len(predictions),
        "note": (
            "Satu center crop per gambar non-Braille; ini guardrail publik/offline, "
            "bukan negative corpus kamera native."
        ),
    }


def write_predictions_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=os.environ.get("BRAILLE_ANGELINA_DATASET_DIR"),
        help="Checkout AngelinaDataset pada revision yang dipin.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=repository_root
        / "assets/models/braille-cell-classifier-v1/braille_model.tflite",
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help=(
            "Partisi relatif Angelina. Ulangi argumen untuk beberapa partisi; "
            "default: books/val, handwritten/val, uploaded/test."
        ),
    )
    parser.add_argument("--negative-split", default=DEFAULT_NEGATIVE_SPLIT)
    parser.add_argument(
        "--expected-revision",
        default=PINNED_REVISION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / ".braille-training-work/angelina-benchmark",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset_dir is None:
        raise SystemExit(
            "Isi --dataset-dir atau BRAILLE_ANGELINA_DATASET_DIR dengan checkout Angelina."
        )

    dataset_root = args.dataset_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Root Angelina tidak ditemukan: {dataset_root}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model TFLite tidak ditemukan: {model_path}")

    revision = validate_dataset_revision(dataset_root, args.expected_revision)
    split_names = tuple(args.splits or DEFAULT_SPLITS)
    image_paths_by_split = {
        split_name: read_split(dataset_root, split_name)
        for split_name in split_names
    }
    image_paths = [
        image_path
        for split_name in split_names
        for image_path in image_paths_by_split[split_name]
    ]
    if len(image_paths) != len(set(image_paths)):
        raise ValueError("Partisi benchmark Angelina saling tumpang tindih.")
    partition_by_image = {
        image_path.resolve(): split_name
        for split_name, paths in image_paths_by_split.items()
        for image_path in paths
    }
    negative_image_paths = read_split(dataset_root, args.negative_split)
    records, excluded_masks, annotation_errors = parse_annotations(
        dataset_root, image_paths, partition_by_image
    )
    if annotation_errors:
        raise ValueError(
            "Anotasi Angelina tidak valid:\n- " + "\n- ".join(annotation_errors[:20])
        )
    if not records:
        raise ValueError("Subset Angelina tidak menghasilkan crop A-Z.")

    class_counts = Counter(record.label for record in records)
    missing_classes = sorted(set(MODEL_LABELS) - set(class_counts))
    data_quality = {
        "dataset_repository": REPOSITORY_URL,
        "dataset_revision": revision,
        "splits": list(split_names),
        "source_page_count": len(image_paths),
        "source_page_counts_by_split": {
            split_name: len(paths)
            for split_name, paths in image_paths_by_split.items()
        },
        "annotation_count": len(records) + sum(excluded_masks.values()),
        "included_a_z_cell_count": len(records),
        "excluded_non_a_z_cell_count": sum(excluded_masks.values()),
        "excluded_mask_counts": {
            str(mask): count for mask, count in sorted(excluded_masks.items())
        },
        "class_counts": {label: class_counts[label] for label in MODEL_LABELS},
        "missing_classes": missing_classes,
        "annotation_error_count": len(annotation_errors),
        "duplicate_record_count": len(records) - len(set(records)),
        "license_status": "no_explicit_license_file_observed_at_repository_root",
        "intended_role": "external_public_offline_benchmark_only",
    }
    if missing_classes:
        raise ValueError(f"Subset Angelina tidak mencakup kelas: {missing_classes}")

    runner = TfliteRunner(model_path)
    predictions, metrics, details = evaluate_cells(runner, dataset_root, records)
    negative_predictions, negative_metrics = evaluate_negatives(
        runner, dataset_root, negative_image_paths
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(output_dir / "predictions.csv", predictions)
    write_predictions_csv(
        output_dir / "negative_predictions.csv", negative_predictions
    )
    save_json(output_dir / "data_quality.json", data_quality)
    save_json(output_dir / "classification_report.json", details["classification_report"])
    save_json(
        output_dir / "confusion_matrix.json",
        {"labels": details["labels"], "matrix": details["matrix"]},
    )

    benchmark_report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data_quality,
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "labels": list(MODEL_LABELS),
            "input": "1x28x28x3 float32 RGB, pixel/255",
        },
        "preprocessing": {
            "cell_source": "Angelina normalized annotation bounding box",
            "cell_crop": "square around annotation center",
            "cell_crop_scale": CELL_CROP_SCALE,
            "resize": "28x28 bilinear",
            "channel_order": "RGB",
            "normalization": "pixel/255",
            "negative_crop": f"center square {NEGATIVE_CENTER_CROP_FRACTION:.2f}",
        },
        "metrics": metrics,
        "metrics_by_partition": details["metrics_by_partition"],
        "negative_guardrail": negative_metrics,
        "interpretation": {
            "target_scope": "external_public_offline_character benchmark",
            "passes_internal_75_percent_target": metrics["internal_target_passed"],
            "production_release_evidence": False,
            "native_camera_accuracy_claim_allowed": False,
            "caveats": [
                "Subset tidak pernah dipakai untuk training atau tuning model BrailleMe ini.",
                "Crop berasal dari anotasi ground-truth, bukan deteksi ROI kamera aplikasi.",
                "Angelina tidak memiliki LICENSE eksplisit pada root revision yang dipin.",
                "Benchmark ini tidak menggantikan physical holdout atau native-camera parity.",
            ],
        },
    }
    save_json(output_dir / "benchmark_report.json", benchmark_report)
    print(json.dumps(benchmark_report, indent=2, ensure_ascii=False))
    print(f"\nEvidence tersimpan di: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
