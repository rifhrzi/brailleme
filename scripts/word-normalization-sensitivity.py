"""Validation-only preprocessing sensitivity of a frozen, failed word model."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image, __version__ as pillow_version
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/scripts"))
from word_decode import decode  # noqa: E402

EXPECTED_MODEL = "36fa9c2532ad526b8e3cf93a4b4fd43758aeb1b78ecfac80dee047ba732d0dc8"
EXPECTED_FIXTURES = "2ee26bbffd0457830eebe81dd48beb0b4bcb3a6dafe7957544f2ca3a0984c346"
VARIANTS = ("rgb_baseline", "standardize_gray_std_0.05", "standardize_gray_std_0.10", "standardize_gray_std_0.15")
STD_TARGETS = dict(zip(VARIANTS[1:], (.05, .10, .15)))
MINIMUM_AVAILABLE_BYTES = 1024 ** 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality(image):
    luminance = image @ np.asarray([.2126, .7152, .0722], np.float32)
    mean, std = float(luminance.mean()), float(luminance.std())
    return "dark" if mean < .06 else "glare" if mean > .985 else "low_confidence" if std < .01 else ""


def transform(rgb, variant):
    if variant == "rgb_baseline":
        return rgb
    means = rgb.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = rgb.std(axis=(0, 1), ddof=1, dtype=np.float64).astype(np.float32)
    standardized = (rgb - means) / np.maximum(stds, np.float32(1e-6))
    gray = standardized.mean(axis=-1, dtype=np.float32)
    gray = np.clip(np.float32(.5) + np.float32(STD_TARGETS[variant]) * gray, 0., 1.)
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)


def metric(rows, text_key):
    count = len(rows)
    accepted = sum(bool(row[text_key]) for row in rows)
    correct = sum(row[text_key] == row["expected"] for row in rows)
    return {"total": count, "accepted": accepted, "correct": correct,
            "exact_word_accuracy": correct / count if count else 0,
            "accepted_accuracy": correct / accepted if accepted else None}


def summarize(rows):
    output = {}
    for name, key in (("decoder_only", "decoder_text"), ("quality_after_variant", "post_quality_text"),
                      ("original_rgb_quality", "original_quality_text")):
        output[name] = metric(rows, key)
        output[name]["by_length"] = {str(length): metric([row for row in rows if len(row["expected"]) == length], key)
                                     for length in range(1, 13) if any(len(row["expected"]) == length for row in rows)}
    output["original_quality_rejections"] = dict(Counter(row["original_quality_reason"] for row in rows if row["original_quality_reason"]))
    output["post_transform_quality_rejections"] = dict(Counter(row["post_quality_reason"] for row in rows if row["post_quality_reason"]))
    output["decoder_rejections"] = dict(Counter(row["decoder_reason"] for row in rows if row["decoder_reason"]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.run / "word_reader.tflite"
    fixture_path = args.data / "validation-words.json"
    selection_path = args.run / "selected-validation-fullprecision.json"
    assert sha(model_path) == EXPECTED_MODEL, "This diagnostic is pinned to the failed selected-v1 model"
    assert sha(fixture_path) == EXPECTED_FIXTURES, "Natural validation fixture set changed"
    records = json.loads(fixture_path.read_text(encoding="utf-8"))["records"]
    assert len(records) == 135
    thresholds = json.loads(selection_path.read_text(encoding="utf-8"))["thresholds"]
    assert {row["source"].split('/')[0] for row in records} <= {"books", "handwritten"}, "No external inputs allowed"
    process = psutil.Process()
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    memory = {"minimum_available_bytes": psutil.virtual_memory().available, "maximum_rss_bytes": process.memory_info().rss}
    def check_memory():
        available = psutil.virtual_memory().available
        memory["minimum_available_bytes"] = min(memory["minimum_available_bytes"], available)
        memory["maximum_rss_bytes"] = max(memory["maximum_rss_bytes"], process.memory_info().rss)
        if available < MINIMUM_AVAILABLE_BYTES:
            raise MemoryError("Stopped diagnostic: less than1GiB system memory available")
    check_memory()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert input_details["shape"].tolist() == [1, 192, 480, 3]
    check_memory()
    start = time.monotonic()
    results = {name: [] for name in VARIANTS}
    stopped = None
    try:
        for index, record in enumerate(records):
            check_memory()
            with Image.open(args.data / record["image"]) as source:
                rgb = np.asarray(source.convert("RGB").resize((480, 192), Image.Resampling.BILINEAR), np.float32) / 255.
            original_quality = quality(rgb)
            for variant in VARIANTS:
                pixels = transform(rgb, variant)
                post_quality = quality(pixels)
                interpreter.set_tensor(input_details["index"], pixels[None])
                interpreter.invoke()
                output = interpreter.get_tensor(output_details["index"])[0]
                decoded = decode(output, thresholds, record.get("selection"))
                results[variant].append({"index": index, "image": record["image"], "source": record["source"], "expected": record["expected"],
                    "decoder_text": decoded["text"], "decoder_reason": decoded["reason"],
                    "original_quality_reason": original_quality, "post_quality_reason": post_quality,
                    "post_quality_text": "" if post_quality else decoded["text"],
                    "original_quality_text": "" if original_quality else decoded["text"]})
            if (index + 1) % 30 == 0:
                print(f"validation {index+1}/135; elapsed{time.monotonic()-start:.1f}s; available{memory['minimum_available_bytes']/1024**3:.2f}GiB", flush=True)
    except MemoryError as error:
        stopped = str(error)
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen failed-model sensitivity diagnosis on135 natural validation words only; no external evaluation, retraining, calibration or promotion",
        "model_sha256": EXPECTED_MODEL, "fixture_manifest_sha256": EXPECTED_FIXTURES,
        "selected_thresholds_file_sha256": sha(selection_path), "thresholds": thresholds,
        "preprocessing": {"baseline": "RGB float32 /255, unchanged internal model Rescaling(2,-1)",
            "variants_fixed_before_inference": list(VARIANTS), "formula": "per-channel z=(RGB-mean)/max(sample_std,1e-6); mean z across3 channels; clip(.5+target_std*zgray,0,1); repeat3 RGB channels",
            "channel_statistics": "float64 accumulation, cast mean/std tofloat32, sample std ddof1; transform remainsfloat32",
            "std_targets": STD_TARGETS, "clipping": "After grayscale; values outside[0,1] clipped; no uint8 requantization",
            "source_inspiration": "https://raw.githubusercontent.com/IlyaOvodov/AngelinaReader/master/data_utils/data.py#L107-L119",
            "source_distinction": "AngelinaReader subtracts channel means, divides by3*std with a configurable std floor, then averages channels. These bounded[0,1] probes are adaptations for this different frozen model, not reproduction of that model's input contract."},
        "quality_assumptions": {"formula": "luminance=.2126R+.7152G+.0722B; reject mean<.06, mean>.985, orstd<.01",
            "reported_modes": "decoder_only; unchanged quality thresholds applied after each transform; unchanged precheck applied to original RGB before every transform. All rejections count as word failures."},
        "backend": {"tensorflow": tf.__version__, "numpy": np.__version__, "pillow": pillow_version, "tflite_threads": 2, "gpu_disabled": True, "windows_priority": "BELOW_NORMAL" if os.name == 'nt' else None},
        "elapsed_seconds": time.monotonic()-start, "memory": memory, "completed": stopped is None, "stop_reason": stopped,
        "source_sha256": {"scripts/word-normalization-sensitivity.py": sha(__file__), "training/scripts/word_decode.py": sha(ROOT/'training/scripts/word_decode.py')},
        "variants": {name: summarize(rows) for name, rows in results.items()}, "rows": results,
        "limitations": ["Normalization changes the input distribution of a model trained onrawRGB; this does not predict success after retraining.", "Validation-only results cannot establish external or physical-camera accuracy.", "No thresholds, clipping parameters, preprocessing choice, or model was promoted from this diagnostic."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding="utf-8")
    print(json.dumps({"completed": report["completed"], "elapsed_seconds": report["elapsed_seconds"], "memory": memory,
        "variants": {name: {key: value for key, value in summary.items() if key in ('decoder_only', 'quality_after_variant', 'original_rgb_quality')} for name, summary in report['variants'].items()}}, indent=2), flush=True)
    if stopped:
        raise SystemExit(stopped)


if __name__ == "__main__":
    main()
