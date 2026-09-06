"""Replay cached tensors and image quality only. Never loads or runs a model."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/scripts"))
from word_decode import decode  # noqa: E402


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark_path = args.run / "public-benchmark.json"
    tensor_path = args.run / "public-word-outputs.npz"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    records = benchmark["words"]
    with np.load(tensor_path, allow_pickle=False) as cached:
        outputs = cached["outputs"]
        # The revised benchmark keeps raw tensors and stores prechecks separately.
        raw_quality_outputs_retained = "quality_reasons" in cached.files
        archived_quality_reasons = cached["quality_reasons"].tolist() if raw_quality_outputs_retained else None
    if outputs.shape != (len(records), 48, 120, 11) or outputs.dtype != np.float32:
        raise ValueError(f"Unexpected frozen tensor contract: {outputs.shape}/{outputs.dtype}")
    if archived_quality_reasons is not None and len(archived_quality_reasons) != len(records):
        raise ValueError("Archived quality reasons do not cover every word")
    args.output.mkdir(parents=True, exist_ok=True)
    raw_outputs = args.output / "outputs.f32le"
    outputs.astype("<f4", copy=False).tofile(raw_outputs)
    # Compile the actual benchmark quality function without importing TensorFlow.
    benchmark_source = ROOT / "training/scripts/word_benchmark.py"
    syntax = ast.parse(benchmark_source.read_text(encoding="utf-8"))
    quality_node = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == "quality")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[quality_node], type_ignores=[]), str(benchmark_source), "exec"), namespace)
    quality = namespace["quality"]
    rows = []
    raw_inputs = args.output / "inputs.rgb"
    with raw_inputs.open("wb") as stream:
        for index, (record, output) in enumerate(zip(records, outputs)):
            with Image.open(args.data / record["image"]) as source:
                image = source.convert("RGB").resize((480, 192), Image.Resampling.BILINEAR)
                stream.write(image.tobytes())
                pixels = np.asarray(image, np.float32) / 255.
            quality_reason = quality(pixels)
            if archived_quality_reasons is not None and quality_reason != archived_quality_reasons[index]:
                raise ValueError(f"Archived image quality differs from replay at {index}")
            decoder = decode(output, benchmark["thresholds"], record.get("selection"))
            text, reason = ("", quality_reason) if quality_reason else (decoder["text"], decoder["reason"])
            if text != record["predicted"] or reason != record["rejection"]:
                raise ValueError(f"Python replay differs from frozen benchmark at {index}: {(text, reason)} vs {(record['predicted'], record['rejection'])}")
            rows.append({"index": index, "image": record["image"], "source": record["source"], "expected": record["expected"], "selection": record.get("selection", [0., 0., 1., 1.]), "python_text": text, "python_reason": reason, "python_quality_reason": quality_reason, "python_decoder_text": decoder["text"], "python_decoder_reason": decoder["reason"], "saved_output_is_zero_placeholder": bool(not raw_quality_outputs_retained and quality_reason and not np.any(output))})
    source_paths = ["training/scripts/word_decode.py", "training/scripts/word_data.py", "training/scripts/word_benchmark.py", "training/scripts/benchmark_angelina.py", "scripts/word-decoder-parity/prepare.py"]
    report = {"schema_version": 1, "scope": "Cached-output Python replay and RGB quality reference; no model inference", "created_at_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": benchmark["model_sha256"], "benchmark_sha256": sha(benchmark_path), "fixture_manifest_sha256": benchmark["fixture_manifest_sha256"], "cached_npz_sha256": sha(tensor_path), "raw_outputs_sha256": sha(raw_outputs), "raw_inputs_sha256": sha(raw_inputs), "output_shape": list(outputs.shape), "input_shape": [len(rows), 192, 480, 3], "archive_retains_raw_quality_rejects": raw_quality_outputs_retained, "thresholds": benchmark["thresholds"], "source_sha256": {path: sha(ROOT / path) for path in source_paths}, "rows": rows}
    (args.output / "python-reference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(rows), "quality_bypasses": sum(bool(row["python_quality_reason"]) for row in rows), "zero_placeholders": sum(row["saved_output_is_zero_placeholder"] for row in rows), "python_benchmark_replay_passed": True}, indent=2))


if __name__ == "__main__":
    main()
