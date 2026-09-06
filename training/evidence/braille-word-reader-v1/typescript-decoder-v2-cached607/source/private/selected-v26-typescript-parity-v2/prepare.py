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


ROOT = Path.cwd()
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
    protocol_path = Path(__file__).with_name('protocol.json')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in protocol['bound_artifacts'].items():
        assert sha(ROOT / path) == digest, path
    tensor_path = ROOT / protocol['cache_file']
    replay = json.loads((ROOT / protocol['replay_file']).read_text(encoding='utf-8'))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    records = benchmark["words"]
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    def memory():
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters
    budget = 350 * 1024 * 1024
    assert memory().WorkingSetSize + len(records) * 48 * 120 * 11 * 4 + 40 * 1024 * 1024 < budget, 'Insufficient process memory headroom for bounded cache replay'
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
            image_path = args.data / record['image']
            assert sha(image_path) == replay['input_image_sha256'][record['image']], ('Changed existing PNG', index)
            with Image.open(image_path) as source:
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
            rows.append({"index": index, "image": record["image"], "source": record["source"], "expected": record["expected"], "selection": record.get("selection", [0., 0., 1., 1.]), "python_text": text, "python_reason": reason, "python_quality_reason": quality_reason, "python_decoder_text": decoder["text"], "python_decoder_reason": decoder["reason"], "python_decoder_cells": decoder["cells"], "saved_output_is_zero_placeholder": bool(not raw_quality_outputs_retained and quality_reason and not np.any(output))})
    source_paths = ["training/scripts/word_decode.py", "training/scripts/word_data.py", "training/scripts/word_benchmark.py", "training/scripts/benchmark_angelina.py", "scripts/word-decoder-parity/prepare.py"]
    report = {"schema_version": 1, "scope": "Cached-output Python replay and RGB quality reference; no model inference", "created_at_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": benchmark["model_sha256"], "benchmark_sha256": sha(benchmark_path), "fixture_manifest_sha256": benchmark["fixture_manifest_sha256"], "cached_npz_sha256": sha(tensor_path), "raw_outputs_sha256": sha(raw_outputs), "raw_inputs_sha256": sha(raw_inputs), "output_shape": list(outputs.shape), "input_shape": [len(rows), 192, 480, 3], "archive_retains_raw_quality_rejects": raw_quality_outputs_retained, "thresholds": benchmark["thresholds"], "source_sha256": {path: sha(ROOT / path) for path in source_paths}, "rows": rows}
    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    assert not any(name in sys.modules for name in ['torch', 'tensorflow', 'keras'])
    peak = memory().PeakWorkingSetSize
    assert peak < budget, ('Process exceeded 350 MiB budget', peak)
    report.update(protocol_file=str(protocol_path.relative_to(ROOT)), protocol_sha256=sha(protocol_path),
        decoder_version=protocol['decoder_version'], peak_working_set_bytes=peak,
        existing_png_hashes_verified=len(rows), no_model_backend_imported=True)
    report['source_sha256'].update(protocol['source_sha256'])
    (args.output / "python-reference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(rows), "quality_bypasses": sum(bool(row["python_quality_reason"]) for row in rows), "zero_placeholders": sum(row["saved_output_is_zero_placeholder"] for row in rows), "python_benchmark_replay_passed": True}, indent=2))


if __name__ == "__main__":
    main()
