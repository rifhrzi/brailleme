"""Frozen public whole-word, legacy-block, and negative regression evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
from benchmark_angelina import read_split
from word_data import WIDTH, HEIGHT, save_json
from word_decode import decode, summarize


def quality(image):
    luminance = image @ np.asarray([.2126, .7152, .0722], np.float32)
    mean, std = float(luminance.mean()), float(luminance.std())
    return "dark" if mean < .06 else "glare" if mean > .985 else "low_confidence" if std < .01 else ""


def summarize_public_words(predictions, records, thresholds, rows):
    metric = summarize(predictions, records, thresholds)
    # Zero-output placeholders preserve rejected-word counts, but decode them
    # as no_grid. Report the final capture reason already recorded in each row.
    metric["rejection_reasons"] = dict(Counter(row["rejection"] for row in rows if not row["predicted"]))
    return metric


def main():
    import tensorflow as tf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("C:/bm-datasets/AngelinaDataset"))
    parser.add_argument("--legacy", type=Path, default=Path(".braille-training-work/word-detector-20260905/public-evaluation/report.json"))
    args = parser.parse_args()
    if (args.run/"public-benchmark.json").exists():
        raise SystemExit("A frozen public evaluation already exists; preserve it and use a new candidate directory")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = args.run/"word_reader.tflite"
    selection = json.loads((args.run/"selected-validation-fullprecision.json").read_text())
    thresholds = selection["thresholds"]
    interpreter = tf.lite.Interpreter(model_path=str(model), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0]["index"], interpreter.get_output_details()[0]["index"]
    def infer(image, selection=None):
        pixels = np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR), np.float32)/255.
        interpreter.set_tensor(inp, pixels[None])
        interpreter.invoke()
        output = interpreter.get_tensor(out)[0].copy()
        reason = quality(pixels)
        result = decode(output, thresholds, selection) if not reason else {"text": "", "reason": reason, "cells": []}
        return output, result, reason
    fixtures = json.loads((args.data/"external-words.json").read_text())
    rows, predictions, raw_predictions, quality_reasons = [], [], [], []
    for index, record in enumerate(fixtures["records"]):
        with Image.open(args.data/record["image"]) as image:
            output, result, reason = infer(image, record.get("selection"))
        # Preserve outputs to independently reproduce threshold/decoder checks.
        # Quality-rejected output becomes no proposals for aggregate word metrics.
        predictions.append(np.zeros_like(output) if reason else output)
        raw_predictions.append(output)
        quality_reasons.append(reason)
        rows.append(dict(record, predicted=result["text"], rejection=result["reason"], detected_cells=result["cells"]))
        if index % 100 == 0:
            print(f"external {index+1}/{len(fixtures['records'])}", flush=True)
    metric = summarize_public_words(predictions, fixtures["records"], thresholds, rows)
    negatives = []
    for path in read_split(args.dataset, "not_braille/train.txt"):
        with Image.open(path) as image:
            _, result, _ = infer(image, [0., .1875, 1., .8125])
        negatives.append({"source": path.relative_to(args.dataset).as_posix(), "predicted": result["text"], "rejection": result["reason"]})
    legacy = []
    if args.legacy.is_file():
        for row in json.loads(args.legacy.read_text())["blocks"]:
            with Image.open(args.dataset/row["source"]) as image:
                _, result, _ = infer(image.crop(row["crop"]))
            legacy.append({"source": row["source"], "crop": row["crop"], "expected": row["expected"], "predicted": result["text"], "rejection": result["reason"]})
    negative_accepted = sum(bool(row["predicted"]) for row in negatives)
    report = {"scope": "public_offline_geometric_whole_words_not_physical_camera_evidence", "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "fixture_manifest_sha256": hashlib.sha256((args.data/"external-words.json").read_bytes()).hexdigest(), "threshold_source": "validation families only; frozen before external evaluation", "thresholds": thresholds, "whole_words": metric, "fixture_boundary_definition": fixtures["scope"], "fixture_human_boundary_reviewed": fixtures["human_boundary_reviewed"], "excluded_by_ground_truth_geometry": fixtures["excluded_by_ground_truth_geometry"], "negative_regression": {"total": len(negatives), "accepted": negative_accepted, "independent_holdout": False, "reason": "These 44 public negatives are also training/regression inputs."}, "legacy_arbitrary_block_regression": {"total": len(legacy), "accepted": sum(bool(row["predicted"]) for row in legacy), "correct": sum(row["predicted"]==row["expected"] for row in legacy), "is_complete_word_benchmark": False}, "numerical_internal_gates_pass": metric["exact_word_accuracy"] >= .75 and (metric["accepted_accuracy"] or 0) >= .95 and negative_accepted == 0, "release_ready": False, "limitations": ["Geometric word boundaries require visual review before acceptance authority.", "PIL antialiased bilinear does not establish native Android resize parity.", "No independent embossed-paper physical-camera evaluation performed.", "Missed terminal cells cannot be proven absent solely by gap checks."], "words": rows, "negative_rows": negatives, "legacy_rows": legacy}
    report['external_evaluation_status']='observed_development_regression'
    report['external_independent_holdout']=False
    report['external_observation_note']='The unchanged 607 fixtures were first evaluated for selected-v1. Later model development may be informed by aggregate failure diagnosis; gradients and checkpoint/threshold selection remain restricted to training and frozen held validation.'
    report['quality_rejected_before_decode']=sum(bool(reason) for reason in quality_reasons)
    report['cached_outputs_scope']='All 607 actual model outputs; quality_reasons stores the separate original-RGB pre-decoder rejection.'
    save_json(args.run/"public-benchmark.json", report)
    # Store actual model values even when the separate image-quality gate rejects
    # the capture. The quality reasons reproduce the app's pre-decoder decision.
    np.savez_compressed(args.run/"public-word-outputs.npz", outputs=np.stack(raw_predictions), quality_reasons=np.asarray(quality_reasons))
    print(json.dumps({k: v for k, v in report.items() if k not in ("words", "negative_rows", "legacy_rows")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
