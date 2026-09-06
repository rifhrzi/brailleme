"""Run a frozen word model on separately labeled diagnostic compositions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
import tensorflow as tf
from word_decode import decode
from word_data import WIDTH, HEIGHT, save_json
from word_benchmark import quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fixtures", "--manifest", dest="fixtures", type=Path, required=True)
    parser.add_argument("--report-name", default="repetition-diagnostics.json")
    args = parser.parse_args()
    report_path=args.run/args.report_name
    if report_path.exists():
        raise SystemExit('Use a new report name; frozen diagnostic results must not be overwritten')
    manifest = json.loads(args.fixtures.read_text())
    conversion = json.loads((args.run/"export-report.json").read_text())
    thresholds = conversion['thresholds']
    artifact = args.run/"word_reader.tflite"
    artifact_hash=hashlib.sha256(artifact.read_bytes()).hexdigest()
    if artifact_hash != conversion['artifact_sha256'] or not conversion['parity_pass']:
        raise ValueError('Diagnostics require the exact model that passed conversion parity')
    model = tf.lite.Interpreter(model_path=str(artifact), num_threads=4)
    model.allocate_tensors()
    inp, out = model.get_input_details()[0]["index"], model.get_output_details()[0]["index"]
    rows = []
    for record in manifest["records"]:
        image_path=args.fixtures.parent/record['image']
        image_hash=hashlib.sha256(image_path.read_bytes()).hexdigest()
        if image_hash != record.get('image_sha256'):
            raise ValueError(f'Diagnostic image changed or lacks a frozen hash: {image_path}')
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR), np.float32)/255.
        reason=quality(pixels)
        if reason:
            result={'text':'','reason':reason,'cells':[]}
        else:
            model.set_tensor(inp, pixels[None])
            model.invoke()
            result=decode(model.get_tensor(out)[0], thresholds, record.get('selection'))
        expected_reject=bool(record.get('expected_reject',False))
        correct=not result['text'] and bool(result['reason']) if expected_reject else result['text']==record['expected']
        rows.append({"image": record["image"], 'image_sha256':image_hash,
                     'category':record.get('category','repetition'), 'expected_reject':expected_reject,
                     "expected": record["expected"], "predicted": result["text"], "rejection": result["reason"],
                     "detected_cell_count": len(result["cells"]), "correct": correct})
    report = {"scope": manifest["scope"], "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "fixture_manifest_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(), "thresholds": thresholds, "total": len(rows), "correct": sum(row["correct"] for row in rows), "accepted": sum(bool(row["predicted"]) for row in rows), "used_for_external_checkpoint_or_threshold_selection": False, "is_natural_whole_word_benchmark": False, "rows": rows}
    report['by_category']={category:{'total':sum(row['category']==category for row in rows),
                                   'correct':sum(row['category']==category and row['correct'] for row in rows),
                                   'accepted':sum(row['category']==category and bool(row['predicted']) for row in rows)}
                           for category in sorted({row['category'] for row in rows})}
    report['original_rgb_quality_checked_before_inference']=True
    report['all_frozen_image_hashes_verified']=True
    report['source_sha256']={f'training/scripts/{name}':hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                            for name in ('word_diagnostics.py', 'word_benchmark.py', 'word_decode.py', 'word_data.py')}
    save_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
