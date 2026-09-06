"""Choose a known validation word and preserve source pixels for native self-test."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
from word_decode import decode
from word_data import WIDTH, HEIGHT, save_json


ROOT = Path(__file__).resolve().parents[2]


def fixture_descriptor(record, dataset):
    x0, y0, x1, y1 = record["crop"]
    with Image.open(dataset/record["source"]) as original:
        padding = max(24, int(round((x1-x0)*.2)))
        bounds = [max(0, x0-padding), max(0, y0-padding), min(original.width, x1+padding), min(original.height, y1+padding)]
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    left, top, right, bottom = record["guide"]
    return {"width": width, "height": height, "source_crop": bounds,
            "roi": {"x": (left-bounds[0])/width, "y": (top-bounds[1])/height,
                    "width": (right-left)/width, "height": (bottom-top)/height},
            "expected_local_model_crop": [x0-bounds[0], y0-bounds[1], x1-bounds[0], y1-bounds[1]]}


def runtime_geometry(descriptors):
    completed = subprocess.run(["node", str(ROOT/"scripts/word-native-parity/fixture-geometry.cjs")],
                               input=json.dumps({"fixtures": descriptors}), text=True,
                               capture_output=True, check=True, cwd=ROOT)
    return json.loads(completed.stdout)


def check_source_input(record, descriptor, geometry, dataset, data):
    # A normalized guide can cross an integer pixel boundary after JS arithmetic.
    # Never infer on a cached input unless the actual runtime crop is identical.
    if geometry["crop"] != descriptor["expected_local_model_crop"]:
        return {"eligible": False, "reason": "runtime_integer_crop_differs"}
    with Image.open(dataset/record["source"]) as original:
        reconstructed = original.convert("RGB").crop(descriptor["source_crop"]).crop(geometry["crop"])
        rgb = np.asarray(reconstructed.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR))
    with Image.open(data/record["image"]) as cached:
        cached_rgb = np.asarray(cached.convert("RGB"))
    if not np.array_equal(rgb, cached_rgb):
        return {"eligible": False, "reason": "runtime_source_pixels_differ"}
    return {"eligible": True, "reason": "", "input_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}


def audit_geometry(records, dataset, data):
    descriptors = [fixture_descriptor(record, dataset) for record in records]
    reference = runtime_geometry(descriptors)
    rows = []
    for index, (record, descriptor, geometry) in enumerate(zip(records, descriptors, reference["fixtures"], strict=True)):
        rows.append({"index": index, "image": record["image"], "expected": record["expected"],
                     "expected_crop": descriptor["expected_local_model_crop"], "actual_crop": geometry["crop"],
                     **check_source_input(record, descriptor, geometry, dataset, data)})
    report = {"scope": "Validation startup-fixture geometry/input eligibility; no inference or native execution.",
              "total": len(rows), "eligible": sum(row["eligible"] for row in rows),
              "validation_manifest_sha256": hashlib.sha256((data/"validation-words.json").read_bytes()).hexdigest(),
              "source_sha256": reference["source_sha256"], "rows": rows}
    report["source_sha256"]["training/scripts/word_fixture.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return descriptors, reference["fixtures"], report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("C:/bm-datasets/AngelinaDataset"))
    parser.add_argument("--geometry-only", action="store_true", help="Audit all source crops/pixels without loading TensorFlow or a model")
    args = parser.parse_args()
    records = json.loads((args.data/"validation-words.json").read_text())["records"]
    descriptors, geometries, audit = audit_geometry(records, args.dataset, args.data)
    save_json(args.run/"native-fixture-geometry.json", audit)
    if args.geometry_only:
        print(json.dumps({"total": audit["total"], "eligible": audit["eligible"], "excluded": [row for row in audit["rows"] if not row["eligible"]]}, indent=2), flush=True)
        return
    import tensorflow as tf
    thresholds = json.loads((args.run/"export-report.json").read_text())["thresholds"]
    model = tf.lite.Interpreter(model_path=str(args.run/"word_reader.tflite"), num_threads=4)
    model.allocate_tensors()
    inp, out = model.get_input_details()[0]["index"], model.get_output_details()[0]["index"]
    candidates = []
    for index, record in enumerate(records):
        if not audit["rows"][index]["eligible"] or len(record["expected"]) < 3:
            continue
        with Image.open(args.data/record["image"]) as image:
            pixels = np.asarray(image.convert("RGB"), np.float32)/255.
        model.set_tensor(inp, pixels[None])
        model.invoke()
        output = model.get_tensor(out)[0]
        selection = geometries[index]["selection"]
        result = decode(output, thresholds, [selection["x"], selection["y"], selection["x"]+selection["width"], selection["y"]+selection["height"]])
        if result["text"] != record["expected"] or len(result["text"])<3:
            continue
        confidence = min(min(cell["score"], cell["dot_confidence"]) for cell in result["cells"])
        rank = (len(set(result["text"]))<len(result["text"]), confidence, min(len(result["text"]), 8))
        candidates.append((rank, index, output))
    chosen = None
    for _, index, output in sorted(candidates, key=lambda item: item[0], reverse=True):
        # Execute actual TS quality checks and tensor conversion before accepting
        # a self-test fixture. This still does not claim Android execution.
        with Image.open(args.data/records[index]["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        reference = runtime_geometry([{**descriptors[index], "rgb_base64": base64.b64encode(rgb.tobytes()).decode("ascii"),
                                       "output_float32_base64": base64.b64encode(output.tobytes()).decode("ascii"),
                                       "thresholds": thresholds}])["fixtures"][0]
        if reference["quality_reason"]:
            continue
        tensor_sha = hashlib.sha256((rgb.astype(np.float32)/255.).tobytes()).hexdigest()
        if reference["input_float32_sha256"] != tensor_sha or reference["selection"] != geometries[index]["selection"]:
            raise RuntimeError("Native self-test TypeScript input/selection differs from the evaluated input")
        if reference["decoded_text"] != records[index]["expected"]:
            raise RuntimeError(f"Native self-test TypeScript decoding differs for {records[index]['image']}: {reference}")
        chosen = (index, reference)
        break
    if chosen is None:
        raise SystemExit("No correctly recognized validation word for native self-test")
    index, reference = chosen
    record, descriptor = records[index], descriptors[index]
    bounds = descriptor["source_crop"]
    with Image.open(args.dataset/record["source"]) as original:
        original = original.convert("RGB")
        fixture = original.crop(bounds)
        artifact = args.run/"native-word-fixture.png"
        fixture.save(artifact)
    roi = descriptor["roi"]
    report = {"expectedText": record["expected"], "file": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "roi": roi, "source": record["source"], "source_sha256": hashlib.sha256((args.dataset/record["source"]).read_bytes()).hexdigest(), "source_crop": bounds, "source_model_context_crop": record["crop"], "width": fixture.width, "height": fixture.height, "guideAspect": record["guideAspect"], "purpose": "Original RGB source crop with surrounding pixels; not an already resized 480x192 model input.", "native_validated": False}
    report["runtime_input_reference"] = {**reference, "input_rgb_sha256": audit["rows"][index]["input_rgb_sha256"],
                                         "cached_validation_image": record["image"], "exact_cached_pixels": True,
                                         "model_sha256": hashlib.sha256((args.run/"word_reader.tflite").read_bytes()).hexdigest(),
                                         "thresholds": thresholds,
                                         "source_sha256": audit["source_sha256"]}
    save_json(args.run/"native-fixture.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
