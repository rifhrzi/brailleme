"""Export a frozen word checkpoint to float32 TFLite; verify backend parity."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from word_model import word_loss, image_normalization_spec
from word_data import HEIGHT, WIDTH, save_json
from word_decode import decode, DECODER_VERSION
import word_geometry_head  # register optional serialized sampling layers
import word_bn_affine  # register fixed-moment BN for trained affine candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = keras.models.load_model(args.run/"best-word.keras", compile=False)
    selected = json.loads((args.run/"selected-validation-fullprecision.json").read_text())
    if selected.get('decoder_version') != DECODER_VERSION or selected.get('oracle', {}).get('decoder_version') != DECODER_VERSION:
        raise SystemExit('Selected decoder version differs; use the selected source snapshot for export')
    for filename,expected_hash in selected.get('oracle',{}).get('source_sha256',{}).items():
        if hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest()!=expected_hash:
            raise SystemExit(f'Selection source changed: {filename}; restore its retained source before export')
    fixture_path=args.data/selected.get('oracle',{}).get('fixture_file','validation-words.json')
    records = json.loads(fixture_path.read_text())["records"]
    count = len(records)
    images = np.stack([np.asarray(Image.open(args.data/row["image"]).convert("RGB"), np.float32)/255. for row in records[:count]])
    oracle = np.load(args.run/"backend-fullprecision.npz")
    if str(oracle["fixture_sha256"]) != hashlib.sha256(fixture_path.read_bytes()).hexdigest():
        raise SystemExit("Validation inputs changed after parity oracle was frozen")
    if str(oracle["checkpoint_sha256"]) != hashlib.sha256((args.run/"best-word.keras").read_bytes()).hexdigest():
        raise SystemExit("Checkpoint changed after parity oracle was frozen")
    trained_outputs = oracle["outputs"][:count]
    tf_outputs = model.predict(images, batch_size=4, verbose=0)
    if not np.isfinite(tf_outputs).all():
        raise SystemExit("TensorFlow produced non-finite outputs; do not convert or promote")
    serving = tf.function(lambda image: model(image, training=False), input_signature=[tf.TensorSpec([1, HEIGHT, WIDTH, 3], tf.float32)])
    concrete = serving.get_concrete_function()
    # Freeze explicitly: TF2.20 otherwise emitted unfrozen resource variables
    # from this portable Keras model (a small artifact yielding NaNs).
    frozen = convert_variables_to_constants_v2(concrete)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    artifact = args.run/"word_reader.tflite"
    artifact.write_bytes(converter.convert())
    interpreter = tf.lite.Interpreter(model_path=str(artifact), num_threads=4)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    outputs = []
    for image in images:
        interpreter.set_tensor(inp["index"], image[None])
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(out["index"])[0])
    outputs = np.stack(outputs)
    backend_error = float(np.max(np.abs(trained_outputs-tf_outputs)))
    lite_error = float(np.max(np.abs(tf_outputs-outputs)))
    text_match = all(decode(a, selected["thresholds"], row.get("selection"))["text"] == decode(b, selected["thresholds"], row.get("selection"))["text"] for a, b, row in zip(trained_outputs, outputs, records))
    reason_match = all(decode(a, selected["thresholds"], row.get("selection"))["reason"] == decode(b, selected["thresholds"], row.get("selection"))["reason"] for a, b, row in zip(trained_outputs, outputs, records))
    report = {"status": "unverified_candidate", "source_checkpoint_sha256": hashlib.sha256((args.run/"best-word.keras").read_bytes()).hexdigest(), "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "artifact_bytes": artifact.stat().st_size, "input_shape": inp["shape"].tolist(), "output_shape": out["shape"].tolist(), "dtype": str(inp["dtype"]), "parity_samples": count, "parity_fixture_sha256": str(oracle["fixture_sha256"]), "training_backend_vs_tensorflow_max_abs": backend_error, "tensorflow_vs_tflite_max_abs": lite_error, "decoded_text_parity": text_match, "decoded_rejection_parity": reason_match, "parity_pass": backend_error<=5e-4 and lite_error<=5e-4 and text_match and reason_match, "native_preprocessing_parity_validated": False, "physical_camera_validated": False, "thresholds": selected["thresholds"]}
    report['imageNormalization']=image_normalization_spec(model)
    report['decoder_version']=selected['oracle']['decoder_version']
    if not np.isfinite(outputs).all():
        report.update(parity_pass=False, tensorflow_vs_tflite_max_abs=None)
    save_json(args.run/"export-report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["parity_pass"]:
        raise SystemExit("Model conversion parity failed; do not promote")


if __name__ == "__main__":
    main()
