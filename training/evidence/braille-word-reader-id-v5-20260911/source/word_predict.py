"""Freeze full-float32 Torch validation outputs for conversion parity/calibration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from PIL import Image
import torch
import keras
from word_model import word_loss
from word_decode import calibrate
from word_data import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    checkpoint = args.run/"best-word.keras"
    fixture = args.data/"validation-words.json"
    records = json.loads(fixture.read_text())["records"]
    images = np.stack([np.asarray(Image.open(args.data/row["image"]).convert("RGB"), np.float32)/255. for row in records])
    model = keras.models.load_model(checkpoint, compile=False)
    predictions = model.predict(images, batch_size=args.batch_size, verbose=0)
    if not np.isfinite(predictions).all():
        raise SystemExit("Non-finite full-precision Torch oracle")
    fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    np.savez_compressed(args.run/"backend-fullprecision.npz", outputs=predictions, fixture_sha256=fixture_hash, checkpoint_sha256=checkpoint_hash)
    thresholds, metrics = calibrate(predictions, records)
    original = json.loads((args.run/"selected-validation.json").read_text())
    save_json(args.run/"selected-validation-fullprecision.json", dict(original, thresholds=thresholds, words=metrics, oracle={"backend": "torch", "tf32_enabled": False, "fixture_sha256": fixture_hash, "checkpoint_sha256": checkpoint_hash}))
    print(json.dumps({"fixture_sha256": fixture_hash, "checkpoint_sha256": checkpoint_hash, "thresholds": thresholds, "words": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
