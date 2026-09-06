"""Compare Android Bitmap evidence with actual TS geometry and Pillow pixels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version


def differences(actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "max_channel_error_0_255": float(diff.max()),
        "mean_channel_error_0_255": float(diff.mean()),
        "rmse_0_255": float(np.sqrt(np.mean(diff * diff))),
        "exact_pixel_fraction": float(np.all(diff == 0, axis=-1).mean()),
    }


def unfiltered_bilinear(source, width, height):
    """Diagnostic center-sampled bilinear, without a shrinking antialias kernel.

    This is a mathematical comparator, not a claim of bit-exact Skia emulation.
    """
    x = np.maximum(0, (np.arange(width) + .5) * source.shape[1] / width - .5)
    y = np.maximum(0, (np.arange(height) + .5) * source.shape[0] / height - .5)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, source.shape[1] - 1), np.minimum(y0 + 1, source.shape[0] - 1)
    wx, wy = (x - x0)[None, :, None], (y - y0)[:, None, None]
    top = source[y0[:, None], x0[None, :]] * (1 - wx) + source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1 - wx) + source[y1[:, None], x1[None, :]] * wx
    return np.clip(np.floor(top * (1 - wy) + bottom * wy + .5), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads((args.evidence / "native.json").read_text())
    geometry = {row["id"]: row for row in json.loads(args.geometry.read_text())["cases"]}
    base = Image.open(args.evidence / "source.png").convert("RGB")
    rows = []
    for row in native["cases"]:
        name = row["id"]
        expected = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if row["mirror_before_rotation"] else base.copy()
        transpose = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}
        if row["rotation_clockwise"]:
            expected = expected.transpose(transpose[row["rotation_clockwise"]])
        upright = np.asarray(Image.open(args.evidence / f"{name}-upright.png").convert("RGB"))
        reference = geometry[name]
        crop = expected.crop(tuple(int(value) for value in row["context_crop_pixels"]))
        native_crop = np.asarray(Image.open(args.evidence / f"{name}-crop.png").convert("RGB"))
        actual = np.fromfile(args.evidence / f"{name}.rgb", np.uint8).reshape(192, 480, 3)
        raw = np.fromfile(args.evidence / f"{name}.raw", np.uint8).reshape(192, 480, 4)
        pillow = np.asarray(crop.resize((480, 192), Image.Resampling.BILINEAR))
        tensor = np.fromfile(args.evidence / f"{name}.ts.f32", "<f4").reshape(192, 480, 3)
        tensor_rgb = np.clip(np.rint(tensor * 255), 0, 255).astype(np.uint8)
        rows.append({
            "id": name,
            "rotation_and_mirroring": differences(upright, np.asarray(expected)),
            "crop_matches_typescript": row["context_crop_pixels"] == reference["context_crop_pixels"],
            "native_crop_pixel_error": differences(native_crop, np.asarray(crop)),
            "selection_max_error": float(np.max(np.abs(np.array(row["selection_photo"]) - reference["selection_photo"]))),
            "pillow_bilinear_vs_native": differences(actual, pillow),
            "typescript_antialias_vs_pillow": differences(tensor_rgb, pillow),
            "typescript_float32_vs_pillow_max_error": float(np.max(np.abs(tensor - pillow.astype(np.float32) / np.float32(255)))),
            "center_sampled_bilinear_vs_native": differences(actual, unfiltered_bilinear(np.asarray(crop), 480, 192)),
            "raw_as_rgba_vs_get_pixels": differences(raw[..., :3], actual),
            "raw_as_bgra_vs_get_pixels": differences(raw[..., [2, 1, 0]], actual),
            "native_rgb_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "native_crop_raw_sha256": hashlib.sha256((args.evidence / f"{name}-crop.raw").read_bytes()).hexdigest(),
            "typescript_float32_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
        })
    repository = Path(__file__).resolve().parents[2]
    sources = ["src/ml/word-image-processing.ts", "src/ml/word-model-contract.ts", "scripts/word-native-parity/WordNativeParity.java", "scripts/word-native-parity/geometry-reference.cjs", "scripts/word-native-parity/pixel-reference.cjs", "scripts/word-native-parity/compare.py"]
    report = {
        "scope": native["scope"],
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "pillow_version": pillow_version,
        "source_code_sha256": {source: hashlib.sha256((repository / source).read_bytes()).hexdigest() for source in sources},
        "device": {key: native[key] for key in ("sdk", "device", "fingerprint", "native_byte_order")},
        "argb8888_raw_bytes_for_rgb_11_22_33": native["argb8888_raw_bytes_for_rgb_11_22_33"],
        "rotation_mirroring_geometry_passed": all(row["rotation_and_mirroring"]["max_channel_error_0_255"] == 0 and row["crop_matches_typescript"] and row["native_crop_pixel_error"]["max_channel_error_0_255"] == 0 and row["selection_max_error"] < 1e-9 for row in rows),
        "pillow_native_resize_pixel_parity_passed": all(row["pillow_bilinear_vs_native"]["max_channel_error_0_255"] == 0 for row in rows),
        "typescript_antialias_pillow_pixel_parity_passed": all(row["typescript_float32_vs_pillow_max_error"] == 0 for row in rows),
        "nitro_declared_bgra_matches_native_bytes": all(row["raw_as_bgra_vs_get_pixels"]["max_channel_error_0_255"] == 0 for row in rows),
        "limitations": [
            "Synthetic emulator Bitmap primitives do not validate real camera capture, autofocus, lighting, or accessibility.",
            "The Nitro/JS bridge is inspected in source but is not invoked by this standalone DEX harness.",
            "Decoder/model prediction parity is separate from pixel and coordinate parity.",
            "High-frequency colored patterns intentionally stress channel ordering and downsampling; they are not an accuracy benchmark.",
        ],
        "cases": rows,
    }
    destination = args.evidence / "comparison.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
