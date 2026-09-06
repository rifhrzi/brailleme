"""Prepare an isolated Hermes pixel/performance harness; never changes app routes."""

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-png", type=Path, required=True,
                        help="source.png produced by scripts/word-native-parity")
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    repo = Path(__file__).resolve().parents[2]
    if stage == repo or repo in stage.parents or stage in repo.parents:
        raise ValueError("Use a dedicated staging directory outside the source checkout")
    if stage.exists() and any(stage.iterdir()):
        raise ValueError("Staging directory must be empty; preserve previous evidence")
    encoded = args.source_png.read_bytes()
    with Image.open(args.source_png) as image:
        if image.size != (1440, 1920):
            raise ValueError("Expected the 1440x1920 native color fixture")
        expected = image.convert("RGB").crop((72, 700, 1368, 1220)).resize(
            (480, 192), Image.Resampling.BILINEAR).tobytes()
    fixture = {
        "sourceBase64": base64.b64encode(encoded).decode("ascii"),
        "expectedBase64": base64.b64encode(expected).decode("ascii"),
        "sourceSha256": hashlib.sha256(encoded).hexdigest(),
        "expectedRgbSha256": hashlib.sha256(expected).hexdigest(),
    }
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "output").mkdir()
    (stage / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
    scripts = Path(__file__).resolve().parent
    for name in ("package.json", "tsconfig.json"):
        shutil.copyfile(repo / name, stage / name)
    shutil.copyfile(scripts / "entry.tsx.template", stage / "index.tsx")
    shutil.copyfile(scripts / "metro.config.js.template", stage / "metro.config.js")
    for name in ("ml/word-image-processing.ts", "ml/word-model-contract.ts",
                 "ml/word-decoder.ts", "types/word-scan.ts", "types/live-scan.ts"):
        target = stage / "src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / "src" / name, target)
    print(json.dumps({"stage": str(stage), "source_sha256": fixture["sourceSha256"],
                      "expected_rgb_sha256": fixture["expectedRgbSha256"]}, indent=2))


if __name__ == "__main__":
    main()
