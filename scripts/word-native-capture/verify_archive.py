"""Verify the frozen camera diagnostic, optionally replaying pixels without ADB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO / "training/evidence/braille-word-reader-v1/native-camera-capture-diagnostic-v3"
FROZEN_REPORT_SHA256 = "afeace64ae29721b46de9598584f664659f021b32ff7fa5ad166af53143161c7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_rows(root: Path, rows: list[dict]) -> set[str]:
    checked = set()
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or row["path"] in checked:
            raise ValueError(f"Invalid or duplicate archive path: {row['path']}")
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Changed evidence bytes: {row['path']}")
        checked.add(row["path"])
    return checked


def verify(archive: Path, replay: bool) -> dict:
    index_path = archive / "archive-index.json"
    if sha(index_path) != (archive / "archive-index.sha256").read_text().split()[0]:
        raise ValueError("Archive index hash mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    checked = verify_rows(archive, index["files"])
    actual = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    if actual != checked | {"archive-index.json", "archive-index.sha256"}:
        raise ValueError("Archive file inventory differs from its index")
    verify_rows(REPO, index["replay_sources"])
    report = json.loads((archive / "report.json").read_text(encoding="utf-8"))
    if sha(archive / "report.json") != FROZEN_REPORT_SHA256:
        raise ValueError("Original frozen report changed")
    original_rows = report["source_rows"] + report["evidence_rows"]
    verify_rows(archive, original_rows)
    if any(p.lower().endswith((".apk", ".aab", ".tflite", ".keras")) for p in checked):
        raise ValueError("This archive must not contain an APK or recognition model")
    captures = json.loads((archive / "camera-capture-v3/report.json").read_text(encoding="utf-8"))
    if (captures["capture_count"], captures["passed_count"]) != (8, 6):
        raise ValueError("Frozen mixed diagnostic results changed")

    result = {
        "status": "verified_mixed_camera_diagnostic_not_main_apk_or_physical_evidence",
        "archive_files_checked": len(checked),
        "original_source_and_evidence_rows_checked": len(original_rows),
        "archive_index_sha256": sha(index_path),
        "report_sha256": sha(archive / "report.json"),
        "four_by_three_passed": 6,
        "sixteen_by_nine_failed": 2,
        "offline_pixel_replay": "not_requested",
    }
    if replay:
        # The original analyzer writes output. Run it only in a disposable copy;
        # preflight all inputs so its missing-file fallback can never call ADB.
        records = captures["records"]
        for record in records:
            folder = archive / "camera-capture-v3" / record["capture"]["id"]
            for filename in ("preview.png", "photo.png", "imageproxy.jpg"):
                if not (folder / filename).is_file():
                    raise ValueError(f"Missing offline input: {folder / filename}")
        with tempfile.TemporaryDirectory(prefix="word-camera-replay-") as temporary:
            staging = Path(temporary)
            shutil.copytree(archive / "camera-capture-v3", staging / "camera-capture-v3")
            analyzer = staging / "analyze-captures.py"
            shutil.copyfile(archive / "harness/analyze-captures.py", analyzer)
            subprocess.run(
                [sys.executable, str(analyzer), "--directory", "camera-capture-v3", "--offline"],
                check=True, capture_output=True, text=True,
            )
            replayed = json.loads((staging / "camera-capture-v3/report.json").read_text())
            if replayed != captures:
                raise ValueError("Pixel replay differs from the frozen per-capture report")
        # Independently reproduce the raw JPEG crop result, before EXIF rotation.
        import numpy as np
        from PIL import Image
        for pair in report["results"]["raw_jpeg_crop_comparisons"]:
            folder = archive / "camera-capture-v3"
            full = np.asarray(Image.open(folder / pair["full_capture"] / "imageproxy.jpg").convert("RGB"))
            short = np.asarray(Image.open(folder / pair["short_capture"] / "imageproxy.jpg").convert("RGB"))
            for check in pair["checks"]:
                y = check["raw_jpeg_crop_y"]
                crop = full[y:y + short.shape[0], :short.shape[1]]
                difference = np.abs(crop.astype(np.int16) - short.astype(np.int16))
                actual_check = {"raw_jpeg_crop_y": y, "mae": float(difference.mean()),
                                "max_abs": int(difference.max()),
                                "identical_channel_fraction": float((difference == 0).mean())}
                if actual_check != check:
                    raise ValueError("Raw JPEG crop comparison changed")
        result["offline_pixel_replay"] = "8_captures_and_2_raw_jpeg_pairs_exactly_reproduced"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--replay", action="store_true", help="Requires NumPy and Pillow; never starts ADB/emulator")
    args = parser.parse_args()
    print(json.dumps(verify(args.archive.resolve(), args.replay), indent=2))


if __name__ == "__main__":
    main()
