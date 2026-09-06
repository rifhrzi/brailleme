"""Replace the JS bundle of a verified internal APK for isolated diagnostics only."""

import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in (args.template.resolve(), args.bundle.resolve()):
        raise ValueError("Output must not replace input artifacts")
    if args.output.exists():
        raise ValueError("Output already exists; preserve previous diagnostic evidence")
    with zipfile.ZipFile(args.template) as source:
        if "assets/index.android.bundle" not in source.namelist():
            raise ValueError("Template APK has no embedded Android bundle")
        with zipfile.ZipFile(args.output, "w") as target:
            for entry in source.infolist():
                if entry.filename.startswith("META-INF/"):
                    continue
                data = args.bundle.read_bytes() if entry.filename == "assets/index.android.bundle" else source.read(entry.filename)
                target.writestr(entry, data)
    print(args.output)


if __name__ == "__main__":
    main()
