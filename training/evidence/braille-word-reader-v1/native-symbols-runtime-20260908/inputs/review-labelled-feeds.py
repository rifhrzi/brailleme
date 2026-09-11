"""Review fixed labels and first-match source selection without reading predictions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/scripts'))
from word_symbols import parse_word_masks


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new report destination')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    fixtures = json.loads(args.fixtures.read_text(encoding='utf-8'))['records']
    assert manifest['fixture_sha256'] == digest(args.fixtures)
    assert manifest['generator_sha256'] == digest(Path(__file__).with_name('prepare-runtime-cases.py'))
    assert [row['expected'] for row in manifest['rows']] == ['abc', 'Abc', 'ABC', '2026', '001', 'a', 'aab', None]
    results = []
    for row in manifest['rows']:
        output = args.manifest.parent / row['output']
        assert digest(output) == row['output_sha256']
        with Image.open(output) as image:
            actual = image.convert('RGB')
        assert actual.size == (1024, 1024)
        if row['expected'] is None:
            assert actual.tobytes() == bytes([128]) * (1024 * 1024 * 3)
            results.append(dict(output=row['output'], expected=None, blank_pixels_verified=True))
            continue
        first = next(fixture for fixture in fixtures if fixture['expected'] == row['expected'])
        assert first == row['source']
        parsed = parse_word_masks(first['masks'])
        assert parsed['status'] == 'success' and parsed['text'] == row['expected']
        source = Path(first['image'])
        assert digest(source) == row['source_sha256']
        with Image.open(source) as image:
            resized = image.convert('RGB').resize((960, 384), Image.Resampling.BILINEAR)
        reconstructed = Image.new('RGB', (1024, 1024), (128, 128, 128))
        reconstructed.paste(resized, (32, 320))
        assert row['placement'] == [32, 320, 992, 704] and row['source_pixels_cropped'] is False
        assert actual.tobytes() == reconstructed.tobytes()
        results.append(dict(output=row['output'], expected=row['expected'], masks=first['masks'],
                            source_image=first['image'], first_matching_fixture_verified=True,
                            full_source_scaled_pixels_verified=True))
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), status='labels_and_composition_verified',
                  scope='Offline source label, fixed first-match selection and pixel composition only; no prediction outputs were read.',
                  manifest_sha256=digest(args.manifest), fixture_sha256=digest(args.fixtures),
                  reviewer_sha256=digest(__file__), parser_sha256=digest(ROOT / 'training/scripts/word_symbols.py'),
                  rows=results, physical_camera_validated=False, runtime_orientation_validated=False)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
