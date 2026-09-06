"""Prepare frozen ground-truth geometry and RGB inputs; never load a model."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, manifests = [], {}
    pixels_sha = hashlib.sha256()
    with (args.output/'inputs.rgb').open('wb') as stream:
        for group, filename, count in [
            ('natural', 'validation-words.json', 135),
            ('stress', 'validation-stress-selection.json', 405),
            ('external', 'external-words.json', 607),
        ]:
            content = (args.data/filename).read_bytes()
            records = json.loads(content)['records']
            assert len(records) == count
            manifests[group] = {'file': filename, 'sha256': hashlib.sha256(content).hexdigest(), 'count': count}
            for index, record in enumerate(records):
                assert len(record['context_boxes']) == len(record['context_masks'])
                with Image.open(args.data/record['image']) as image:
                    rgb = image.convert('RGB').resize((480, 192), Image.Resampling.BILINEAR).tobytes()
                stream.write(rgb)
                pixels_sha.update(rgb)
                rows.append({'group': group, 'index': index, 'image': record['image'],
                             'source': record['source'], 'expected': record['expected'],
                             'selection': record['selection'], 'boxes': record['boxes'],
                             'masks': record['masks'], 'context_boxes': record['context_boxes'],
                             'context_masks': record['context_masks']})
    report = {'scope': 'Frozen annotated geometry and original RGB; no model outputs, training, threshold tuning or fixture changes.',
              'manifests': manifests, 'raw_inputs_sha256': pixels_sha.hexdigest(), 'rows': rows,
              'preparation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'geometry-inputs.json').write_text(json.dumps(report, separators=(',', ':'))+'\n', encoding='utf-8')
    print(json.dumps({'fixtures': len(rows), 'manifests': manifests}, indent=2))


if __name__ == '__main__':
    main()
