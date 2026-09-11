"""Freeze deterministic first-match labelled emulator inputs before runtime prediction."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[2]
destination = Path('C:/bm-symbols-apk-20260908/evidence/labelled-feeds-v1')
destination.mkdir(exist_ok=False)
fixture_path = Path('C:/bm-symbols-20260908/validation-symbols.json')
fixtures = json.loads(fixture_path.read_text(encoding='utf-8'))['records']
digest = lambda value: hashlib.sha256(value).hexdigest()
rows = []
for expected in ['abc', 'Abc', 'ABC', '2026', '001', 'a', 'aab']:
    record = next(row for row in fixtures if row['expected'] == expected)
    source_path = Path(record['image'])
    with Image.open(source_path) as original:
        source = original.convert('RGB')
    scaled = source.resize((source.width * 2, source.height * 2), Image.Resampling.BILINEAR)
    assert scaled.width <= 1024 and scaled.height <= 1024
    canvas = Image.new('RGB', (1024, 1024), (128, 128, 128))
    left, top = (1024 - scaled.width) // 2, (1024 - scaled.height) // 2
    canvas.paste(scaled, (left, top))
    output = destination / (record['symbol_group'] + '-' + expected + '.png')
    canvas.save(output)
    rows.append({'expected': expected, 'selection': 'first matching fixture record, independent of predictions',
                 'source': record, 'source_sha256': digest(source_path.read_bytes()),
                 'source_size': list(source.size), 'output': output.name,
                 'output_sha256': digest(output.read_bytes()), 'output_size': list(canvas.size),
                 'scale': 2, 'interpolation': 'pillow_bilinear', 'clockwise_rotation_degrees': 0,
                 'placement': [left, top, left + scaled.width, top + scaled.height],
                 'source_pixels_cropped': False,
                 'scope': 'Synthetic recomposed public photo cells, further resized for static emulator feed. No natural or physical-camera claim.'})
blank = destination / 'blank.png'
Image.new('RGB', (1024, 1024), (128, 128, 128)).save(blank)
rows.append({'expected': None, 'expected_runtime': 'searching with shutter disabled when no target exists',
             'output': blank.name, 'output_sha256': digest(blank.read_bytes()), 'scope': 'Synthetic blank negative input'})
report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'generator_sha256': digest(Path(__file__).read_bytes()),
          'fixture_sha256': digest(fixture_path.read_bytes()), 'runtime_predictions_observed': False,
          'note': 'Feed orientation/placement must be checked visually before captures; record any later placement change as a separate input.',
          'rows': rows}
(destination / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'output': str(destination), 'cases': len(rows)}))
