"""Place every source-fixture pixel inside a padded emulator camera feed."""
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

root = Path(__file__).resolve().parents[2]
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
image_path = output / 'camera-word.png'
report_path = output / 'camera-word-source.json'
if image_path.exists() or report_path.exists():
    raise FileExistsError('Use a fresh evidence directory; fixtures are immutable')
model_dir = root / 'assets/models/braille-word-reader-v1'
manifest = json.loads((model_dir / 'manifest.json').read_text(encoding='utf-8'))
fixture = manifest['nativeSelfTest']
source_path = model_dir / fixture['file']
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha(source_path.read_bytes()) == fixture['sha256']
source = Image.open(source_path).convert('RGB')
assert source.size == (450, 257)
width, height = source.size
edge = ([source.getpixel((x, 0)) for x in range(width)]
        + [source.getpixel((x, height - 1)) for x in range(width)]
        + [source.getpixel((0, y)) for y in range(height)]
        + [source.getpixel((width - 1, y)) for y in range(height)])
neutral = round(statistics.median([statistics.mean(pixel) for pixel in edge]))
canvas = Image.new('RGB', (1024, 1024), (neutral, neutral, neutral))
left, top = (1024 - width) // 2, (1024 - height) // 2
canvas.paste(source, (left, top))
canvas.save(image_path)
reloaded = Image.open(image_path).convert('RGB')
assert reloaded.crop((left, top, left + width, top + height)).tobytes() == source.tobytes()
roi = fixture['roi']
placed_roi = {'x': (left + roi['x'] * width) / 1024,
              'y': (top + roi['y'] * height) / 1024,
              'width': roi['width'] * width / 1024,
              'height': roi['height'] * height / 1024}
report = {
    'schema_version': 1,
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Padded static emulator feed, all source pixels retained. No camera recognition or physical-device evidence.',
    'expected_text_current_decoder': 'rawenstwa',
    'source_file': str(source_path.relative_to(root)).replace('\\', '/'),
    'source_file_sha256': fixture['sha256'],
    'source_size': list(source.size),
    'source_rgb_sha256': sha(source.tobytes()),
    'source_native_self_test_metadata': fixture,
    'output_size': list(canvas.size),
    'source_placement_pixels': [left, top, left + width, top + height],
    'source_pixels_preserved_exactly': True,
    'scale': 1, 'clockwise_rotation_degrees': 0,
    'orientation_observed_in_new_apk': False,
    'neutral_padding_rgb': [neutral] * 3,
    'word_roi_in_camera_feed': placed_roi,
    'camera_word_sha256': sha(image_path.read_bytes()),
    'generator_sha256': sha(Path(__file__).read_bytes()),
}
report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
