"""Analytical color landmarks; no Braille pixels, model, or predicted crop."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw

out = Path(__file__).parent / 'camera-capture'
out.mkdir(exist_ok=True)
image = Image.new('RGB', (1920, 1440), (220, 220, 220))
draw = ImageDraw.Draw(image)
for x in range(0, 1920, 120):
    draw.line((x, 0, x, 1440), fill=(170, 170, 170), width=2)
for y in range(0, 1440, 120):
    draw.line((0, y, 1920, y), fill=(170, 170, 170), width=2)
markers = []
for name, x, y, rgb in [
    ('red', 768, 576, (230, 30, 40)),
    ('magenta', 1152, 576, (215, 35, 195)),
    ('cyan', 1152, 864, (20, 180, 200)),
    ('blue', 768, 864, (35, 65, 230)),
]:
    draw.ellipse((x-24, y-24, x+24, y+24), fill=rgb)
    markers.append({'id': name, 'center': [x, y], 'radius': 24, 'rgb': list(rgb)})
draw.text((700, 440), 'TOP / ORIGINAL SENSOR FEED', fill=(0, 0, 0), font_size=36)
draw.text((700, 960), 'BOTTOM', fill=(0, 0, 0), font_size=36)
path = out / 'landmark-feed.png'
image.save(path)
record = {'schema_version': 1, 'scope': 'isolated_synthetic_camera_geometry_only',
          'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'size': list(image.size), 'markers': markers,
          'tolerance_photo_pixels': 4, 'tolerance_preview_pixels': 3,
          'comparison': 'Actual preview landmark centroids projected via native preview sensor points and actual Photo matrix, compared with independently detected full upright photo centroids. No predicted word or fitted landmark transform is used.'}
(out / 'feed-manifest.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
