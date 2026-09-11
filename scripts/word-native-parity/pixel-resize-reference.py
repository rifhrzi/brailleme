"""Independent Pillow reference for the host Java/TypeScript resizer audit."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, __version__ as pillow_version

manifest = Path(sys.argv[1])
rows = json.loads(manifest.read_text())
channels = {'RGB': (3, (0, 1, 2)), 'BGR': (3, (2, 1, 0)),
            'RGBA': (4, (0, 1, 2)), 'RGBX': (4, (0, 1, 2)),
            'BGRA': (4, (2, 1, 0)), 'BGRX': (4, (2, 1, 0)),
            'ARGB': (4, (1, 2, 3)), 'XRGB': (4, (1, 2, 3)),
            'ABGR': (4, (3, 2, 1)), 'XBGR': (4, (3, 2, 1))}
for row in rows:
    pixel_format = 'RGBA' if row['androidBitmap'] and row['format'] == 'BGRA' else row['format']
    stride, indices = channels[pixel_format]
    raw = np.fromfile(manifest.parent / row['input'], dtype=np.uint8).reshape(row['height'], row['width'], stride)
    rgb = raw[:, :, list(indices)]
    resized = np.asarray(Image.fromarray(rgb).resize((480, 192), Image.Resampling.BILINEAR))
    resized.tofile(manifest.parent / (row['name'] + '.pillow.rgb'))
    (resized.astype(np.float32) / 255.0).astype('<f4').tofile(manifest.parent / (row['name'] + '.pillow.f32'))
print(json.dumps({'pillow_version': pillow_version, 'numpy_version': np.__version__, 'cases': len(rows)}))
