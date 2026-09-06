"""Decode sparse IEEE-754 fixtures without loading a model or TensorFlow."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/scripts'))
from word_decode import decode

fixtures = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
rows = []
for fixture in fixtures['cases']:
    output = np.zeros((48, 120, 11), np.float32)
    for row, column, *bits in fixture['peaks']:
        output[row, column] = np.asarray(bits, np.uint32).view(np.float32)
    selection = fixture['selection']
    edges = fixture.get('selection_xyxy', [selection['x'], selection['y'],
                        selection['x']+selection['width'], selection['y']+selection['height']])
    result = decode(output, fixture['thresholds'], edges)
    rows.append({'id': fixture['id'], 'text': result['text'], 'reason': result['reason']})

assert not any(name == 'torch' or name == 'tensorflow' or name.startswith('tensorflow.') for name in sys.modules)
print(json.dumps({'rows': rows, 'numpy_version': np.__version__, 'python_version': sys.version,
                  'model_loaded': False, 'torch_imported': False, 'tensorflow_imported': False,
                  'decoder_sha256': hashlib.sha256((ROOT/'training/scripts/word_decode.py').read_bytes()).hexdigest()}))
