"""Bind a previously selected validation fixture to the actual exported model."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path('C:/bm-symbols-20260908/overall80-20260911')
sys.path.insert(0, str(EXPERIMENT / 'final-method-source'))
from word_overall_v5_evaluate import make_predictor
from word_overall_v5_zoom import infer_decisions
from word_decode_id_v4 import decode

sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
method = json.loads((EXPERIMENT / 'final-method-selection.json').read_text())
source = Path('C:/bm-symbols-20260908/accuracy80-20260910/data/validation/000101.png')
expected = 'Yys'
pixels = np.asarray(Image.open(source).convert('RGB'), np.float32) / np.float32(255)
assert pixels.shape == (192, 480, 3)
predict, backend = make_predictor('tflite', method)
result = infer_decisions(predict, pixels[None], method['thresholds'], [None],
                         policy=method['selected_policy'], batch_size=1, decoder=decode)
decision = result['decisions'][0]
assert decision['text'] == expected, decision
value = dict(status='passed', purpose='Prepared validation-input startup smoke; not camera geometry evidence',
             fixture_source=str(source), fixture_sha256=sha(source), fixture_size=[480, 192],
             expected_text=expected, source_split='previously_observed_validation',
             test_used_for_selection=False, model_sha256=method['tflite_sha256'],
             checkpoint_sha256=method['checkpoint_sha256'], thresholds=method['thresholds'],
             decoder_version='six-dot-id-v4-experimental', selected_policy=method['selected_policy'],
             runtime=backend, decision=decision, route=result['routes'][0],
             input_float32_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
             script_sha256=sha(__file__), native_camera_validated=False)
destination = Path(__file__).with_name('startup-fixture-tflite.json')
with destination.open('x', encoding='utf-8') as stream:
    json.dump(value, stream, indent=2)
print(json.dumps(value, indent=2), flush=True)
