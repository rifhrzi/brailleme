"""Independent current/run19 Patches replay, without importing Keras/models."""
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT/'.braille-training-work/word-scan-v1-20260905'
OUT = Path(__file__).resolve().parent
DATA = WORK/'data'
sys.path.insert(0, str(ROOT/'training/scripts'))
import word_data as current
from word_point_metadata import load_point_states
from word_point_geometry import OccupiedPointState, visible_partial_candidates, safe_center_mask
from word_sampling import BalancedFamilyOrder
from word_decode import decode

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
names = ['word_data.py', 'word_train.py', 'word_point_geometry.py', 'word_point_metadata.py', 'word_model.py', 'word_sampling.py']
bindings = {name: sha(ROOT/'training/scripts'/name) for name in names}
for name in names:
    assert sha(WORK/'run20/source'/name) == bindings[name]
assert sha(WORK/'run19/source/word_model.py') == bindings['word_model.py']
assert sha(WORK/'run19/source/word_sampling.py') == bindings['word_sampling.py']
spec = importlib.util.spec_from_file_location('run19_word_data', WORK/'run19/source/word_data.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
config = json.loads((WORK/'run19/training-config.json').read_text())
all_records = json.loads((DATA/config['train_manifest']).read_text())
metadata_path = WORK/'dsbi-point-metadata-v1/points.json'
states, metadata_audit = load_point_states(metadata_path, DATA, config['supplemental_source_indexes'], all_records)
metadata = json.loads(metadata_path.read_text())
dsbi_records = [r for r in all_records if r['image'] in states]
assert len(dsbi_records) == len(states) == 2338

# Reconstruct source points independently from raw TXT, without its parser.
index = json.loads((WORK/'dsbi-training-contexts-v2/source-index.json').read_text())
raw_points = {}
for source in index['sources']:
    path = Path(index['root'])/source['original_annotation']
    assert sha(path) == source['annotation_sha256']
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if not lines: continue
    xs, ys = [float(v) for v in lines[1].split()], [float(v) for v in lines[2].split()]
    for number, line in enumerate(lines[3:], 4):
        if not line.strip(): continue
        values = [int(v) for v in line.split()]
        row, col = values[:2]
        mask = sum(bit<<i for i, bit in enumerate(values[2:]))
        if not mask: continue
        raw_points[(source['source'], row, col, number, mask)] = [[xs[2*(col-1)+x], ys[3*(row-1)+y]] for x in (0, 1) for y in (0, 1, 2)]
point_occurrences = 0
for r in dsbi_records:
    m = metadata['records'][r['image']]
    for cell, mask, actual in zip(r['source_cells'], r['masks'], m['source_points']):
        assert actual == raw_points[(r['source'], cell['row'], cell['col'], cell['annotation_line'], mask)]
        point_occurrences += 1
assert point_occurrences == 27055

# Verify every unaugmented DSBI target, including all complete-center entries.
target_counts = defaultdict(int)
gradient_record = None
for r in dsbi_records:
    boxes = np.asarray(r['boxes'], np.float32).reshape(-1, 4)
    masks = np.asarray(r['masks'], int)
    eligible = np.asarray(r['originally_complete'], bool)
    eligible &= (boxes[:, 0]>=0)&(boxes[:, 1]>=0)&(boxes[:, 2]<=480)&(boxes[:, 3]<=192)
    ignored = np.asarray(r['ignore_boxes'], np.float32).reshape(-1, 4)
    old_target = old.make_target(boxes, masks, eligible, False, ignore_boxes=ignored)
    off = current.make_target(boxes, masks, eligible, False, ignore_boxes=ignored)
    assert np.array_equal(old_target, off)
    indices, fragments = visible_partial_candidates(boxes, states[r['image']])
    on = current.make_target(boxes, masks, eligible, False, ignore_boxes=ignored, visible_partial_boxes=fragments)
    full = off[..., 0] == 1
    assert np.array_equal(off[full], on[full])
    added = (on[..., 0] == 1)&~full
    assert np.all(on[added, 5:] == -1)
    known_fragment_slots = {(int((b[1]+b[3])/8), int((b[0]+b[2])/8)) for b in fragments}
    assert all(tuple(slot) in known_fragment_slots for slot in np.argwhere(added))
    if not len(boxes):
        assert np.array_equal(off, on)
        target_counts['zero_known_records_unchanged'] += 1
    target_counts['records'] += 1
    target_counts['full_centers_unchanged'] += int(full.sum())
    target_counts['visible_fragment_inputs'] += len(indices)
    target_counts['new_unknown_dot_centers'] += int(added.sum())
    if gradient_record is None and added.any() and full.any():
        np.save(OUT/'gradient-target.npy', on)
        gradient_record = {'image': r['image'], 'image_sha256': r['image_sha256'], 'target_sha256': sha(OUT/'gradient-target.npy')}

class Base:
    def __init__(self, **kwargs): pass

class NumpyProxy:
    def __init__(self):
        self.generators = []
        self.random = SimpleNamespace(default_rng=self.default_rng)
    def default_rng(self, seed=None):
        rng = np.random.default_rng(seed); self.generators.append(rng); return rng
    def __getattr__(self, name): return getattr(np, name)
    def snapshot(self):
        values = [g.bit_generator.state for g in self.generators]
        self.generators.clear()
        return values

def patches(path, module):
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'Patches')
    proxy = NumpyProxy()
    namespace = dict(np=proxy, math=math, Image=Image, keras=SimpleNamespace(utils=SimpleNamespace(PyDataset=Base)),
                     BalancedFamilyOrder=BalancedFamilyOrder, visible_partial_candidates=visible_partial_candidates,
                     HEIGHT=192, WIDTH=480)
    namespace.update({name: getattr(module, name) for name in ('augment_patch', 'make_target', 'rotate_patch_180', 'mirror_patch_horizontal')})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Patches'], proxy

OldPatches, old_rng = patches(WORK/'run19/source/word_train.py', old)
NewPatches, new_rng = patches(ROOT/'training/scripts/word_train.py', current)
groups = defaultdict(list)
for r in all_records:
    family = r.get('source_family', r['source'].rsplit('/', 1)[0])
    groups[(family, bool(r['boxes']))].append(r)
sample = [r for key in sorted(groups) for r in groups[key][:2]]
profiles = [{}, {'camera_augmentation': True}, {'illumination_augmentation': True},
            {'polarity_augmentation': True}, {'half_turn_augmentation': True},
            {'illumination_augmentation': True, 'horizontal_mirror_augmentation': True}]
trace = defaultdict(int)

class TrackedState:
    def __init__(self, state, lost=None):
        self.state = state
        self.lost = np.zeros_like(state.survivors) if lost is None else lost
    def __getattr__(self, name): return getattr(self.state, name)
    def operation(self, name, *args):
        after = getattr(self.state, name)(*args)
        previous = self.state.survivors
        lost = self.lost.copy()
        if name == 'mirror_horizontal':
            previous, lost = previous[:, [3,4,5,0,1,2]], lost[:, [3,4,5,0,1,2]]
        if name == 'half_turn':
            previous, lost = previous[:, ::-1], lost[:, ::-1]
        assert not np.any(after.survivors & ~previous)
        lost |= previous & ~after.survivors
        trace[name] += 1
        trace['occupied_centers_lost'] += int((previous & ~after.survivors).sum())
        trace['previously_lost_coordinates_inside_still_invisible'] += int((lost & safe_center_mask(after.points, after.width, after.height)).sum())
        return TrackedState(after, lost)
    def affine(self, *args): return self.operation('affine', *args)
    def rotate(self, *args): return self.operation('rotate', *args)
    def mirror_horizontal(self): return self.operation('mirror_horizontal')
    def half_turn(self): return self.operation('half_turn')

tracked = {image: TrackedState(state) for image, state in states.items()}
replay = []
for p, profile in enumerate(profiles):
    for training in (True, False):
        kwargs = dict(root=DATA, records=sample, batch_size=8, training=training, seed=20260905+p,
                      balance_source_families=training, **profile)
        a, b, c = OldPatches(**kwargs), NewPatches(**kwargs), NewPatches(**kwargs, point_states=tracked)
        old_rng.snapshot(); new_rng.snapshot()
        assert np.array_equal(a.order, b.order) and np.array_equal(a.order, c.order)
        for batch in range(len(a)):
            x_old, y_old = a[batch]; rng_old = old_rng.snapshot()
            x_off, y_off = b[batch]; rng_off = new_rng.snapshot()
            x_on, y_on = c[batch]; rng_on = new_rng.snapshot()
            assert rng_old == rng_off == rng_on
            assert np.array_equal(x_old, x_off) and np.array_equal(x_old, x_on)
            assert np.array_equal(y_old, y_off)
            full = y_off[..., 0] == 1
            assert np.array_equal(y_off[full], y_on[full])
            added = (y_on[..., 0] == 1)&~full
            assert np.all(y_on[added, 5:] == -1)
        replay.append(dict(profile=profile, training=training, batches=len(a), records=len(sample), passed=True))

# A diagnostic Angelina crop gets no invented point state. Its direct target
# remains five complete cells: transfer from DSBI is a hypothesis, not a fix.
fixture = next(r for r in json.loads((ROOT/'training/evidence/braille-word-reader-v1/rejection-fixtures.json').read_text())['records'] if r['id'] == 'selection_cut-02')
assert fixture['image'] not in states
target = current.make_target(np.asarray(fixture['boxes'], np.float32), np.asarray(fixture['masks']), np.asarray(fixture['context_complete_in_crop']), False)
thresholds = json.loads((WORK/'selected-v13/rejection-diagnostics.json').read_text())['thresholds']
result = decode(np.maximum(target, 0), thresholds, fixture['selection'])
assert result['text'] == 'BLYCS' and not result['reason']

class FixedGeometryRng:
    # Legal existing augmentation values: translation +3 px loses the right
    # edge point, then -4 degrees returns its coordinates inside the raster.
    def __init__(self):
        self.random_values = iter([0., 0., 1., 1.])
        self.uniform_values = iter([1., 3/480, 0., -4., 1., .5])
    def random(self): return next(self.random_values)
    def uniform(self, low, high):
        value = next(self.uniform_values)
        assert low <= value <= high
        return value

marker = np.zeros((192, 480, 3), np.uint8); marker[150, 479] = [255, 0, 0]
cut_state = OccupiedPointState.from_source_crop([[[479, 150]]*6], [1], [0,0,480,192])
forced = current.augment_patch(Image.fromarray(marker), np.asarray([[465.,135.,496.,172.]], np.float32),
                               FixedGeometryRng(), illumination_augmentation=True,
                               originally_complete=np.asarray([False]), point_state=cut_state)
assert safe_center_mask(forced[3].points, 480, 192).all()
assert not forced[3].survivors.any() and not forced[0].any()
assert not len(visible_partial_candidates(forced[1], forced[3])[0])
forced_case = dict(final_coordinates=forced[3].points[0,0].tolist(), final_coordinates_inside=True,
                   occupied_survivors=0, generated_fragment_targets=0, actual_marker_pixels_recovered=False)
for name, digest in bindings.items(): assert sha(ROOT/'training/scripts'/name) == digest
report = dict(status='passed', scope='Independent CPU NumPy/Pillow replay of actual AST-loaded Patches against frozen run19, plus all2338 raw-TXT point records and target deltas. No Keras/model/inference/training/GPU.',
              script_sha256=sha(__file__), source_sha256=bindings,
              baseline_source_sha256={name:sha(WORK/'run19/source'/name) for name in ('word_data.py','word_train.py')},
              metadata_sha256=sha(metadata_path), metadata_known_occurrences_checked=point_occurrences,
              unaugmented_target_counts=dict(target_counts), paired_profiles=replay,
              paired_sample_sources=sorted({r['source'] for r in sample}), paired_sample_images=[r['image'] for r in sample],
              geometry_trace=dict(trace), gradient_fixture=gradient_record,
              forced_affine_outside_then_rotation_inside=forced_case,
              physical_F_strategy=dict(unchanged_angelina_ideal_target_text=result['text'], reason=result['reason'],
                                       scope='No source point metadata exists for Angelina, so this ablation does not directly change that target. Recognition transfer from DSBI cuts must be measured after training.'),
              limitations=['PyDataset base is a minimal constructor stub; the actual Patches class, data functions, sampler and RNG are used without Keras scheduling.',
                           'Paired augmentation replay is bounded to recorded profiles/sources/seeds; all2338 unaugmented DSBI target cases are checked.',
                           'Gradient evidence is generated separately from the saved actual target using the unchanged loss.'])
(OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:report[k] for k in ('status','unaugmented_target_counts','geometry_trace','physical_F_strategy')},indent=2))
