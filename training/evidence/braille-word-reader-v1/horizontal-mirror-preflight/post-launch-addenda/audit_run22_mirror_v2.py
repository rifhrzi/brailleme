"""Independent CPU mirror/Patches audit; no backend, training or fixture edits."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path.cwd()
WORK = ROOT/'.braille-training-work/word-scan-v1-20260905'
SOURCE = WORK/'run22/source'
sys.path.insert(0, str(SOURCE))
from word_point_geometry import OccupiedPointState, visible_partial_candidates
from word_sampling import BalancedFamilyOrder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nodes(path, names):
    module = ast.parse(Path(path).read_text(encoding='utf-8'))
    selected = [node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert len(selected) == len(names)
    return ast.Module(body=selected, type_ignores=[])


DATA_NAMES = ['make_target', 'affine_patch', 'mirror_patch_horizontal', 'rotate_patch_180', 'augment_patch']


def data_namespace(path):
    namespace = dict(np=np, Image=Image, ImageEnhance=ImageEnhance, ImageFilter=ImageFilter,
                     WIDTH=480, HEIGHT=192, STRIDE=4, GRID_W=120, GRID_H=48)
    exec(compile(nodes(path, DATA_NAMES), str(path), 'exec'), namespace)
    return namespace


current = data_namespace(SOURCE/'word_data.py')
old = data_namespace(WORK/'run20/source/word_data.py')
for name in ['word_data.py', 'word_point_geometry.py', 'word_point_metadata.py', 'word_sampling.py']:
    assert sha(SOURCE/name) == sha(WORK/'run20/source'/name), name
assert ast.dump(nodes(SOURCE/'word_train.py', ['Patches'])) == ast.dump(nodes(WORK/'run20/source/word_train.py', ['Patches']))


def permute_mask(mask):
    # Independent position loop, not the implementation's bitwise halves.
    return sum(1 << ((dot+3) % 6) for dot in range(6) if int(mask) & (1 << dot))


def reflect_points(points):
    result = np.asarray(points, np.float64).copy()
    result[..., 0] = 479-result[..., 0]
    return result[:, [3, 4, 5, 0, 1, 2]]


def hull(points, width=480, height=192):
    return (points[..., 0] >= 0) & (points[..., 0] <= width-1) & (points[..., 1] >= 0) & (points[..., 1] <= height-1)


# All 63 known masks, actual RGB reflection, complete target labels and a
# deliberately non-midpoint middle row. Target cell centers do not collide.
points = np.asarray([[[100, 41], [100, 58], [100, 84], [120, 41], [120, 58], [120, 84]]], np.float64)
pixels = np.random.default_rng(901).integers(0, 256, (192, 480, 3), dtype=np.uint8)
box = np.asarray([[94, 35, 126, 90]], np.float32)
for mask in range(1, 64):
    image, mirrored_box, mirrored_masks = current['mirror_patch_horizontal'](Image.fromarray(pixels), box, np.asarray([mask]))
    np.testing.assert_array_equal(image, pixels[:, ::-1])
    np.testing.assert_array_equal(mirrored_box, [[354, 35, 386, 90]])
    assert mirrored_masks.tolist() == [permute_mask(mask)]
    state = OccupiedPointState.from_source_crop(points, [mask], [0, 0, 480, 192])
    mirrored = state.mirror_horizontal()
    np.testing.assert_array_equal(mirrored.points, reflect_points(points))
    assert mirrored.survivors[0].tolist() == [bool(permute_mask(mask) & (1 << dot)) for dot in range(6)]
    np.testing.assert_array_equal(mirrored.mirror_horizontal().points, state.points)
    np.testing.assert_array_equal(mirrored.mirror_horizontal().survivors, state.survivors)
    target = current['make_target'](mirrored_box, mirrored_masks, np.asarray([True]))
    center = target[15, 92]
    assert center[0] == 1
    assert center[5:].tolist() == [int(bool(permute_mask(mask) & (1 << dot))) for dot in range(6)]
    restored = current['mirror_patch_horizontal'](image, mirrored_box, mirrored_masks)
    np.testing.assert_array_equal(restored[0], pixels)
    np.testing.assert_array_equal(restored[1], box)
    np.testing.assert_array_equal(restored[2], [mask])

# Crop+resize hull then mirror, scale/translate, rotation and a return
# transform: compare every slot's coordinates and irreversible visibility.
# Cases include lost occupied points, empty occupied slots and exact borders.
stage_cases = 0
for mask in range(1, 64):
    raw = np.asarray([[[-1, 12], [0, 0], [5, 8], [29, 18], [30, 19], [15, 20]]], np.float64)
    crop = [0, 0, 30, 20]
    occupied = np.asarray([[(mask >> bit) & 1 for bit in range(6)]], bool)
    q = (raw+.5)*[16, 9.6]-.5
    visible = occupied & hull(raw, 30, 20) & hull(q)
    state = OccupiedPointState.from_source_crop(raw, [mask], crop)
    np.testing.assert_array_equal(state.survivors, visible)
    np.testing.assert_allclose(state.points, q, rtol=0, atol=1e-12)
    q = reflect_points(q)
    visible = visible[:, [3, 4, 5, 0, 1, 2]] & hull(q)
    state = state.mirror_horizontal()
    np.testing.assert_array_equal(state.survivors, visible)
    for operation in [('affine', (1.13, .91, 18.3, -7.1)), ('rotate', (3.25,)), ('affine', (1, 1, -45, 25)), ('mirror', ())]:
        if operation[0] == 'affine':
            sx, sy, tx, ty = operation[1]
            q = (q+.5)*[sx, sy]+[240*(1-sx)+tx, 96*(1-sy)+ty]-.5
            state = state.affine(sx, sy, tx, ty)
        elif operation[0] == 'rotate':
            angle = math.radians(operation[1][0])
            x, y = q[..., 0]-239.5, q[..., 1]-95.5
            q = np.stack([math.cos(angle)*x+math.sin(angle)*y+239.5,
                          -math.sin(angle)*x+math.cos(angle)*y+95.5], axis=-1)
            state = state.rotate(operation[1][0])
        else:
            q = reflect_points(q)
            visible = visible[:, [3, 4, 5, 0, 1, 2]]
            state = state.mirror_horizontal()
        visible &= hull(q)
        np.testing.assert_allclose(state.points, q, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(state.survivors, visible)
        stage_cases += 1

# Known partial geometry always gets six unknown bits after mirror. A lost
# occupied point cannot supply a fragment even if its annotation returns.
partial_checks = 0
for mask in range(1, 64):
    state = OccupiedPointState.from_source_crop(np.asarray([[[2, 45], [2, 53], [2, 65], [6, 45], [6, 53], [6, 65]]]), [mask], [0, 0, 480, 192])
    partial = np.asarray([[-4, 40, 10, 70]], np.float32)
    _, reflected, labels = current['mirror_patch_horizontal'](Image.fromarray(pixels), partial, np.asarray([mask]))
    indices, fragments = visible_partial_candidates(reflected, state.mirror_horizontal())
    assert indices.tolist() == [0]
    target = current['make_target'](reflected, labels, np.asarray([False]), visible_partial_boxes=fragments)
    peaks = target[..., 0] == 1
    assert peaks.sum() == 1
    assert np.all(target[..., 5:][peaks] == -1)
    lost = state.affine(1, 1, -50, 0).affine(1, 1, 50, 0).mirror_horizontal()
    assert visible_partial_candidates(reflected, lost)[0].size == 0
    partial_checks += 1

# Actual Pillow RGB oracle includes source crop/resize, then horizontal mirror,
# affine and rotate with distinct marker channels, preserving vertical relief.
centers = np.asarray([[101., 77.], [327., 131.], [552., 207.]])
yy, xx = np.indices((261, 655))
rgb = np.stack([np.rint(255*np.exp(-((xx-x)**2+(yy-y)**2)/32)) for x, y in centers], axis=-1).astype(np.uint8)
image = Image.fromarray(rgb).crop([20, 10, 620, 250]).resize((480, 192), Image.Resampling.BILINEAR)
state = OccupiedPointState.from_source_crop(np.repeat(centers[:, None], 6, axis=1), [1]*3, [20, 10, 620, 250])
image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
state = state.mirror_horizontal()
marker_errors = []
for stage in ['mirror', 'affine', 'rotation']:
    if stage == 'affine':
        image, _ = current['affine_patch'](image, np.empty((0, 4)), .91, 1.08, 7.2, -4.3)
        state = state.affine(.91, 1.08, 7.2, -4.3)
    elif stage == 'rotation':
        image = image.rotate(3.3, Image.Resampling.BILINEAR, fillcolor=tuple(np.asarray(image).mean(axis=(0, 1)).astype(int)))
        state = state.rotate(3.3)
    array = np.asarray(image, np.float64)
    y, x = np.indices(array.shape[:2])
    observed = np.asarray([[(array[..., c]*x).sum()/array[..., c].sum(), (array[..., c]*y).sum()/array[..., c].sum()] for c in range(3)])
    error = float(np.max(np.abs(observed-state.points[:, 3])))
    assert error < .1, (stage, error)
    marker_errors.append({'stage': stage, 'max_centroid_error_pixels': error})

# The unchanged Patches class executes under a neutral PyDataset base stub,
# without importing Keras/TF/Torch. Actual PNGs, order, functions and NumPy RNG
# execute normally. Record batches are selected from the full balanced order.
class Base:
    def __init__(self, **kwargs):
        pass


class RandomTrace:
    def __init__(self, seed):
        self.inner = np.random.default_rng(seed)
        self.calls = []

    def __getattr__(self, name):
        method = getattr(self.inner, name)
        def call(*args, **kwargs):
            result = method(*args, **kwargs)
            self.calls.append((name, args, kwargs, np.asarray(result).tolist()))
            return result
        return call


class NumpyProxy:
    def __init__(self, traces):
        def rng(seed):
            item = RandomTrace(seed)
            traces.append(item)
            return item
        self.random = SimpleNamespace(default_rng=rng)

    def __getattr__(self, name):
        return getattr(np, name)


mirror_invocations = []
target_audits = []


def factory(path, data):
    traces = []
    namespace = dict(data, np=NumpyProxy(traces), math=math, keras=SimpleNamespace(utils=SimpleNamespace(PyDataset=Base)),
                     BalancedFamilyOrder=BalancedFamilyOrder, visible_partial_candidates=visible_partial_candidates)
    original_mirror = namespace['mirror_patch_horizontal']
    def checked_mirror(image, boxes, masks):
        result = original_mirror(image, boxes, masks)
        np.testing.assert_array_equal(np.asarray(result[0]), np.asarray(image)[:, ::-1])
        expected = boxes.copy()
        expected[:, 0], expected[:, 2] = 480-boxes[:, 2], 480-boxes[:, 0]
        np.testing.assert_array_equal(result[1], expected)
        np.testing.assert_array_equal(result[2], [permute_mask(mask) for mask in masks])
        mirror_invocations.append(len(masks))
        return result
    namespace['mirror_patch_horizontal'] = checked_mirror
    original_target = namespace['make_target']
    def checked_target(boxes, masks, eligible, *args, **kwargs):
        target = original_target(boxes, masks, eligible, *args, **kwargs)
        final_known = {}
        for box, mask, complete in zip(boxes, masks, eligible):
            if complete and box[0] >= 0 and box[1] >= 0 and box[2] <= 480 and box[3] <= 192:
                gx, gy = ((box[:2]+box[2:])/8).astype(int)
                final_known[(gy, gx)] = int(mask)
        for (gy, gx), mask in final_known.items():
            assert target[gy, gx, 0] == 1
            assert target[gy, gx, 5:].tolist() == [int(bool(mask & (1 << bit))) for bit in range(6)]
        peaks = np.argwhere(target[..., 0] == 1)
        if not len(boxes):
            assert not len(peaks), 'unknown-only or explicit negative acquired objectness'
        for gy, gx in peaks:
            if (gy, gx) not in final_known:
                assert np.all(target[gy, gx, 5:] == -1), 'partial/unknown supplied a label'
        target_audits.append({'complete': len(final_known), 'geometry_only': len(peaks)-len(final_known), 'unknown_boxes': len(kwargs.get('ignore_boxes', [])), 'negative_or_unknown_only': int(not len(boxes))})
        return target
    namespace['make_target'] = checked_target
    exec(compile(nodes(path, ['Patches']), str(path), 'exec'), namespace)
    return namespace['Patches'], traces


config = json.loads((WORK/'run20/training-config.json').read_text())
data_root = WORK/'data'
records = json.loads((data_root/config['train_manifest']).read_text())
index = json.loads((data_root/config['supplemental_source_indexes'][0]).read_text())
extra_sources = {row['source'] for row in index['sources']}
base_neg = [row for row in records if not row['boxes'] and row['source'] not in extra_sources]
extra_neg = [row for row in records if not row['boxes'] and row['source'] in extra_sources]
assert (len(base_neg), len(extra_neg)) == (44, 47)
expanded = records+base_neg*26+extra_neg
assert len(expanded) == 24882
metadata = json.loads((WORK/'dsbi-point-metadata-v1/points.json').read_text())
states = {image: OccupiedPointState.from_source_crop(np.asarray(row['source_points']).reshape(-1, 6, 2), row['masks'], row['crop'])
          for image, row in metadata['records'].items()}
before_metadata = sha(WORK/'dsbi-point-metadata-v1/points.json')
point_digest = lambda: hashlib.sha256(b''.join(state.points.tobytes()+state.survivors.tobytes() for state in states.values())).hexdigest()
before_states = point_digest()
Old, old_trace = factory(WORK/'run20/source/word_train.py', old)
Current, current_trace = factory(SOURCE/'word_train.py', current)
On, on_trace = factory(SOURCE/'word_train.py', current)
options = dict(batch_size=6, training=True, seed=config['seed'], illumination_augmentation=True,
               balance_source_families=True, point_states=states)
off_old, off_current = Old(data_root, expanded, **options), Current(data_root, expanded, **options)
on = On(data_root, expanded, **options, horizontal_mirror_augmentation=True)
for dataset in [off_old, off_current, on]:
    dataset.on_epoch_end()  # Actual Keras first training phase observed in run20.
np.testing.assert_array_equal(off_old.order, off_current.order)
np.testing.assert_array_equal(off_old.order, on.order)
families = set()
chosen = []
for batch in range(len(on)):
    group = on.order[batch*6:(batch+1)*6]
    new = {str(Path(expanded[i]['source']).parent).replace('\\', '/') for i in group if expanded[i]['boxes']}
    if not new.issubset(families):
        chosen.append(batch)
        families |= new
    if len(families) == 12:
        break
assert len(families) == 12
# Include a supplemental negative and a partial+unknown DSBI context.
for predicate in [lambda row: not row['boxes'] and row['source'] in extra_sources,
                  lambda row: row['source'] in extra_sources and row.get('ignore_boxes') and not all(row.get('originally_complete', []))]:
    position = next(i for i, record_index in enumerate(on.order) if predicate(expanded[record_index]))
    chosen.append(position//6)
chosen = sorted(set(chosen))
batch_rows = []
for batch in chosen:
    a, b, c = off_old[batch], off_current[batch], on[batch]
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    assert old_trace[-1].calls == current_trace[-1].calls == on_trace[-1].calls
    for trace in [current_trace[-1], on_trace[-1]]:
        assert trace.inner.bit_generator.state == old_trace[-1].inner.bit_generator.state
    batch_rows.append({'batch': batch, 'record_indices': on.order[batch*6:(batch+1)*6].tolist(),
                       'off_pixels_sha256': hashlib.sha256(b[0].tobytes()).hexdigest(),
                       'off_targets_sha256': hashlib.sha256(b[1].tobytes()).hexdigest(),
                       'on_pixels_sha256': hashlib.sha256(c[0].tobytes()).hexdigest(),
                       'on_targets_sha256': hashlib.sha256(c[1].tobytes()).hexdigest(),
                       'paired_random_calls': len(on_trace[-1].calls)})
assert before_states == point_digest()
assert before_metadata == sha(WORK/'dsbi-point-metadata-v1/points.json')
assert any(row['geometry_only'] for row in target_audits)
assert any(row['unknown_boxes'] for row in target_audits)
assert mirror_invocations
for forbidden in ['tensorflow', 'torch', 'keras']:
    assert forbidden not in sys.modules

plan_path = WORK/'run22/run22-plan.json'
plan = json.loads(plan_path.read_text())
for name, digest in plan['source_sha256'].items():
    assert sha(SOURCE/name) == digest
for key, expected in {'illumination_augmentation': True, 'horizontal_mirror_augmentation': True,
                      'half_turn_augmentation': False, 'partial_geometry_supervision': False,
                      'dsbi_visible_partial_supervision': True, 'geometry_conditioned_residual': False,
                      'planned_epochs': 10, 'negative_repeat': 27, 'supplemental_negative_repeat': 2}.items():
    assert plan[key] == expected
assert plan['mirror']['probability'] == .5
assert plan['mirror']['slot_permutation_zero_based'] == [3, 4, 5, 0, 1, 2]
assert plan['training_manifest_sha256'] == sha(data_root/config['train_manifest'])
assert plan['point_metadata_sha256'] == before_metadata
assert plan['parent_training_config_sha256'] == sha(WORK/'run20/training-config.json')
assert plan['parent_sampling_lineage_sha256'] == sha(WORK/'run20/sampling-lineage-final.json')
assert plan['parent_selected_validation_sha256'] == sha(WORK/'selected-v15/selected-validation.json')

report = {'status': 'passed', 'scope': 'Read-only CPU NumPy/Pillow proof. Exact unchanged Patches AST is executed with a neutral PyDataset base; no Keras worker scheduling/backend/model is executed. This is transform/target integrity, not accuracy or physical-camera evidence.',
          'post_launch_addendum': True, 'prelaunch_report_sha256': 'efd565a8d372c141a5f5e3456a855f4e59d0486da75a9f09ab807e19d52a62c6',
          'source_sha256': {name: sha(SOURCE/name) for name in ['word_data.py', 'word_train.py', 'word_point_geometry.py', 'word_point_metadata.py', 'word_sampling.py']},
          'script_sha256': sha(__file__), 'run20_config_sha256': sha(WORK/'run20/training-config.json'),
          'run22_draft_plan_sha256': sha(plan_path), 'draft_flags_and_all_source_hashes_match': True,
          'training_manifest_sha256': sha(data_root/config['train_manifest']), 'point_metadata_sha256': before_metadata,
          'run20_data_point_metadata_sampling_sources_byte_identical': True, 'run20_patches_ast_identical': True,
          'all_known_mask_mirror_involutions_and_target_labels': 63, 'stage_visibility_cases': stage_cases,
          'partial_unknown_bit_checks': partial_checks, 'mirror_pixel_operation_exact': True,
          'point_slot_permutation': [3, 4, 5, 0, 1, 2], 'pillow_marker_oracles': marker_errors,
          'full_balanced_order_identical': True, 'expanded_negative_exposure': {'base': [44, 27], 'supplemental': [47, 2]},
          'real_batches': batch_rows, 'real_source_families': sorted(families),
          'mirror_invocations': len(mirror_invocations), 'target_checks': {key: sum(row[key] for row in target_audits) for key in ['complete', 'geometry_only', 'unknown_boxes', 'negative_or_unknown_only']},
          'off_mode_rgb_targets_rng_exact_vs_run20': True, 'on_off_random_calls_and_final_rng_states_equal': True,
          'metadata_and_cached_point_states_unchanged': True, 'defects': [],
          'limitations': ['Collision priority in the existing one-cell-per-grid head is unchanged; complete known labels win over geometry-only fragments.',
                         'Unknown slot regions can overlap a known-cell Gaussian; unknown slots themselves never supply a positive center, geometry or bits.',
                         'Mirrored Braille patterns can become non-A-Z; all 63 masks remain valid TRAIN supervision and runtime still rejects unsupported patterns.',
                         'Actual mirrored augmentation does not guarantee recognition gain; original validation/public/native/physical criteria remain unchanged.']}
(WORK/'run22-mirror-independent-audit-v2.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'status': report['status'], 'real_batches': len(chosen), 'real_examples_per_mode': len(chosen)*6,
                  'families': len(families), 'mirror_invocations': len(mirror_invocations), 'target_checks': report['target_checks'],
                  'marker_max_error': max(row['max_centroid_error_pixels'] for row in marker_errors)}))
