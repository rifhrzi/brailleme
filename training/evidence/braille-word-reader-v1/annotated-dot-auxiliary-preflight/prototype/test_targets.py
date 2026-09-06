"""Synthetic array tests only; no dataset pixels or ML backend imports."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'candidate'))
from word_dot_aux_targets import complete_dot_queries, pad_dot_queries


def state(masks):
    masks = np.asarray(masks, np.int64)
    points = np.tile(np.asarray([[101., 61.], [101., 81.], [101., 101.],
                                [121., 61.], [121., 81.], [121., 101.]]), (len(masks), 1, 1))
    survivors = ((masks[:, None] >> np.arange(6)) & 1).astype(bool)
    return SimpleNamespace(points=points, survivors=survivors)


def reference_bce(labels, logits):
    known = labels >= 0
    loss = np.maximum(logits, 0) - logits * np.maximum(labels, 0) + np.log1p(np.exp(-np.abs(logits)))
    return np.sum(loss * known) / max(np.sum(known), 1)


class DotTargetTests(unittest.TestCase):
    def test_all_63_masks_use_explicit_absent_labels(self):
        masks = np.arange(1, 64)
        queries, labels = complete_dot_queries(state(masks), masks, np.ones(63, bool))
        np.testing.assert_array_equal(labels.reshape(63, 6), (masks[:, None] >> np.arange(6)) & 1)
        self.assertEqual(queries.shape, (378, 2))
        self.assertEqual(labels[:6, 0].tolist(), [1, 0, 0, 0, 0, 0])

    def test_unknown_angelina_partial_and_lost_occupied_are_ignored(self):
        self.assertEqual(len(complete_dot_queries(None, [1], [True])[0]), 0)
        masks = np.asarray([-1, 0, 1, 3])
        dots = state(masks)
        dots.survivors[3, 0] = False
        self.assertEqual(len(complete_dot_queries(dots, masks, np.asarray([True, True, False, True]))[0]), 0)

    def test_all_six_points_must_be_inside_feature_hull(self):
        masks = np.asarray([1, 1, 1, 1, 1])
        dots = state(masks)
        dots.points[0, 1] = [3, 3]
        dots.points[0, 5] = [479, 191]
        for i, position in enumerate(([2.999, 50], [479.001, 50], [50, 2.999], [50, 191.001]), 1):
            dots.points[i, 5] = position  # This unoccupied slot still excludes the cell.
        queries, labels = complete_dot_queries(dots, masks, np.ones(5, bool))
        self.assertEqual(len(queries), 6)
        np.testing.assert_array_equal(queries[1], [0, 0])
        np.testing.assert_array_equal(queries[5], [119, 47])

    def test_fractional_query_origin_is_not_rounded(self):
        dots = state([63])
        dots.points[0, 0] = [104.5, 86.25]
        queries, _ = complete_dot_queries(dots, np.asarray([63]), np.ones(1, bool))
        np.testing.assert_array_equal(queries[0], [(104.5 - 3) / 4, (86.25 - 3) / 4])
        x, y = queries[0]
        gx, gy = np.floor([x, y]).astype(int)
        wx, wy = x-gx, y-gy
        feature = lambda xx, yy: 2*xx+3*yy+5
        interpolated = ((1-wy)*((1-wx)*feature(gx, gy)+wx*feature(gx+1, gy))
                        + wy*((1-wx)*feature(gx, gy+1)+wx*feature(gx+1, gy+1)))
        self.assertEqual(interpolated, feature(x, y))

    def test_inputs_and_random_stream_are_unchanged(self):
        masks = np.asarray([1, 3, 63]); complete = np.ones(3, bool); dots = state(masks)
        before = deepcopy((dots, masks, complete)); rng = np.random.default_rng(42); rng_before = deepcopy(rng.bit_generator.state)
        complete_dot_queries(dots, masks, complete)
        np.testing.assert_array_equal(dots.points, before[0].points)
        np.testing.assert_array_equal(dots.survivors, before[0].survivors)
        np.testing.assert_array_equal(masks, before[1]); np.testing.assert_array_equal(complete, before[2])
        self.assertEqual(rng.bit_generator.state, rng_before)

    def test_record_cell_slot_order_and_original_partial(self):
        masks = np.asarray([3, 1, 5])
        dots = state(masks)
        dots.points[0] += [40, 0]
        dots.points[1] += [80, 0]
        dots.points[2] += [120, 0]
        # The last cell's points are now all inside, but its original crop was partial.
        queries, labels = complete_dot_queries(dots, masks, np.asarray([True, True, False]))
        self.assertEqual(labels[:, 0].tolist(), [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(queries.reshape(2, 6, 2), (dots.points[:2]-3)/4)

    def test_dynamic_batch_padding_never_truncates(self):
        masks = np.ones(420, np.int64)
        large = complete_dot_queries(state(masks), masks, np.ones(420, bool))
        empty = complete_dot_queries(None, [], [])
        queries, labels = pad_dot_queries([empty, large])
        self.assertEqual(queries.shape, (2, 2520, 2))
        self.assertEqual(np.sum(labels >= 0), 2520)
        self.assertTrue(np.all(labels[0] == -1))

    def test_empty_auxiliary_batch_and_separate_normalizer(self):
        empty = complete_dot_queries(None, [], [])
        queries, labels = pad_dot_queries([empty, empty])
        self.assertEqual(queries.shape, (2, 1, 2))
        self.assertEqual(reference_bce(labels, np.ones_like(labels)*100), 0)
        known = np.asarray([[[1.], [0.]]]); logits = np.asarray([[[.3], [-.7]]])
        padded_labels = np.concatenate((known, np.full((1, 50, 1), -1)), axis=1)
        padded_logits = np.concatenate((logits, np.full((1, 50, 1), 100)), axis=1)
        self.assertEqual(reference_bce(known, logits), reference_bce(padded_labels, padded_logits))

    def test_nonfinite_and_misaligned_inputs_fail(self):
        dots = state([1]); dots.points[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): complete_dot_queries(dots, np.asarray([1]), np.ones(1, bool))
        with self.assertRaises(ValueError): complete_dot_queries(state([1]), np.asarray([1]), np.ones(2, bool))

    def test_callback_and_training_wrapper_source_keep_reader_boundary(self):
        root = Path(__file__).parent
        tree = ast.parse((root/'candidate/word_train.py').read_text())
        callback = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WordCheckpoint')
        method = next(n for n in callback.body if isinstance(n, ast.FunctionDef) and n.name == 'on_epoch_end')
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        predict = [n for n in calls if n.func.attr == 'predict']
        self.assertEqual(len(predict), 1)
        self.assertEqual(ast.unparse(predict[0].func.value), 'reader')
        saved = {ast.unparse(n.args[0]): ast.unparse(n.func.value) for n in calls if n.func.attr == 'save'}
        self.assertEqual(saved, {"self.output / 'last.keras'": 'reader',
                                 "self.output / 'last-training.keras'": 'self.model',
                                 "self.output / 'best-word.keras'": 'reader'})
        wrapper = (root/'candidate/word_dot_aux.py').read_text()
        self.assertIn('AUXILIARY_DOT_COEFFICIENT = .25', wrapper)
        self.assertIn('(reader.output, samples)', wrapper)
        self.assertNotIn('build_model', wrapper)
        self.assertNotIn('set_weights', wrapper)
        self.assertIn('loss=(main_loss, known_dot_bce)', wrapper)

    def test_real_getitem_source_preserves_main_arrays_and_rng_off_and_on(self):
        """Execute actual old/new method AST; synthetic image/augmentation adapters."""
        root = Path(__file__).parent
        make_target = next(n for n in ast.parse((root/'original/word_data.py').read_text()).body
                           if isinstance(n, ast.FunctionDef) and n.name == 'make_target')

        class FakeImage:
            def __init__(self, value): self.value = value
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def convert(self, mode): return self
            def __array__(self, dtype=None, copy=None):
                return np.full((192, 480, 3), self.value, dtype=dtype or np.uint8)

        records = [{'image': str(i), 'boxes': [[90, 50, 135, 115]], 'masks': [mask],
                    'originally_complete': [i != 2]} for i, mask in enumerate((1, 3, 5))]
        observed = []
        for filename, enabled in ((root/'original/word_train.py', False),
                                  (root/'candidate/word_train.py', False),
                                  (root/'candidate/word_train.py', True)):
            cls = next(n for n in ast.parse(filename.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Patches')
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
            traces = []

            def augment(image, boxes, rng, *args, **kwargs):
                draws = rng.random(3); traces.append(draws.tolist())
                tx = draws[0] - .5
                shifted = boxes + np.asarray([tx, 0, tx, 0])
                current = kwargs['point_state']
                if current is not None:
                    current = SimpleNamespace(points=current.points + [tx, 0], survivors=current.survivors)
                values = (np.asarray(image, np.uint8), shifted, kwargs['originally_complete'].copy())
                return values if current is None else values + (current,)

            namespace = {'np': np, 'WIDTH': 480, 'HEIGHT': 192, 'STRIDE': 4, 'GRID_W': 120, 'GRID_H': 48,
                         'Image': SimpleNamespace(open=lambda path: FakeImage(100+int(path.name))),
                         'augment_patch': augment,
                         'visible_partial_candidates': lambda boxes, points: ([], np.empty((0, 4))),
                         'complete_dot_queries': complete_dot_queries, 'pad_dot_queries': pad_dot_queries}
            exec(compile(ast.Module(body=[make_target, method], type_ignores=[]), str(filename), 'exec'), namespace)
            instance = SimpleNamespace(seed=20260905, epoch=2, order=np.asarray([2, 0, 1]), batch_size=3,
                                       training=True, root=Path('.'), records=records,
                                       point_states={'0': state([1]), '2': state([5])},
                                       illumination_augmentation=True, partial_geometry_supervision=False,
                                       auxiliary_dot_supervision=enabled, vertical_stretch_augmentation=False,
                                       horizontal_aspect_augmentation=False, horizontal_mirror_augmentation=False,
                                       camera_augmentation=False, half_turn_augmentation=False,
                                       polarity_augmentation=False, anchor_zoom_cap=False)
            observed.append((namespace['__getitem__'](instance, 0), traces))
        (baseline_x, baseline_y), baseline_rng = observed[0]
        (disabled_x, disabled_y), disabled_rng = observed[1]
        ((enabled_x, queries), (enabled_y, occupancy)), enabled_rng = observed[2]
        for actual in (disabled_x, enabled_x): np.testing.assert_array_equal(actual, baseline_x)
        for actual in (disabled_y, enabled_y): np.testing.assert_array_equal(actual, baseline_y)
        self.assertEqual(baseline_rng, disabled_rng); self.assertEqual(baseline_rng, enabled_rng)
        self.assertTrue(np.all(occupancy[0] == -1))  # Old partial stayed excluded despite inside points.
        self.assertEqual(occupancy[1, :, 0].tolist(), [1, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(occupancy[2] == -1))  # Angelina/no point state.


if __name__ == '__main__':
    unittest.main()
