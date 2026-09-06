"""Create a reviewable private integration diff, never edit active sources."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('training/scripts')


def main():
    original = ROOT / 'original'
    original.mkdir(exist_ok=False)
    for name in ('word_train.py', 'word_model.py', 'word_data.py', 'word_point_geometry.py', 'word_geometry_head.py'):
        (original / name).write_bytes((SOURCE / name).read_bytes())
    text = (original / 'word_train.py').read_text()
    replacements = []

    def replace(before, after):
        nonlocal text
        assert text.count(before) == 1, (before[:100], text.count(before))
        text = text.replace(before, after)
        replacements.append({'before': before, 'after': after})

    replace('from word_normalization_transition import validated_normalization_transition, training_normalization_metadata',
            'from word_normalization_transition import validated_normalization_transition, training_normalization_metadata\nfrom word_dot_aux import AUXILIARY_DOT_COEFFICIENT, wrap_dot_training\nfrom word_dot_aux_targets import complete_dot_queries, pad_dot_queries')
    replace("'word_anchor_zoom.py', 'word_normalization_transition.py')", "'word_anchor_zoom.py', 'word_normalization_transition.py', 'word_dot_aux.py', 'word_dot_aux_targets.py')")
    replace('point_states=None,vertical_stretch_augmentation=False,horizontal_aspect_augmentation=False,anchor_zoom_cap=False):',
            'point_states=None,vertical_stretch_augmentation=False,horizontal_aspect_augmentation=False,anchor_zoom_cap=False,auxiliary_dot_supervision=False):')
    replace('        self.point_states = point_states', '        self.point_states = point_states\n        self.auxiliary_dot_supervision = auxiliary_dot_supervision')
    replace('        images, labels = [], []', '        images, labels = [], []\n        auxiliary_rows = []')
    replace('        return np.stack(images), np.stack(labels)',
            "            if self.auxiliary_dot_supervision:\n                auxiliary_rows.append(complete_dot_queries(point_state, masks[:positive_count], eligible[:positive_count]))\n        if self.auxiliary_dot_supervision:\n            queries, occupancy = pad_dot_queries(auxiliary_rows)\n            return (np.stack(images), queries), (np.stack(labels), occupancy)\n        return np.stack(images), np.stack(labels)")
    replace('def __init__(self, root, output, batch_size, retain_epochs=False,train_dataset=None,expected_bn_moments=None):',
            'def __init__(self, root, output, batch_size, retain_epochs=False,train_dataset=None,expected_bn_moments=None,reader=None):')
    replace('        self.expected_bn_moments = expected_bn_moments', '        self.expected_bn_moments = expected_bn_moments\n        self.reader = reader')
    replace('        callback_started=time.perf_counter()\n        moment_hash = None',
            '        callback_started=time.perf_counter()\n        reader = self.reader if self.reader is not None else self.model\n        moment_hash = None')
    replace('moment_hash = moving_moments_sha256(self.model)', 'moment_hash = moving_moments_sha256(reader)')
    replace('        self.model.save(self.output/"last.keras")',
            '        reader.save(self.output/"last.keras")\n        if self.reader is not None:\n            self.model.save(self.output/"last-training.keras")')
    replace('predictions = self.model.predict(self.images, batch_size=self.batch_size, verbose=0)',
            'predictions = reader.predict(self.images, batch_size=self.batch_size, verbose=0)')
    replace('            self.model.save(self.output/"best-word.keras")', '            reader.save(self.output/"best-word.keras")')
    replace("    parser.add_argument('--normalization-runtime-proof', type=Path, help='Actual std.15 six-input Torch/TF/TFLite runtime proof.')",
            "    parser.add_argument('--normalization-runtime-proof', type=Path, help='Actual std.15 six-input Torch/TF/TFLite runtime proof.')\n    parser.add_argument('--auxiliary-dot-supervision', action='store_true', help='Training-only explicit DSBI complete-cell occupancy objective with fixed weight .25.')")
    replace('    return args\n\n\n',
            "    if args.auxiliary_dot_supervision and (not args.resume or not args.illumination_augmentation\n            or not args.dsbi_visible_partial_supervision or not args.balance_source_families\n            or args.anchor_zoom_cap or args.horizontal_aspect_augmentation or args.vertical_stretch_augmentation\n            or args.horizontal_mirror_augmentation or args.camera_augmentation or args.half_turn_augmentation\n            or args.polarity_augmentation or args.partial_geometry_supervision or args.geometry_head_initialization\n            or args.bn_affine_initialization or args.fine_branch_initialization or args.normalization_initialization\n            or args.image_normalization is not None):\n        parser.error('Auxiliary dots require the unchanged full-tail illumination/DSBI reader without other ablations')\n    return args\n\n\n")
    replace('anchor_zoom_cap=args.anchor_zoom_cap)\n    validation = Patches',
            'anchor_zoom_cap=args.anchor_zoom_cap,auxiliary_dot_supervision=args.auxiliary_dot_supervision)\n    validation = Patches')
    replace('False, args.seed+1,partial_geometry_supervision=args.partial_geometry_supervision)',
            'False, args.seed+1,partial_geometry_supervision=args.partial_geometry_supervision,auxiliary_dot_supervision=args.auxiliary_dot_supervision)')
    replace('    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate, clipnorm=10.), loss=word_relative_loss if args.geometry_loss==\'relative_size_huber\' else word_loss, jit_compile=False)',
            "    if args.auxiliary_dot_supervision:\n        training_model = wrap_dot_training(model, word_relative_loss if args.geometry_loss=='relative_size_huber' else word_loss, args.learning_rate)\n    else:\n        model.compile(optimizer=keras.optimizers.Adam(args.learning_rate, clipnorm=10.), loss=word_relative_loss if args.geometry_loss=='relative_size_huber' else word_loss, jit_compile=False)\n        training_model = model")
    replace("    config['fine_detail_branch'] = fine_branch", "    config['fine_detail_branch'] = fine_branch\n    config['auxiliary_dot_supervision'] = {'enabled': args.auxiliary_dot_supervision}\n    if args.auxiliary_dot_supervision:\n        config['auxiliary_dot_supervision'].update(coefficient=AUXILIARY_DOT_COEFFICIENT,\n            target='Explicit mask bits at actual transformed DSBI TRAIN points, complete at every raster, all six positions within stride4/origin3 feature hull.',\n            unknown_policy='No points for Angelina, unknown slots or partial cells; missing auxiliary labels are -1.',\n            denominator='Separate known-point count; original main loss and denominator unchanged.',\n            validation='All validation records have ignored auxiliary queries; val_loss is the original reader loss. Whole-word callbacks use reader only.',\n            training_parameters=training_model.count_params(),\n            checkpoints='Selectable last/best/retained Keras files contain reader only; last-training.keras separately preserves full wrapper and optimizer. Reader resume starts a fresh auxiliary head and optimizer; full-state resume needs an explicit separate workflow.')")
    replace('    history = model.fit(train,', '    history = training_model.fit(train,')
    replace("expected_bn_moments=batchnorm_affine.get('moving_moments_sha256'))", "expected_bn_moments=batchnorm_affine.get('moving_moments_sha256'),reader=model if args.auxiliary_dot_supervision else None)")
    ast.parse(text)
    candidate = ROOT / 'candidate/word_train.py'
    candidate.write_text(text)
    diffs = []
    for name in ('word_train.py', 'word_dot_aux_targets.py', 'word_dot_aux.py'):
        before = (original / name).read_text().splitlines(keepends=True) if (original / name).exists() else []
        after = (ROOT / 'candidate' / name).read_text().splitlines(keepends=True)
        diffs.extend(difflib.unified_diff(before, after, fromfile='a/training/scripts/'+name if before else '/dev/null', tofile='b/training/scripts/'+name))
    (ROOT / 'integration.patch').write_text(''.join(diffs))
    (ROOT / 'replacements.json').write_text(json.dumps(replacements, indent=2) + '\n')
    print(json.dumps({'replacements': len(replacements), 'candidate_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(), 'patch_sha256': hashlib.sha256((ROOT / 'integration.patch').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
