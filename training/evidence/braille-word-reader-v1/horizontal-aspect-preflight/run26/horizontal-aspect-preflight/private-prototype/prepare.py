"""Prepare PRIVATE copies only; no active source, profile or model changes."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path.cwd().resolve()
BASE=Path(__file__).resolve().parent
RUN=BASE.parent/'run25'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
config=json.loads((RUN/'training-config.json').read_text())
assert config['source_sha256']['word_data.py']=='659ab6b508a69b34bfb8dc899c5fa6e2214bd64e08281f40f9c39aa91d74996c'
assert config['source_sha256']['word_train.py']=='f472b80d0466e17cf1efcb59661b3fea300f5618d4288ec1847fa8d3b628c97b'
original=BASE/'source-original';private=BASE/'source-private'
assert not original.exists()
assert {p.name for p in private.iterdir()}=={'word_horizontal_aspect.py'}
original.mkdir()
for name,expected in config['source_sha256'].items():
    assert sha(RUN/'source'/name)==sha(ROOT/'training/scripts'/name)==expected
    (original/name).write_bytes((RUN/'source'/name).read_bytes())

def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

data=(original/'word_data.py').read_text(encoding='utf-8')
data=replace_once(data,'point_state=None, vertical_stretch=1.0):','point_state=None, vertical_stretch=1.0, horizontal_aspect=1.0):')
data=replace_once(data,"    boxes = boxes.copy()\n    def inside_canvas():", "    if not np.isfinite(horizontal_aspect) or not .55 <= horizontal_aspect <= 1.25:\n        raise ValueError('Horizontal aspect must be finite in [.55, 1.25]')\n    boxes = boxes.copy()\n    def inside_canvas():")
data=replace_once(data,'        scale_y=zoom*vertical_stretch',
    '        if horizontal_aspect != 1.0:\n            scale_x *= horizontal_aspect\n        scale_y=zoom*vertical_stretch')
(private/'word_data.py').write_text(data,encoding='utf-8')
train=(original/'word_train.py').read_text(encoding='utf-8')
train=replace_once(train,'class Patches(keras.utils.PyDataset):',
    'from word_horizontal_aspect import horizontal_aspect_factors\n\n\nclass Patches(keras.utils.PyDataset):')
train=replace_once(train,'point_states=None,vertical_stretch_augmentation=False):',
    'point_states=None,vertical_stretch_augmentation=False,horizontal_aspect_augmentation=False):')
train=replace_once(train,'        self.vertical_stretch_augmentation = vertical_stretch_augmentation',
    '        self.vertical_stretch_augmentation = vertical_stretch_augmentation\n        self.horizontal_aspect_augmentation = horizontal_aspect_augmentation')
train=replace_once(train,'        for batch_offset, record_index in enumerate(batch_indices):',
    '        width_factors = (horizontal_aspect_factors(self.seed, self.epoch, batch_indices)\n                         if self.training and self.horizontal_aspect_augmentation else None)\n        for batch_offset, record_index in enumerate(batch_indices):')
train=replace_once(train,'vertical_stretch=float(height_factors[batch_offset]) if height_factors is not None else 1.0)',
    'vertical_stretch=float(height_factors[batch_offset]) if height_factors is not None else 1.0,\n                                          horizontal_aspect=float(width_factors[batch_offset]) if width_factors is not None else 1.0)')
(private/'word_train.py').write_text(train,encoding='utf-8')
diff=''.join(''.join(difflib.unified_diff((original/name).read_text().splitlines(True),(private/name).read_text().splitlines(True),fromfile='source-original/'+name,tofile='source-private/'+name)) for name in ('word_data.py','word_train.py'))
(BASE/'private-edits.patch').write_text(diff,encoding='utf-8')
spec=dict(scope='PRIVATE preparation only; never used by active run25. No new training, selection, release gate or command-line profile.',
    historical_range=[.55,1.25],range_origin='Unchanged historical run7 camera augmentation; no held/external threshold choice.',
    geometry='Only when existing affine gate succeeds, multiply its scale_x by a per-record factor. Keep zoom, scale_y, translations, rotation and lighting RNG unchanged. Existing affine_patch handles pixels+all boxes once, and OccupiedPointState.affine receives the same scales.',
    stream='SeedSequence([training seed,dataset epoch,combined manifest record index,0x48415350]); one independent uniform per record. Independent of batch grouping/order and all legacy RNG; repeated same record in an epoch repeats factor.',
    off='DefaultFalse Patches flag sends factor1.0; no factor stream call and no scale_x multiply. Existing pixels/boxes/targets/order/RNG must remain exact.',
    intended_profile='Current illumination=True with camera/polarity/half-turn/mirror/vertical=False. Combining with camera-only anisotropy would multiply two factors and is outside this proposal.',
    point_rules='Known DSBI occupied centers only; unchanged sample-center hull and irreversible survival after each raster. All six partial labels remain unknown; unknown slots stay ignore-only; no bit or label remapping.',
    original_training_sources=config['source_sha256'],run25_config_sha256=sha(RUN/'training-config.json'),
    prerequisites='Tests here do not demonstrate useful training or recognition accuracy. Any future integration needs its own frozen protocol and selection after run25 finishes.')
(BASE/'prototype-spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(private_root=str(BASE),original_sources=len(config['source_sha256']),active_sources_unchanged=True)))
