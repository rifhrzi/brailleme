"""Run only after the private prototype's focused independent review passes."""
import hashlib
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent;SOURCE=OUT.parents[2]/'training/scripts'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=Path(sys.argv[1]);report=json.loads(review.read_text())
assert report['status']=='passed'
spec=json.loads((OUT/'prototype-spec.json').read_text())
for name,digest in spec['source_original_sha256'].items():assert sha(SOURCE/name)==digest,name
for name,digest in spec['source_private_sha256'].items():assert sha(OUT/'source-private'/name)==digest,name
for name in ('word_anchor_zoom.py','word_data.py','word_train.py'):
    (SOURCE/name).write_bytes((OUT/'source-private'/name).read_bytes())
p=SOURCE/'word_train.py';s=p.read_text()
s=s.replace("'word_horizontal_aspect.py')","'word_horizontal_aspect.py', 'word_anchor_zoom.py')",1)
s=s.replace('        self.anchor_zoom_cap = anchor_zoom_cap','''        if anchor_zoom_cap and (not illumination_augmentation or camera_augmentation or polarity_augmentation
                or half_turn_augmentation or horizontal_mirror_augmentation or vertical_stretch_augmentation
                or horizontal_aspect_augmentation or partial_geometry_supervision):
            raise ValueError('Anchor zoom requires the isolated isotropic illumination profile')
        self.anchor_zoom_cap = anchor_zoom_cap''',1)
needle="    args = parser.parse_args(argv)"
s=s.replace(needle,"    parser.add_argument('--anchor-zoom-cap', action='store_true', help='Cap existing isotropic zoom to preserve one original full known cell at the affine stage.')\n"+needle,1)
needle='    return args\n'
s=s.replace(needle,"""    if args.anchor_zoom_cap and (not args.illumination_augmentation or args.camera_augmentation
            or args.polarity_augmentation or args.half_turn_augmentation or args.horizontal_mirror_augmentation
            or args.vertical_stretch_augmentation or args.horizontal_aspect_augmentation
            or args.partial_geometry_supervision or args.bn_affine_initialization or args.geometry_head_initialization
            or args.fine_branch_initialization or args.image_normalization is not None):
        parser.error('Anchor zoom requires the unchanged isotropic illumination parent without other ablations')
"""+needle,1)
s=s.replace('horizontal_aspect_augmentation=args.horizontal_aspect_augmentation)','horizontal_aspect_augmentation=args.horizontal_aspect_augmentation,anchor_zoom_cap=args.anchor_zoom_cap)',1)
s=s.replace("    config['horizontal_aspect_augmentation']=args.horizontal_aspect_augmentation","    config['horizontal_aspect_augmentation']=args.horizontal_aspect_augmentation\n    config['anchor_zoom_cap']=args.anchor_zoom_cap",1)
needle='    config["retain_epochs"] = args.retain_epochs'
s=s.replace(needle,"""    if args.anchor_zoom_cap:
        config['augmentation_protocol']['anchor_zoom_cap'] = {
            'enabled': True, 'base_zoom_range': [.8, 1.7], 'existing_affine_probability': .85,
            'anchor': 'Nearest canvas center among originally complete known cells; squared Euclidean distance, original index ties.',
            'cap': 'Only reduce sampled isotropic zoom to the maximum affine containment bound after the existing sampled translations; no anchor or bound<=.8 leaves the sampled draw unchanged.',
            'rounding': 'At most eight float64 nextafter ULPs below the analytic boundary, verified with the actual affine arithmetic.',
            'randomness': 'No new RNG or resampling. Existing zoom, translation, rotation and lighting draws remain paired.',
            'visibility': 'Original completeness and occupied-point visibility remain irreversible; unknown slots cannot anchor; later rotation can still cut any cell.',
            'scope': 'One whole-image affine scale adjustment; no labels, padding, model input, decoder, selection fixtures or acceptance gate changes.'}
"""+needle,1)
p.write_text(s,encoding='utf-8')
(OUT/'integration-record.json').write_text(json.dumps({'review':str(review),'review_sha256':sha(review),'private_spec_sha256':sha(OUT/'prototype-spec.json'),'proof_sha256':sha(OUT/'proof-report.json'),'changed_sources':{name:sha(SOURCE/name) for name in ('word_anchor_zoom.py','word_data.py','word_train.py')}},indent=2)+'\n')
print((OUT/'integration-record.json').read_text())
