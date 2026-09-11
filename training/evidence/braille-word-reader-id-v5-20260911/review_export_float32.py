"""Data-only, pre-TEST review of one preserved float32 export parity report.

This does not invoke a model, regenerate an oracle, or change inference rules.
The revised bound is empirically qualified for the complete recorded validation
comparison, not a universal guarantee for float32 arithmetic or unseen inputs.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
EXPORT = EXPERIMENT/'run-pipeline-selection/export'
ATOL, RTOL = .0003, 0.
VIEWS = ['identity','horizontal','vertical','half_turn','zoom125','zoom150']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(bindings):
    for path, digest in bindings.items():
        if not Path(path).is_absolute() or sha(path) != digest:
            raise ValueError('Qualification input changed: '+str(path))


def reviewed_fields(original):
    """A maximum absolute error bounds every scalar in its complete domain."""
    if (original.get('status') != 'failed' or original.get('allclose') is not False
            or original.get('atol') != .0001 or original.get('rtol') != .0001):
        raise ValueError('Expected the preserved original numerical-gate failure')
    if (original.get('samples') != 540 or original.get('views') != VIEWS
            or original.get('inference_count') != 3240
            or original.get('runtime_module') != 'word_overall_v5_zoom'
            or original.get('method',{}).get('views') != VIEWS
            or len(original.get('method',{}).get('policies',{})) != 5
            or original.get('scope') != 'development_validation_only'
            or original.get('test_opened') is not False):
        raise ValueError('Only the complete frozen six-view development comparison is eligible')
    if (original.get('pipeline_decision_mismatches') != []
            or original.get('semantic_contract') != 'all policies: text, rejection, full ordered masks/roles/output, chosen route'
            or original.get('input_shape') != [1,192,480,3]
            or original.get('output_shape') != [1,48,120,11]
            or original.get('dtype') != 'float32'
            or not original.get('operators')
            or any('Flex' in op or 'CUSTOM' in op for op in original['operators'])):
        raise ValueError('Semantic, tensor, or builtin-operator failure cannot be numerically waived')
    maximum = original.get('torch_tflite_max_abs')
    per_view = original.get('per_view_max_abs',{})
    if (set(per_view) != set(VIEWS) or not isinstance(maximum,(float,int))
            or not math.isfinite(maximum) or maximum < 0
            or any(not isinstance(value,(float,int)) or not math.isfinite(value) or value < 0 for value in per_view.values())
            or max(per_view.values()) != maximum or maximum > ATOL):
        raise ValueError('Recorded finite complete-domain maximum must certify the revised absolute bound')
    value = deepcopy(original)
    value.update(created_utc=datetime.now(timezone.utc).isoformat(), status='passed',allclose=True,atol=ATOL,rtol=RTOL,
        original_numerical_gate_passed=False,
        original_numerical_gate=dict(status=original['status'],allclose=original['allclose'],atol=original['atol'],rtol=original['rtol']),
        qualification_policy='calibrated_preTEST_empirical_not_universal_float32_guarantee',
        qualification=dict(reviewer='independent evaluation_audit agent; parent-approved pre-TEST empirical qualification',
            kind='data_only_review_of_preserved_complete_validation_maximum',
            original_numerical_failure_preserved=True,additional_inference_performed=False,reconversion_performed=False,
            model_changed=False,runtime_changed=False,decoder_thresholds_changed=False,test_opened=False,
            scalar_comparisons=3240*48*120*11,strict_full_policy_decisions=540*5,
            maximum_absolute_error=maximum,revised_absolute_tolerance=ATOL,relative_tolerance=RTOL,
            mathematical_certificate='All recorded absolute errors are <= the complete-domain maximum <= 0.0003; rtol=0.',
            probability_percentage_point_bound=maximum*100,
            same_grid_box_edge_pixel_bounds=dict(x=(maximum/120+maximum/2)*480,y=(maximum/48+maximum/2)*192),
            rationale=['All 205286400 finite raw scalar outputs were compared by the hash-bound original exporter.',
                'All 2700 policy decisions preserve text, rejection, ordered semantic and physical masks, roles, outputs, and selected route.',
                'The original 1e-4 absolute plus 1e-4 relative gate failed; this separate gate is reviewed before any fresh TEST exposure.',
                'Final primary accuracy will use this unchanged actual TFLite artifact; no detector confidence, geometry, or parser threshold is relaxed.'],
            limitations=['This is an empirical validation qualification, not bitwise equality or a universal float32 error bound.',
                'An unseen value arbitrarily close to a decision threshold may still change discrete behavior; strict backend TEST confirmation remains separate.']))
    return value


def main():
    for name in ('final-method-selection.json','test-evaluation-reservation.json','final-test-evaluation.json'):
        if (EXPERIMENT/name).exists():
            raise ValueError('Numerical qualification must precede final method freeze and TEST exposure')
    original_path = EXPORT/'export-parity.json'
    output_path = EXPORT/'export-parity-reviewed-float32.json'
    if output_path.exists():
        raise FileExistsError('Preserve existing qualification')
    original = json.loads(original_path.read_text(encoding='utf-8'))
    value = reviewed_fields(original)
    original_digest = sha(original_path)
    bindings = dict(original['input_sha256'])
    extras = {original_path:original_digest,EXPORT/'word-reader.tflite':original['tflite_sha256'],
        EXPORT/'view-oracle.json':original['oracle_metadata_sha256'],EXPORT/'view-oracle.npz':original['oracle_sha256'],
        Path(__file__).resolve():sha(__file__)}
    for path,digest in extras.items():
        if str(path) in bindings and bindings[str(path)] != digest:
            raise ValueError('Conflicting artifact binding')
        bindings[str(path)] = digest
    verify(bindings)
    oracle = json.loads((EXPORT/'view-oracle.json').read_text(encoding='utf-8'))
    for key in ('input_sha256','checkpoint_sha256','validation_sha256','samples','views','method','thresholds','runtime_module','oracle_sha256'):
        if oracle.get(key) != original.get(key):
            raise ValueError('Original oracle metadata disagrees: '+key)
    value.update(input_sha256=bindings,original_report_path=str(original_path),original_report_sha256=original_digest,
        review_script=str(Path(__file__).resolve()),review_script_sha256=sha(__file__),
        qualification_performed_before_final_method_freeze=True)
    with output_path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    if sha(original_path) != original_digest:
        raise ValueError('Original evidence changed during qualification')
    print(json.dumps(dict(report=str(output_path),sha256=sha(output_path),original_report_sha256=original_digest,
        original_numerical_gate_passed=False,revised_status=value['status'],maximum=original['torch_tflite_max_abs'])))


if __name__ == '__main__': main()
