"""Collect independently counted TFLite symbol results and bounded training provenance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from word_data import save_json
from word_symbol_prepare import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    args=parser.parse_args();experiment=args.experiment
    report_path=experiment/'selection/symbol-readiness.json'
    if report_path.exists():
        raise SystemExit('Preserve existing readiness report; use a new evidence destination')
    reference=json.loads((args.replay/'python-reference.json').read_text())
    fixture_path=experiment/'validation-symbols.json'
    fixtures=json.loads(fixture_path.read_text())['records']
    selection=json.loads((experiment/'selection/checkpoint-selection.json').read_text())
    config=selection['configuration']
    rows=[]
    if len(fixtures)!=len(reference['rows']):
        raise ValueError('Symbol replay is incomplete')
    for fixture,row in zip(fixtures,reference['rows'],strict=True):
        if fixture['image']!=row['image'] or fixture['expected']!=row['expected']:
            raise ValueError('Replay row identity changed')
        rows.append(dict(image=row['image'],source_kind=fixture['origin'],group=fixture['symbol_group'],expected=row['expected'],predicted=row['python_text'],rejection=row['python_reason']))
    groups={}
    for group in ['lowercase','capital_single','capital_word','numbers']:
        values=[row for row in rows if row['group']==group]
        accepted=sum(bool(row['predicted']) for row in values)
        correct=sum(row['predicted']==row['expected'] for row in values)
        groups[group]=dict(total=len(values),accepted=accepted,correct=correct,exact_word_accuracy=correct/len(values),accepted_accuracy=correct/accepted if accepted else None)
    mandatory=[row for row in rows if row['group']=='rejection']
    false_acceptances=sum(bool(row['predicted']) for row in mandatory)
    groups['rejection']=dict(total=len(mandatory),false_acceptances=false_acceptances,all_rejected=false_acceptances==0)
    frozen=json.loads((experiment/'baseline-freeze.json').read_text())
    expected={k.replace('\\','/'):v for k,v in frozen['files'].items()}['training/checkpoints/braille-word-reader-v1/best-word.keras']
    if digest(experiment/'baseline.keras')!=expected:
        raise ValueError('Protected baseline checkpoint changed')
    if config['epochs_limit']>10 or config['learning_rate']!=.00002:
        raise ValueError('The training run exceeded the authorized experiment bounds')
    gates=false_acceptances==0 and all(value['exact_word_accuracy']>=.75 and (value['accepted_accuracy'] or 0)>=.95 for name,value in groups.items() if name!='rejection')
    report=dict(schema_version=1,scope='synthetic_recomposed_public_photograph_validation_only',source_kind='synthetic_recomposed_public_cell_photographs',model_sha256=reference['model_sha256'],decoder_version=reference['decoderVersion'],symbol_protocol=reference['symbolProtocol'],thresholds=reference['thresholds'],rows=rows,groups=groups,mandatory_rejections=mandatory,baseline_groups=selection['baseline'],candidate_comparisons=selection['candidates'],training_epochs_completed=len(selection['candidates']),initial_learning_rate=config['learning_rate'],batch_size=config['batch_size'],training=dict(epochs_completed=len(selection['candidates']),epochs_limit=config['epochs_limit'],learning_rate=config['learning_rate'],batch_size=config['batch_size'],single_bounded_round=True,baseline_checkpoint_sha256=expected,baseline_preserved=True,selected_checkpoint_sha256=selection['selected_sha256'],baseline_retained=selection['baseline_retained']),numeric_symbol_gates_pass=gates,physical_camera_validated=False,natural_indonesian_capital_number_samples=0,release_ready=False,limitations=['Synthetic cell recomposition cannot establish natural Indonesian page or physical camera accuracy.','Held validation lacks dot6 marker source annotations; absent six-dot patterns use explicit geometric transforms of held source cells.','Source page splitting precedes crops and recomposition; no public external607 rows entered training or checkpoint selection.','A grammar-correct result can still be wrong if the model misses an indicator.'],provenance=dict(corpus_audit_sha256=digest(experiment/'corpus-audit.json'),transform_audit_sha256=digest(experiment/'synthetic-transform-audit.json'),fixture_sha256=digest(fixture_path),python_reference_sha256=digest(args.replay/'python-reference.json'),checkpoint_selection_sha256=digest(experiment/'selection/checkpoint-selection.json')))
    save_json(report_path,report)
    print(json.dumps({key:report[key] for key in ['groups','mandatory_rejections','training','numeric_symbol_gates_pass'] if key!='mandatory_rejections'}),flush=True)


if __name__=='__main__':
    main()
