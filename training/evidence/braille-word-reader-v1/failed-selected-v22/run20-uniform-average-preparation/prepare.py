"""Read-only three-checkpoint compatibility and fixed-plan preparation; no ML."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile
import h5py
import numpy as np

HERE=Path(__file__).resolve().parent; W=HERE.parent; ROOT=W.parents[1]
EPOCHS=(8,9,10)
EXPECTED=('1b6c9d810045bbd01a57b98f661557ce46968e41ef433ef78de1d92622b9a1c2',
          'cce2432c0ddb8952c885dd58697589c285209588dafa81aacdd891b1660068b7',
          '01fa26cdf9c0bc7c2d57e0ec0dc79a53f1b70403c2f322d4f634e4d4a68cf5e6')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def load(path):return json.loads(Path(path).read_bytes())
def link(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def frozen_json(path,value):
    with Path(path).open('x',encoding='utf-8') as file:json.dump(value,file,indent=2);file.write('\n')
def layer_index(config):
    result={}
    def visit(node,parent=''):
        cfg=node.get('config',{})
        if not isinstance(cfg,dict) or 'name' not in cfg:return
        name=cfg['name'];assert name not in result,'Layer names must be unique for this graph'
        result[name]={'path':parent+'/'+name,'class_name':node['class_name'],'trainable':cfg.get('trainable')}
        for child in cfg.get('layers',[]):visit(child,result[name]['path'])
    visit(config);return result
def serialized(path):
    with ZipFile(path) as archive:
        assert set(archive.namelist())=={'metadata.json','config.json','model.weights.h5'}
        cfg=json.loads(archive.read('config.json')); graph=dict(cfg);graph.pop('compile_config',None)
        layers=layer_index(graph); arrays={}; inventory=[]; optimizer=[]
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')),'r') as h5:
            def collect(path,node):
                if not isinstance(node,h5py.Dataset):return
                if path.startswith('optimizer/'):
                    optimizer.append({'h5_path':path,'shape':list(node.shape),'dtype':node.dtype.str});return
                assert path.startswith('layers/') and '/vars/' in path,path
                owner=node.parent.attrs.get('name');assert owner in layers
                var_index=int(path.rsplit('/',1)[1]);info=layers[owner]
                data=node[()];assert data.dtype==np.dtype('float32') and np.isfinite(data).all()
                arrays[path]=data
                inventory.append({'h5_path':path,'layer_name':owner,'layer_config_path':info['path'],
                    'layer_class':info['class_name'],'layer_trainable':info['trainable'],'variable_index':var_index,
                    'shape':list(data.shape),'dtype':data.dtype.str,'size_bytes':data.nbytes,'sha256':digest(data.tobytes()),
                    'preserve_bn':info['class_name']=='BatchNormalization'})
            h5.visititems(collect)
        return cfg,graph,arrays,inventory,optimizer

def main():
    assert not (HERE/'plan.json').exists(),'Never overwrite a frozen plan'
    config_path=W/'run20/training-config.json';config=load(config_path)
    audit_path=W/'run20/sampling-lineage-final.json';audit=load(audit_path)
    history_path=W/'run20/sampling-history.json';history=load(history_path)
    assert audit['status']=='passed' and audit['actual']['config_sha256']==sha(config_path)
    assert audit['actual']['sampling_history_sha256']==sha(history_path)
    assert audit['actual']['manifest_sha256']==config['data_sha256'][config['train_manifest']]
    assert audit['eligible_retained_epochs']==list(EPOCHS)
    inputs=[]; models=[]
    for epoch,expected in zip(EPOCHS,EXPECTED,strict=True):
        path=W/'run20/retained-epochs'/f'epoch-{epoch:02d}.keras';assert sha(path)==expected
        exposure=next(row for row in audit['actual']['observed_epochs'] if row['epoch']==epoch)
        seen=next(row for row in history if row['epoch']==epoch)
        assert exposure['checkpoint_sha256']==expected and exposure['eligible_by_sampling_lineage'] and exposure['all_positive_records_seen_in_this_run']
        assert exposure['actual_cumulative_positive_records']==23600
        assert seen['complete_epoch'] and seen['completed_batches']==seen['expected_batches']==1037
        assert all(row['cumulative_unique_records']==row['total_unique_records'] for row in seen['families'].values())
        cfg,graph,arrays,inventory,optimizer=serialized(path);models.append((cfg,graph,arrays,inventory,optimizer))
        inputs.append(dict(link(path),epoch=epoch,weight={'numerator':1,'denominator':3},
            config_json_sha256=digest(ZipFile(path).read('config.json')),graph_sha256=digest(canonical(graph)),
            compile_config_sha256=digest(canonical(cfg.get('compile_config'))),
            model_array_count=len(arrays),optimizer_dataset_count=len(optimizer),sampling=exposure))
    assert all(graph==models[0][1] for _,graph,_,_,_ in models)
    assert all(len(m[2])==280 and len(m[4])==146 for m in models)
    bn=[row for row in models[0][3] if row['preserve_bn']]
    assert len(bn)==208 and len({row['layer_name'] for row in bn})==52
    assert all(row['layer_trainable'] is False and row['variable_index'] in range(4) for row in bn)
    inventory=[]
    for row in models[0][3]:
        path=row['h5_path']; peers=[next(v for v in model[3] if v['h5_path']==path) for model in models]
        identity=lambda item:{k:v for k,v in item.items() if k!='sha256'}
        assert all(identity(p)==identity(row) for p in peers)
        equal=all(models[0][2][path].tobytes()==m[2][path].tobytes() for m in models)
        if row['preserve_bn']:assert equal,path
        inventory.append(dict(identity(row),input_sha256=[p['sha256'] for p in peers],all_three_byte_equal=equal,
            prospective_action='copy_epoch09_exact' if row['preserve_bn'] else 'sum_float64_in_epoch_8_9_10_order_divide_3_cast_float32'))
    assert sha(W/'selected-v15/best-word.keras')==EXPECTED[1]
    frozen_json(HERE/'inventory.json',{'model_arrays':inventory,'optimizer_excluded':models[1][4]})
    frozen_json(HERE/'graph.json',models[1][1])
    fixture=W/'data/validation-stress-selection.json'
    assert sha(fixture)=='e2ab968f7c4dcf7e774181fde687e1321f58d20567528af4b21fd03ff5a3e625'
    selector_names=('word_select.py','word_model.py','word_decode.py','word_data.py','word_geometry_head.py','word_bn_affine.py')
    selector_source=HERE/'selection-source';selector_source.mkdir()
    for name in selector_names:(selector_source/name).write_bytes((ROOT/'training/scripts'/name).read_bytes())
    plan={'schema_version':1,'status':'frozen_preparation_not_evaluated','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'One uniform checkpoint-weight candidate only. No model execution, output calibration, source filtering or new training.',
        'inputs':inputs,'input_epoch_order':list(EPOCHS),'coefficient_rule':'Exactly1/3 each; no subset, greedy addition, coefficient search or learned mixture.',
        'source_training_config':link(config_path),'source_sampling_lineage':link(audit_path),'source_sampling_history':link(history_path),
        'source_parent_checkpoint_sha256':config['resume_sha256'],'source_data_sha256':config['data_sha256'],
        'source_training_sha256':config['source_sha256'],'model_graph_sha256':inputs[0]['graph_sha256'],
        'inventory':link(HERE/'inventory.json'),'model_array_count':280,'bn_array_count_preserved':208,'arrays_to_average':72,
        'arithmetic':'For each non-BN float32 array, promote inputs individually to float64, add in epoch order8,9,10, divide by3 in float64, cast once tofloat32. Copy208BN arrays byte-for-byte fromepoch9.',
        'candidate_path':str(HERE/'uniform-08-09-10.keras'),'candidate_use':'Untested inference-only; never an active asset before original gates. Do not resume training from this bundle.',
        'optimizer_policy':'Drop entire optimizer HDF5 group and set compile_config={}; do not average146optimizer datasets or inherit optimizer learning-rate/step state.',
        'graph_policy':'Preserve epoch9 graph/build serialization exactly; compile_config alone is emptied. No added layers or normalization/BN-stat recalibration.',
        'control_sequence':{'after':'run26 selection completes into selected-v21','control_path':str(W/'selected-v21/best-word.keras'),
            'control_hash_binding':'Append separate evaluation protocol afterselected-v21exists; do not edit this plan.',
            'candidate_order':['current selected-v21 control','uniform-08-09-10.keras'],'output':str(W/'selected-v22'),
            'tie_rule':'Keep first candidate because selector replaces only on rank>best; control first.',
            'before':'Any new external607/44/66 evaluation or TF/export of selected-v21/v22'},
        'selection':{'fixture':link(fixture),'fixture_content_read':False,'expected_records':405,'threshold_grid':'weak-peaks-v2',
            'threshold_grid_values':{'proposal':[.15,.25,.35],'accept':[.2,.25,.3,.35,.4,.55,.65,.75],'dot':[.7,.8,.9,.95]},
            'ranking':'Require accepted precision>=.95 separately on natural/stress; maximizeequal-group meanexact including rejections; unchanged existing fallback/tie behavior.',
            'sources':{name:link(selector_source/name) for name in selector_names},'external_selection':False},
        'source_prepare':link(__file__),'source_average':link(HERE/'average.py'),
        'primary_reference':{'url':'https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf','file':link(HERE/'wortsman22a.pdf'),
            'limits':'Evidence on image/text classification, not this Braille detector. Uniform averages can degrade performance and do not guarantee calibration. Ours is one training trajectory, not the paper main independent-run recipe.'},
        'requirements_before_model_claim':['Independent whole-array arithmetic and graph/BN/source-preservation review','Actual Keras compile=False load and frozen405 control-first comparison afterrun26','If selected, required original conversion/native/word/negative gates remain unchanged'],
        'no_model_execution':True,'no_training_mutation':True}
    frozen_json(HERE/'plan.json',plan)
    print(json.dumps({'plan_sha256':sha(HERE/'plan.json'),'graph_sha256':inputs[0]['graph_sha256'],
        'bn_arrays_exact':208,'other_arrays_equal':sum(r['all_three_byte_equal'] for r in inventory if not r['preserve_bn'])}))
if __name__=='__main__':main()
