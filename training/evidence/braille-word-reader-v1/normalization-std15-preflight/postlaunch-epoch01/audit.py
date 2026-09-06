"""Actual run29 epoch1 inventory, HDF5/config only; no backend loading."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import h5py
import numpy as np

OUT=Path(__file__).resolve().parent;RUN=OUT.parent;WORK=RUN.parent
INIT=WORK/'normalization-std15-parent-v1/std15-parent.keras'
INITIAL_SHA='918d148a43399642f44c5d18693eb2733eecca338f0332fed8782cbe79beea00'
RETAINED=RUN/'retained-epochs/epoch-01.keras'
CONFIG_SHA='c20b26876b042015fdd580d6acfaf3509d34edbfb7f1fbcc1b8709d719ac5dfb'
def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    assert not (OUT/'report.json').exists()
    started=datetime.now(timezone.utc).isoformat()
    config_raw=(RUN/'training-config.json').read_bytes();assert sha(config_raw)==CONFIG_SHA
    config=json.loads(config_raw)
    assert config['resume_sha256']==INITIAL_SHA and config['parameters']==829307
    assert config['imageNormalization']['targetStd']==.15 and config['imageNormalization']['stdFloor']==.001
    assert config['feature_pyramid_strides']==[32,16,8,4]
    assert len(config['source_sha256'])==24
    for name,expected in config['source_sha256'].items():assert sha((RUN/'source'/name).read_bytes())==expected
    history_raw=(RUN/'validation-history.json').read_bytes()
    epoch=next(row for row in json.loads(history_raw) if row['epoch']==1)
    assert epoch['sampling']['complete_epoch'] and epoch['sampling']['completed_batches']==1037
    retained_sha=sha(RETAINED.read_bytes())
    original_reader=WORK/'run27-parent-independent-audit/audit.py'
    reader_bytes=original_reader.read_bytes()
    assert sha(reader_bytes)=='ceaa8908add7b37a0287521cc9df181bb0f4f14a1ec95ced0eb55665faf86fc8'
    nodes=[node for node in ast.parse(reader_bytes).body if isinstance(node,ast.FunctionDef) and node.name=='inventory']
    assert len(nodes)==1
    namespace=dict(sha=sha,io=io,json=json,zipfile=zipfile,h5py=h5py,np=np,hashlib=hashlib)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(original_reader),'exec'),namespace)
    before=namespace['inventory'](INIT,INITIAL_SHA);after=namespace['inventory'](RETAINED,retained_sha)
    assert before['metadata']==after['metadata'] and before['arrays'].keys()==after['arrays'].keys()
    graph=lambda c:{k:v for k,v in c.items() if k!='compile_config'}
    assert graph(before['config'])==graph(after['config'])
    norm=next(l for l in after['config']['config']['layers'] if l['class_name']=='CameraNormalization')
    assert norm['config']['target_std']==.15 and norm['config']['std_floor']==.001
    backbone=next(l for l in after['config']['config']['layers'] if l['config']['name']=='mobilenetv2_half')
    assert backbone['config']['output_layers']==[['block_3_expand_relu',0,0],['block_6_expand_relu',0,0],['block_13_expand_relu',0,0],['out_relu',0,0]]
    assert all(n.get('kwargs',{}).get('training') is False for n in backbone['inbound_nodes'])
    assert before['moving_sha256']==after['moving_sha256']=='1f190a8fe5f4b3f7162147f7cf254a9fc437dfd20e0b34f3603003fffdc00ae7'
    rows=[]
    for name in sorted(before['arrays']):
        old,new=before['arrays'][name],after['arrays'][name]
        rows.append(dict(path=name,**before['metadata'][name],initial_sha256=sha(old.tobytes()),retained_sha256=sha(new.tobytes()),
            identical_bytes=old.tobytes()==new.tobytes(),changed_values=int(np.count_nonzero(old!=new)),
            max_abs_change=float(np.max(np.abs(new.astype(np.float64)-old.astype(np.float64))))))
    categories=('bn_moving','bn_gamma_beta','backbone_convolution','fpn_and_output_head')
    counts={category:dict(arrays=sum(r['category']==category for r in rows),
        changed_arrays=sum(r['category']==category and not r['identical_bytes'] for r in rows),
        changed_values=sum(r['changed_values'] for r in rows if r['category']==category)) for category in categories}
    for category in categories[:2]:assert counts[category]['arrays']==104 and counts[category]['changed_arrays']==0
    assert counts['backbone_convolution']['arrays']==counts['backbone_convolution']['changed_arrays']==52
    assert counts['fpn_and_output_head']['arrays']==counts['fpn_and_output_head']['changed_arrays']==20
    assert len(rows)==280 and sum(v.size for v in after['arrays'].values())==829307
    assert sha(INIT.read_bytes())==INITIAL_SHA and sha(RETAINED.read_bytes())==retained_sha
    assert (RUN/'training-config.json').read_bytes()==config_raw
    assert next(row for row in json.loads((RUN/'validation-history.json').read_bytes()) if row['epoch']==1)==epoch
    for name,expected in config['source_sha256'].items():assert sha((RUN/'source'/name).read_bytes())==expected
    assert not any(name in sys.modules for name in ('keras','tensorflow','torch'))
    (OUT/'hdf5_inventory_source.py').write_bytes(reader_bytes)
    (OUT/'training-config-snapshot.json').write_bytes(config_raw)
    (OUT/'validation-history-snapshot.json').write_bytes(history_raw)
    report=dict(schema_version=1,status='passed',started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only actual stored run29 epoch1 model arrays and graph; no model loading, inference, optimizer execution, training or validation re-evaluation.',
        initial_checkpoint=str(INIT),initial_checkpoint_sha256=INITIAL_SHA,retained_checkpoint=str(RETAINED),retained_checkpoint_sha256=retained_sha,
        training_configuration_sha256=CONFIG_SHA,training_source_sha256=config['source_sha256'],
        callback_history_snapshot_sha256=sha(history_raw),callback_epoch1=epoch,
        model_arrays=280,model_parameters=829307,all_model_array_names_shapes_dtypes_identical=True,
        target_std=.15,std_floor=.001,full_tail_preserved=True,serialized_topology_equal=True,
        topology_comparison='All serialized fields except compile_config are exact; initializer intentionally uncompiled and training checkpoint carries compile metadata.',
        initial_compile_configuration=before['config'].get('compile_config'),retained_compile_configuration=after['config'].get('compile_config'),
        initial_serialized_config_sha256=sha(before['config_bytes']),retained_serialized_config_sha256=sha(after['config_bytes']),
        frozen_batchnorm_arrays_byte_identical=208,moving_moments_sha256=after['moving_sha256'],counts=counts,arrays=rows,
        optimizer_arrays_scope='HDF5 reader inventories non-model dataset paths/shapes only and excludes them from model comparisons; no optimizer state is loaded or executed.',
        nonmodel_dataset_counts=dict(initial=len(before['auxiliary']),retained=len(after['auxiliary'])),
        audit_source_sha256=sha(Path(__file__).read_bytes()),inventory_source_sha256=sha(reader_bytes),
        backend_modules_imported=[],versions={'h5py':h5py.__version__,'numpy':np.__version__},
        epoch_sampling_eligible=False,
        limitations=['Actual stored convolution/head updates do not prove useful recognition, gradient attribution or improved generalization.',
                     'Only epoch1 is checked. Its local sampling cycle is incomplete, so it is not selection eligible.',
                     'No later checkpoint, full parity, public regression, native runtime, physical camera or APK readiness claim.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    files=[dict(file=p.name,sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file()]
    (OUT/'index.json').write_text(json.dumps(dict(files=files),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report_sha256=sha((OUT/'report.json').read_bytes()),index_sha256=sha((OUT/'index.json').read_bytes()),retained_sha256=retained_sha,counts=counts)))

if __name__=='__main__':main()
