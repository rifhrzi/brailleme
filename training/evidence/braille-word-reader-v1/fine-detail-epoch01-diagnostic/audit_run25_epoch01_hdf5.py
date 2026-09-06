"""Read-only actual stride2 epoch1 inventory; no ML backend loading."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import h5py
import numpy as np

WORK=Path(__file__).resolve().parent
RUN=WORK/'run25'
OUT=RUN/'epoch01-independent-hdf5-audit'
def sha(data):return hashlib.sha256(data).hexdigest()

def inventory(path):
    with zipfile.ZipFile(path) as archive:
        configuration=json.loads(archive.read('config.json'))
        weights=archive.read('model.weights.h5')
    backbone=next(layer for layer in configuration['config']['layers'] if layer['config']['name']=='mobilenetv2_half')
    bn=[layer for layer in backbone['config']['layers'] if layer['class_name']=='BatchNormalization']
    assert len(bn)==52 and all(layer['config']['trainable'] is False for layer in bn)
    names=[layer['config']['name'] for layer in bn]
    values,metadata,auxiliary={},{},[]
    with h5py.File(io.BytesIO(weights),'r') as handle:
        def visit(name,node):
            if not isinstance(node,h5py.Dataset):return
            if not name.startswith('layers/'):
                auxiliary.append(dict(path=name,shape=list(node.shape),dtype=str(node.dtype)))
                return
            data=np.asarray(node)
            assert data.dtype==np.float32 and np.isfinite(data).all()
            layer_name=node.parent.attrs['name']
            index=int(name.rsplit('/',1)[1])
            if layer_name in names:
                variable=('gamma','beta','moving_mean','moving_variance')[index]
                group='moving_moments' if index>=2 else 'frozen_bn_affine'
            elif layer_name=='fpn_2_depthwise':
                assert index==0
                variable,group='depthwise_kernel','new_depthwise'
            elif layer_name=='fpn_2_zero':
                variable,group=('kernel','bias')[index],'new_pointwise'
            else:
                variable=str(index)
                group='backbone_convolution' if name.startswith('layers/functional/') else 'fpn_and_output_head'
            values[name]=data
            metadata[name]=dict(layer=layer_name,variable=variable,category=group,shape=list(data.shape),dtype=str(data.dtype))
        handle.visititems(visit)
    assert len(values)==283
    digest=hashlib.sha256()
    for layer_name in names:
        for variable in ('moving_mean','moving_variance'):
            paths=[name for name,meta in metadata.items() if meta['layer']==layer_name and meta['variable']==variable]
            assert len(paths)==1
            data=np.asarray(values[paths[0]],dtype='<f4')
            digest.update(json.dumps([layer_name,variable,list(data.shape)],separators=(',',':')).encode())
            digest.update(data.tobytes(order='C'))
    return values,metadata,auxiliary,digest.hexdigest()

def main():
    assert not OUT.exists()
    initial=WORK/'fine-branch-parent-v2/fine-parent.keras'
    retained=RUN/'retained-epochs/epoch-01.keras'
    # History is written after the retained model copy and validation complete.
    history_bytes=(RUN/'validation-history.json').read_bytes()
    row=next(record for record in json.loads(history_bytes) if record['epoch']==1)
    initial_sha,retained_sha=sha(initial.read_bytes()),sha(retained.read_bytes())
    assert initial_sha=='f5f3b239d0ead7fea9d8f6d8d95d82b9c8fa5a678bdd67c1d7db7246406e6b4e'
    config_bytes=(RUN/'training-config.json').read_bytes()
    config=json.loads(config_bytes)
    config_sha=sha(config_bytes)
    assert config_sha=='a337384ac622ad6c393f627cadf0a71d8c3e0b0b604c1338f8464578641df550'
    assert config['resume_sha256']==initial_sha
    sources=config['source_sha256']
    assert len(sources)==21
    for name,expected in sources.items():assert sha((RUN/'source'/name).read_bytes())==expected
    spec=config['fine_detail_branch']
    assert spec['enabled'] and spec['initial_checkpoint_sha256']==initial_sha
    preflight=WORK/'fine-branch-parent-v2/fine-initialization.json'
    assert sha(preflight.read_bytes())==spec['initialization_report_sha256']
    before,metadata,initial_aux,moving_before=inventory(initial)
    after,after_metadata,trained_aux,moving_after=inventory(retained)
    assert metadata==after_metadata and before.keys()==after.keys()
    assert moving_before==moving_after=='1f190a8fe5f4b3f7162147f7cf254a9fc437dfd20e0b34f3603003fffdc00ae7'
    rows=[]
    for name in sorted(before):
        old,new=before[name],after[name]
        rows.append(dict(path=name,**metadata[name],identical_bytes=old.tobytes()==new.tobytes(),
            initial_sha256=sha(old.tobytes()),retained_sha256=sha(new.tobytes()),
            initial_nonzero_values=int(np.count_nonzero(old)),retained_nonzero_values=int(np.count_nonzero(new)),
            changed_values=int(np.count_nonzero(old!=new)),
            max_abs_change=float(np.max(np.abs(new.astype(np.float64)-old.astype(np.float64))))))
    categories=('moving_moments','frozen_bn_affine','backbone_convolution','fpn_and_output_head','new_depthwise','new_pointwise')
    counts={category:dict(arrays=sum(row['category']==category for row in rows),
        changed_arrays=sum(row['category']==category and not row['identical_bytes'] for row in rows),
        changed_values=sum(row['changed_values'] for row in rows if row['category']==category)) for category in categories}
    for category in ('moving_moments','frozen_bn_affine'):
        assert counts[category]['arrays']==104 and counts[category]['changed_arrays']==0
    assert counts['new_depthwise']['arrays']==counts['new_depthwise']['changed_arrays']==1
    assert counts['new_pointwise']['arrays']==counts['new_pointwise']['changed_arrays']==2
    for row2 in rows:
        if row2['category']=='new_pointwise':assert row2['initial_nonzero_values']==0 and row2['retained_nonzero_values']>0
        if row2['category']=='new_depthwise':assert row2['initial_nonzero_values']>0 and row2['retained_nonzero_values']>0
    for category in ('backbone_convolution','fpn_and_output_head'):assert counts[category]['changed_arrays']>0
    assert sha(initial.read_bytes())==initial_sha and sha(retained.read_bytes())==retained_sha
    assert (RUN/'training-config.json').read_bytes()==config_bytes
    assert next(record for record in json.loads((RUN/'validation-history.json').read_bytes()) if record['epoch']==1)==row
    assert not any(name in sys.modules for name in ('keras','tensorflow','torch'))
    OUT.mkdir()
    (OUT/'validation-history-snapshot.json').write_bytes(history_bytes)
    report=dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),status='passed',
        scope='Read-only actual epoch1 stored model arrays. No model loading, inference, training, selection or accuracy claim.',
        initial_checkpoint=str(initial),initial_checkpoint_sha256=initial_sha,retained_checkpoint=str(retained),retained_checkpoint_sha256=retained_sha,
        training_config_sha256=config_sha,training_source_sha256=sources,initialization_report_sha256=sha(preflight.read_bytes()),
        callback_history_snapshot_sha256=sha(history_bytes),callback_epoch1=row,
        moving_moments_sha256=moving_before,
        moving_digest_method='BN layer order from serialized config; JSON[layer,variable,shape] then little-endian float32 C bytes. Independently computed, not a claim that run25 callback enforces a BN digest.',
        model_array_count=len(rows),counts=counts,arrays=rows,
        optimizer_and_other_nonmodel_arrays=dict(initial=initial_aux,retained=trained_aux),
        script_sha256=sha(Path(__file__).read_bytes()),h5py_version=h5py.__version__,numpy_version=np.__version__,backend_modules_imported=[],
        epoch_is_sampling_eligible=False,
        limitations='Only epoch1 arrays audited. Changed/nonzero kernels prove stored updates, not useful activations, gradient correctness, better recognition, conversion parity, later-epoch invariants or release readiness. Epoch1 remains ineligible before the complete sampling cycle.')
    target=OUT/'report.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'audit_run25_epoch01_hdf5.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(report=str(target),sha256=sha(target.read_bytes()),retained_sha256=retained_sha,counts=counts,moving_sha256=moving_before)))

if __name__=='__main__':main()
