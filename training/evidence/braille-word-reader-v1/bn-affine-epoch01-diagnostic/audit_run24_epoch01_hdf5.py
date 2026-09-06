"""Read-only HDF5 inventory of the first actual fixed-moment BN training epoch."""
from collections import Counter
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
RUN=WORK/'run24'
OUT=RUN/'epoch01-independent-hdf5-audit'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inventory(path):
    with zipfile.ZipFile(path) as archive:
        configuration=json.loads(archive.read('config.json'))
        weights=archive.read('model.weights.h5')
    backbone=next(layer for layer in configuration['config']['layers'] if layer['config']['name']=='mobilenetv2_half')
    names=[layer['config']['name'] for layer in backbone['config']['layers'] if layer['class_name']=='FrozenMomentBatchNormalization']
    assert len(names)==52 and len(set(names))==52
    values,metadata={},{}
    auxiliary=[]
    with h5py.File(io.BytesIO(weights),'r') as handle:
        def visit(name,node):
            if not isinstance(node,h5py.Dataset):return
            if not name.startswith('layers/'):
                auxiliary.append({'path':name,'shape':list(node.shape),'dtype':str(node.dtype)})
                return
            data=np.asarray(node)
            assert data.dtype==np.float32 and np.isfinite(data).all()
            layer_name=node.parent.attrs['name']
            index=int(name.rsplit('/',1)[1])
            if layer_name in names:
                variable=('gamma','beta','moving_mean','moving_variance')[index]
                group='moving_moments' if index>=2 else 'bn_affine'
            else:
                variable=str(index)
                group='backbone_convolution' if name.startswith('layers/functional/') else 'fpn_and_output_head'
            values[name]=data
            metadata[name]={'layer':layer_name,'variable':variable,'category':group,'shape':list(data.shape),'dtype':str(data.dtype)}
        handle.visititems(visit)
    assert len(values)==280
    # Same ordered identity, shape and little-endian raw-array serialization
    # as the frozen word_train.py callback, reconstructed without Keras objects.
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
    initial=WORK/'bn-affine-parent-v1/bn-affine-parent.keras'
    retained=RUN/'retained-epochs/epoch-01.keras'
    initial_sha,retained_sha=sha(initial.read_bytes()),sha(retained.read_bytes())
    assert initial_sha=='6d4847f3030ec40476a2e29c65a4a0941f6ea9ba7e59260b20c3c6e9d48bf35d'
    config_bytes=(RUN/'training-config.json').read_bytes()
    config=json.loads(config_bytes)
    config_sha=sha(config_bytes)
    assert config_sha.startswith('acba0401') and config_sha.endswith('29bc')
    assert config['resume_sha256']==initial_sha
    sources=config['source_sha256']
    assert len(sources)==17
    for name,expected in sources.items():assert sha((RUN/'source'/name).read_bytes())==expected
    history_bytes=(RUN/'validation-history.json').read_bytes()
    row=next(record for record in json.loads(history_bytes) if record['epoch']==1)
    expected_moments=config['batchnorm_affine']['moving_moments_sha256']
    assert row['batchnorm_moving_moments_sha256']==expected_moments=='1f190a8fe5f4b3f7162147f7cf254a9fc437dfd20e0b34f3603003fffdc00ae7'
    before,metadata,initial_aux,moving_before=inventory(initial)
    after,after_metadata,trained_aux,moving_after=inventory(retained)
    assert metadata==after_metadata and before.keys()==after.keys()
    assert moving_before==moving_after==expected_moments
    rows=[]
    for name in sorted(before):
        old,new=before[name],after[name]
        identical=old.tobytes()==new.tobytes()
        rows.append({'path':name,**metadata[name],'identical_bytes':identical,
                     'initial_sha256':sha(old.tobytes()),'retained_sha256':sha(new.tobytes()),
                     'changed_values':int(np.count_nonzero(old!=new)),
                     'max_abs_change':float(np.max(np.abs(new.astype(np.float64)-old.astype(np.float64))))})
    counts={category:{'arrays':sum(row['category']==category for row in rows),
                      'changed_arrays':sum(row['category']==category and not row['identical_bytes'] for row in rows),
                      'changed_values':sum(row['changed_values'] for row in rows if row['category']==category)}
            for category in ('moving_moments','bn_affine','backbone_convolution','fpn_and_output_head')}
    assert counts['moving_moments']['arrays']==104 and counts['moving_moments']['changed_arrays']==0
    assert counts['bn_affine']['arrays']==104 and counts['bn_affine']['changed_arrays']==104
    assert counts['backbone_convolution']['changed_arrays']>0 and counts['fpn_and_output_head']['changed_arrays']>0
    assert sha(initial.read_bytes())==initial_sha and sha(retained.read_bytes())==retained_sha
    assert (RUN/'training-config.json').read_bytes()==config_bytes
    assert next(record for record in json.loads((RUN/'validation-history.json').read_bytes()) if record['epoch']==1)==row
    assert not any(name in sys.modules for name in ('keras','tensorflow','torch'))
    OUT.mkdir()
    (OUT/'validation-history-snapshot.json').write_bytes(history_bytes)
    report={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
            'scope':'Read-only actual epoch1 model-array HDF5 inventory; no model loading, inference, training or checkpoint selection.',
            'initial_checkpoint':str(initial),'initial_checkpoint_sha256':initial_sha,
            'retained_checkpoint':str(retained),'retained_checkpoint_sha256':retained_sha,
            'training_config_sha256':config_sha,'training_source_sha256':sources,
            'callback_history_snapshot_sha256':sha(history_bytes),'callback_epoch1':row,
            'moving_moments_sha256':moving_before,'moving_digest_method':'BN layer order from serialized Keras config; JSON[layer.name,variable,shape] plus little-endian float32 C bytes, matching the archived callback.',
            'model_array_count':len(rows),'counts':counts,'arrays':rows,
            'optimizer_and_other_nonmodel_arrays':{'initial':initial_aux,'retained':trained_aux},
            'script_sha256':sha(Path(__file__).read_bytes()),'h5py_version':h5py.__version__,'numpy_version':np.__version__,
            'backend_modules_imported':[],'epoch_is_sampling_eligible':False,
            'limitations':'Only epoch1 stored arrays are audited. No activation parity, accuracy, optimizer correctness, later epochs, held405, external607 or physical-camera result is established. Epoch1 is not sampling-eligible; training protocol retains eligibility after full-cycle epochs8-10.'}
    target=OUT/'report.json';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(target),'sha256':sha(target.read_bytes()),'checkpoint_sha256':retained_sha,'config_sha256':config_sha,'counts':counts,'moving_hash':moving_before},indent=2))


if __name__=='__main__':main()
