"""One fixed, inference-only checkpoint average. No Keras/Torch/TF imports."""
from datetime import datetime, timezone
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import h5py
import numpy as np
from prepare import HERE, W, EXPECTED, EPOCHS, sha, digest, canonical, load, link, frozen_json, serialized

def mean_arrays(arrays):
    if len(arrays)!=3 or any(a.dtype!=np.dtype('float32') or a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Exactly three same-shape finitefloat32 arrays required')
    total=np.zeros(arrays[0].shape,np.float64)
    for array in arrays:total+=array.astype(np.float64)
    return (total/3.).astype(np.float32)

def main():
    plan=load(HERE/'plan.json');assert sha(__file__)==plan['source_average']['sha256']
    assert sha(HERE/'prepare.py')==plan['source_prepare']['sha256']
    assert sha(HERE/'inventory.json')==plan['inventory']['sha256']
    assert plan['input_epoch_order']==list(EPOCHS) and len(plan['inputs'])==3
    candidate=Path(plan['candidate_path']);assert candidate==HERE/'uniform-08-09-10.keras' and not candidate.exists()
    for row,epoch,digest_expected in zip(plan['inputs'],EPOCHS,EXPECTED,strict=True):
        assert row['epoch']==epoch and row['weight']=={'numerator':1,'denominator':3}
        assert sha(row['path'])==row['sha256']==digest_expected
    models=[serialized(row['path']) for row in plan['inputs']]
    assert all(digest(canonical(m[1]))==plan['model_graph_sha256'] for m in models)
    inventory=load(HERE/'inventory.json')['model_arrays'];assert len(inventory)==280
    base_zip=ZipFile(plan['inputs'][1]['path'])
    weights=io.BytesIO(base_zip.read('model.weights.h5'))
    output_hashes={}; exact_bn=0; scalar_oracle_checks=0; max_delta=0.
    with h5py.File(weights,'r+') as h5:
        assert 'optimizer' in h5;del h5['optimizer']
        for row in inventory:
            path=row['h5_path'];arrays=[m[2][path] for m in models]
            for arr,d in zip(arrays,row['input_sha256'],strict=True):assert digest(arr.tobytes())==d
            if row['preserve_bn']:
                assert arrays[0].tobytes()==arrays[1].tobytes()==arrays[2].tobytes()
                value=arrays[1];exact_bn+=1
            else:
                value=mean_arrays(arrays)
                # Separate rational scalar oracle at deterministic endpoints/middle;
                # no model outputs or fixture labels determine these positions.
                for index in sorted({0,value.size//2,value.size-1}):
                    exact=sum((Fraction.from_float(float(arr.ravel()[index])) for arr in arrays),Fraction())/3
                    expected=np.float32(float(exact))
                    assert value.ravel()[index].tobytes()==expected.tobytes();scalar_oracle_checks+=1
                max_delta=max(max_delta,float(np.max(np.abs(value.astype(np.float64)-arrays[1]))))
            assert value.dtype==arrays[1].dtype and value.shape==arrays[1].shape and np.isfinite(value).all()
            h5[path][...]=value;output_hashes[path]=digest(value.tobytes())
        assert exact_bn==208
    config=models[1][0];config['compile_config']={}
    metadata=json.loads(base_zip.read('metadata.json'))
    metadata['date_saved']=datetime.now(timezone.utc).isoformat()
    metadata['brailleme_candidate']='untested_inference_only_uniform_run20_08_09_10_do_not_resume'
    with ZipFile(candidate,'x',compression=ZIP_DEFLATED) as out:
        out.writestr('metadata.json',json.dumps(metadata))
        out.writestr('config.json',json.dumps(config))
        out.writestr('model.weights.h5',weights.getvalue())
    reloaded=serialized(candidate)
    assert reloaded[0]['compile_config']=={} and reloaded[1]==models[1][1] and not reloaded[4]
    assert set(reloaded[2])==set(models[1][2]) and len(reloaded[2])==280
    for path,array in reloaded[2].items():assert digest(array.tobytes())==output_hashes[path]
    assert all(sha(row['path'])==row['sha256'] for row in plan['inputs'])
    assert not any(name in sys.modules for name in ('keras','torch','tensorflow'))
    report={'status':'built_untested','created_at_utc':datetime.now(timezone.utc).isoformat(),'plan':link(HERE/'plan.json'),
        'script':link(__file__),'candidate':link(candidate),'inputs':plan['inputs'],'model_graph_sha256':plan['model_graph_sha256'],
        'model_arrays':280,'bn_arrays_preserved_exact':exact_bn,'arrays_uniform_averaged':72,'optimizer_datasets_before':146,'optimizer_datasets_after':0,
        'compile_config_empty':True,'hdf5_roundtrip_all_array_hashes_exact':True,'original_checkpoints_unchanged':True,
        'rational_scalar_oracle_checks':scalar_oracle_checks,'maximum_weight_delta_vs_epoch09':max_delta,'array_sha256':output_hashes,
        'no_model_execution':True,'no_fixture_pixels_or_labels_read':True,'evaluation_pending':True,'training_resume_allowed':False,
        'limits':['Eligibility inherited from allthree independently eligible ingredients; no invented optimizerepoch/checkpoint sampling row.',
                  'Graph/HDF5 checks do not substitute for actual Keras load or output/conversion parity.','No recognition or confidence calibration claim.']}
    frozen_json(HERE/'candidate-report.json',report)
    print(json.dumps({'candidate_sha256':sha(candidate),'candidate_bytes':candidate.stat().st_size,'report_sha256':sha(HERE/'candidate-report.json'),
                      'scalar_oracle_checks':scalar_oracle_checks,'max_weight_delta':max_delta}))
if __name__=='__main__':main()
